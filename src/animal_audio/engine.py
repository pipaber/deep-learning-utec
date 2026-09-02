"""Experiment orchestration for the NDDR-MTL animal-audio model.

This module is intentionally side-effect free on import. Training, evaluation,
and prediction start only when their corresponding functions are called.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig, load_config, save_config
from .data import (
    FILENAME_COLUMN,
    SPLIT_COLUMN,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    AnimalAudioDataset,
    SplitLabelSupport,
    create_group_aware_split,
    label_columns,
    load_metadata,
    load_split,
    save_split,
    summarize_split_label_support,
)
from .features import AudioFeatureExtractor
from .metrics import (
    compute_multilabel_metrics,
    metrics_by_concurrency,
    metrics_to_jsonable,
    optimize_per_class_thresholds,
    plot_precision_recall_curves,
    plot_prevalence_vs_ap,
    save_metrics_json,
    zero_vector_baseline_metrics,
)
from .model import NDDRMTL, build_model

PathLike: TypeAlias = str | Path
ConfigLike: TypeAlias = ExperimentConfig | PathLike
DeviceLike: TypeAlias = str | torch.device
DeviceChoice = Literal["auto", "cpu", "cuda"]


class EngineError(RuntimeError):
    """Raised when experiment artifacts or runtime settings are inconsistent."""


@dataclass(frozen=True, slots=True)
class PreparedSplit:
    """Validated source metadata and its group-disjoint train/validation split."""

    metadata: pd.DataFrame
    split_metadata: pd.DataFrame
    train_metadata: pd.DataFrame
    validation_metadata: pd.DataFrame
    labels: tuple[str, ...]
    split_path: Path
    label_support: SplitLabelSupport


@dataclass(frozen=True, slots=True)
class DataLoaders:
    """Data loaders and the metadata used to construct them."""

    train: DataLoader[Any]
    validation: DataLoader[Any]
    test: DataLoader[Any] | None
    prepared: PreparedSplit
    test_metadata: pd.DataFrame | None


@dataclass(frozen=True, slots=True)
class PosWeightInfo:
    """Class support and the positive weights derived from training metadata."""

    labels: tuple[str, ...]
    positive_counts: tuple[int, ...]
    negative_counts: tuple[int, ...]
    raw_weights: tuple[float, ...]
    weights: tuple[float, ...]
    supported_mask: tuple[bool, ...]
    unsupported_classes: tuple[str, ...]

    @property
    def supported_classes(self) -> tuple[str, ...]:
        return tuple(
            label
            for label, supported in zip(self.labels, self.supported_mask, strict=True)
            if supported
        )


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Batch-accumulated model outputs in stable loader order."""

    logits: np.ndarray
    probabilities: np.ndarray
    targets: np.ndarray | None
    filenames: tuple[str, ...]
    mean_loss: float | None


@dataclass(frozen=True, slots=True)
class LoadedExperiment:
    """A model/frontend pair restored from one checkpoint."""

    model: NDDRMTL
    feature_extractor: AudioFeatureExtractor
    checkpoint: dict[str, Any]
    labels: tuple[str, ...]
    device: torch.device


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    """Paths and best score produced by an explicit training run."""

    output_dir: Path
    best_checkpoint: Path | None
    last_checkpoint: Path | None
    history_csv: Path
    history_json: Path
    config_snapshot: Path
    best_map: float
    epochs_completed: int
    stopped_early: bool


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Evaluation artifacts produced from a validation split."""

    metrics_path: Path
    thresholds_json: Path
    thresholds_csv: Path
    validation_probabilities_csv: Path
    precision_recall_plot: Path
    prevalence_ap_plot: Path
    map_score: float
    fixed_micro_f1: float
    optimized_micro_f1: float
    fixed_exact_match: float
    optimized_exact_match: float


@dataclass(frozen=True, slots=True)
class PredictionSummary:
    """Prediction table paths and the thresholds used to create them."""

    probabilities_csv: Path
    predictions_csv: Path
    filenames: tuple[str, ...]
    labels: tuple[str, ...]
    thresholds: tuple[float, ...]


def seed_everything(seed: int, *, deterministic: bool = True) -> int:
    """Seed Python, NumPy, Torch, and all CUDA devices deterministically."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    numpy_seed = seed % (2**32)
    torch_seed = seed % (2**63)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)

    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    return seed


def resolve_device(device: DeviceLike = "auto") -> torch.device:
    """Resolve ``auto``, ``cpu``, or ``cuda`` and reject unavailable CUDA."""

    requested = str(device).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise EngineError("CUDA was requested but is not available")
        return torch.device("cuda")
    raise ValueError("device must be one of: auto, cpu, cuda")


def _resolved_config(config: ConfigLike) -> ExperimentConfig:
    if isinstance(config, ExperimentConfig):
        return config.resolved(Path.cwd())
    return load_config(config)


def _serialize_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _serialize_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_serialize_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _serialize_value(value.item())
    return value


def serialize_config(config: ExperimentConfig) -> dict[str, Any]:
    """Serialize a resolved config into checkpoint-safe built-in values."""

    resolved = config.resolved(Path.cwd())
    return cast(dict[str, Any], _serialize_value(resolved))


def _validate_audio_files(metadata: pd.DataFrame, audio_directory: Path) -> None:
    if not audio_directory.is_dir():
        raise EngineError(f"audio directory does not exist: {audio_directory}")

    invalid_extensions: list[str] = []
    missing: list[str] = []
    for filename in metadata[FILENAME_COLUMN].astype(str):
        if Path(filename).suffix.lower() != ".wav":
            invalid_extensions.append(filename)
        if not (audio_directory / filename).is_file():
            missing.append(filename)

    if invalid_extensions:
        preview = invalid_extensions[:5]
        raise EngineError(f"metadata contains non-WAV filenames: {preview}")
    if missing:
        preview = missing[:5]
        suffix = "" if len(missing) <= 5 else f" (and {len(missing) - 5} more)"
        raise EngineError(
            f"{len(missing)} metadata audio file(s) are missing from "
            f"{audio_directory}: {preview}{suffix}"
        )


def _validate_existing_split(
    source: pd.DataFrame,
    split_metadata: pd.DataFrame,
    labels: Sequence[str],
    split_path: Path,
) -> None:
    split_labels = label_columns(split_metadata)
    if split_labels != list(labels):
        raise EngineError(
            f"split label columns/order do not match source metadata in {split_path}; "
            "recreate the split"
        )

    source_filenames = source[FILENAME_COLUMN].astype(str).tolist()
    split_filenames = split_metadata[FILENAME_COLUMN].astype(str).tolist()
    if split_filenames != source_filenames:
        source_set = set(source_filenames)
        split_set = set(split_filenames)
        missing = sorted(source_set - split_set)[:5]
        extra = sorted(split_set - source_set)[:5]
        detail = f"missing={missing}, extra={extra}"
        if not missing and not extra:
            detail = "the filenames are in a different order"
        raise EngineError(
            f"split filenames do not match source metadata in {split_path}: {detail}; "
            "recreate the split"
        )

    source_targets = source[list(labels)].to_numpy(dtype=np.int8, copy=False)
    split_targets = split_metadata[list(labels)].to_numpy(dtype=np.int8, copy=False)
    if not np.array_equal(source_targets, split_targets):
        raise EngineError(
            f"split labels are stale relative to source metadata in {split_path}; "
            "recreate the split"
        )


def extract_configured_archives(
    config: ConfigLike,
    *,
    force: bool = False,
) -> dict[str, str]:
    """Extract configured 7z archives with the system 7-Zip executable.

    Extraction is skipped when the target already contains WAV files unless
    ``force`` is set. Archives are extracted into the target directory's parent,
    matching the provided ``train.7z``/``test.7z`` root-folder layout.
    """

    resolved = _resolved_config(config)
    executable = next(
        (path for name in ("7z", "7zz", "7za") if (path := shutil.which(name))),
        None,
    )
    if executable is None:
        raise EngineError(
            "7-Zip was not found; install p7zip/7zip or extract train.7z and "
            "test.7z manually"
        )

    results: dict[str, str] = {}
    for name, archive, target in (
        ("train", resolved.data.train_archive, resolved.data.train_dir),
        ("test", resolved.data.test_archive, resolved.data.test_dir),
    ):
        if target.is_dir() and any(target.glob("*.wav")) and not force:
            results[name] = f"skipped: {target} already contains WAV files"
            continue
        if archive is None:
            raise EngineError(f"data.{name}_archive is not configured")
        if not archive.is_file():
            raise EngineError(f"configured {name} archive does not exist: {archive}")
        target.parent.mkdir(parents=True, exist_ok=True)
        command = [
            executable,
            "x",
            str(archive),
            f"-o{target.parent}",
            "-y",
            "-aoa" if force else "-aos",
        ]
        try:
            completed = subprocess.run(command, check=False)
        except OSError as error:
            raise EngineError(f"could not execute 7-Zip: {error}") from error
        if completed.returncode != 0:
            raise EngineError(
                f"7-Zip failed for {archive} with exit code {completed.returncode}"
            )
        if not target.is_dir() or not any(target.glob("*.wav")):
            raise EngineError(
                f"archive extraction completed but no WAV files were found in {target}"
            )
        results[name] = f"extracted: {archive} -> {target}"
    return results


def prepare_split(config: ConfigLike, *, force: bool = False) -> PreparedSplit:
    """Create or load the deterministic group-aware 80/20 split and validate WAVs.

    Existing split artifacts are accepted only when filename rows, label values,
    and label order exactly match the current source metadata. New splits always
    use the configured reproducibility seed (42 in the provided configs).
    """

    resolved = _resolved_config(config)
    metadata = load_metadata(
        resolved.data.metadata_csv,
        expected_label_count=resolved.model.num_classes,
        require_labels=True,
    )
    labels = tuple(label_columns(metadata))
    if len(labels) != resolved.model.num_classes:
        raise EngineError(
            f"metadata has {len(labels)} labels but the model expects "
            f"{resolved.model.num_classes}"
        )

    split_path = resolved.data.split_csv
    if split_path.is_file() and not force:
        split_metadata = load_split(
            split_path,
            expected_label_count=resolved.model.num_classes,
        )
        _validate_existing_split(metadata, split_metadata, labels, split_path)
    else:
        split_metadata = create_group_aware_split(
            metadata,
            val_fraction=0.2,
            seed=resolved.training.seed,
            labels=labels,
        )
        save_split(split_metadata, split_path)

    label_support = summarize_split_label_support(
        split_metadata,
        labels=labels,
    )
    if label_support.train_unsupported:
        raise EngineError(
            "training split has no positives for globally supported classes: "
            f"{list(label_support.train_unsupported)}; recreate the split"
        )

    _validate_audio_files(split_metadata, resolved.data.train_dir)
    train_metadata = split_metadata.loc[
        split_metadata[SPLIT_COLUMN].eq(TRAIN_SPLIT)
    ].reset_index(drop=True)
    validation_metadata = split_metadata.loc[
        split_metadata[SPLIT_COLUMN].eq(VALIDATION_SPLIT)
    ].reset_index(drop=True)
    return PreparedSplit(
        metadata=metadata,
        split_metadata=split_metadata,
        train_metadata=train_metadata,
        validation_metadata=validation_metadata,
        labels=labels,
        split_path=split_path,
        label_support=label_support,
    )


def create_test_metadata(test_directory: PathLike) -> pd.DataFrame:
    """Build filename-only test metadata from lexicographically sorted WAVs."""

    directory = Path(test_directory).expanduser().resolve()
    if not directory.is_dir():
        raise EngineError(f"test audio directory does not exist: {directory}")
    filenames = [path.name for path in sorted(directory.glob("*.wav"), key=lambda p: p.name)]
    if not filenames:
        raise EngineError(f"test audio directory contains no *.wav files: {directory}")
    return pd.DataFrame({FILENAME_COLUMN: filenames})


def _seed_data_loader_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _make_loader(
    dataset: AnimalAudioDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader[Any]:
    generator = torch.Generator()
    generator.manual_seed(seed % (2**63))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=_seed_data_loader_worker if num_workers > 0 else None,
        generator=generator,
        drop_last=False,
    )


def build_dataloaders(
    config: ConfigLike,
    *,
    prepared: PreparedSplit | None = None,
    include_test: bool = True,
) -> DataLoaders:
    """Construct deterministic train, validation, and optionally test loaders."""

    resolved = _resolved_config(config)
    split = prepared if prepared is not None else prepare_split(resolved)
    if len(split.labels) != resolved.model.num_classes:
        raise EngineError("prepared split does not match the configured class count")

    train_dataset = AnimalAudioDataset(
        split.train_metadata,
        resolved.data.train_dir,
        labels=split.labels,
        sample_rate=resolved.feature.sample_rate,
        duration_seconds=resolved.feature.duration_seconds,
    )
    validation_dataset = AnimalAudioDataset(
        split.validation_metadata,
        resolved.data.train_dir,
        labels=split.labels,
        sample_rate=resolved.feature.sample_rate,
        duration_seconds=resolved.feature.duration_seconds,
    )

    common = {
        "batch_size": resolved.training.batch_size,
        "num_workers": resolved.training.num_workers,
        "pin_memory": resolved.training.pin_memory,
    }
    train_loader = _make_loader(
        train_dataset,
        shuffle=True,
        seed=resolved.training.seed,
        **common,
    )
    validation_loader = _make_loader(
        validation_dataset,
        shuffle=False,
        seed=resolved.training.seed + 1,
        **common,
    )

    test_metadata: pd.DataFrame | None = None
    test_loader: DataLoader[Any] | None = None
    if include_test:
        test_metadata = create_test_metadata(resolved.data.test_dir)
        test_dataset = AnimalAudioDataset(
            test_metadata,
            resolved.data.test_dir,
            labels=split.labels,
            has_targets=False,
            sample_rate=resolved.feature.sample_rate,
            duration_seconds=resolved.feature.duration_seconds,
        )
        test_loader = _make_loader(
            test_dataset,
            shuffle=False,
            seed=resolved.training.seed + 2,
            **common,
        )

    return DataLoaders(
        train=train_loader,
        validation=validation_loader,
        test=test_loader,
        prepared=split,
        test_metadata=test_metadata,
    )


def calculate_pos_weight(
    training_metadata: pd.DataFrame,
    labels: Sequence[str] | None = None,
    *,
    minimum: float = 1.0,
    maximum: float = 20.0,
    device: DeviceLike = "cpu",
) -> tuple[torch.Tensor, PosWeightInfo]:
    """Calculate clamped ``negative / positive`` BCE weights and support info.

    A class with no positive training examples always receives weight ``1.0``,
    even when the configured lower clamp is greater than one.
    """

    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError("positive-weight bounds must be finite")
    if minimum <= 0.0 or maximum < minimum:
        raise ValueError("positive-weight bounds must satisfy 0 < minimum <= maximum")

    selected = tuple(labels) if labels is not None else tuple(label_columns(training_metadata))
    if not selected:
        raise ValueError("at least one label is required to calculate pos_weight")
    missing = [label for label in selected if label not in training_metadata.columns]
    if missing:
        raise ValueError(f"training metadata is missing label columns: {missing}")

    target_frame = training_metadata[list(selected)].apply(pd.to_numeric, errors="coerce")
    target_array = target_frame.to_numpy(dtype=np.float64, copy=True)
    if not np.all(np.isfinite(target_array)) or not np.all(
        (target_array == 0.0) | (target_array == 1.0)
    ):
        raise ValueError("training labels must contain only binary 0/1 values")

    positives = target_array.sum(axis=0, dtype=np.float64)
    negatives = len(target_array) - positives
    supported = positives > 0
    raw_weights = np.ones(len(selected), dtype=np.float64)
    np.divide(negatives, positives, out=raw_weights, where=supported)
    weights = np.clip(raw_weights, minimum, maximum)
    weights[~supported] = 1.0

    resolved_device = resolve_device(device)
    tensor = torch.tensor(weights, dtype=torch.float32, device=resolved_device)
    info = PosWeightInfo(
        labels=selected,
        positive_counts=tuple(int(value) for value in positives),
        negative_counts=tuple(int(value) for value in negatives),
        raw_weights=tuple(float(value) for value in raw_weights),
        weights=tuple(float(value) for value in weights),
        supported_mask=tuple(bool(value) for value in supported),
        unsupported_classes=tuple(
            label
            for label, is_supported in zip(selected, supported, strict=True)
            if not is_supported
        ),
    )
    return tensor, info


def _validate_nddr_configuration(config: ExperimentConfig) -> None:
    if config.feature.n_mels != NDDRMTL.input_frequency_bins:
        raise EngineError(
            f"NDDRMTL requires {NDDRMTL.input_frequency_bins} mel bins, "
            f"configured {config.feature.n_mels}"
        )
    if config.model.conv_layers != 3 or len(config.model.pool_sizes) != 3:
        raise EngineError("NDDRMTL requires exactly three convolutional stages")
    if tuple(config.model.kernel_size) != (5, 5):
        raise EngineError("NDDRMTL uses a fixed 5x5 convolution kernel")
    if not config.model.bidirectional:
        raise EngineError("NDDRMTL uses bidirectional GRU layers")


def build_experiment_components(
    config: ConfigLike,
    device: DeviceLike = "auto",
) -> tuple[NDDRMTL, AudioFeatureExtractor, torch.device]:
    """Build the configured NDDR model and fixed audio frontend on one device."""

    resolved = _resolved_config(config)
    _validate_nddr_configuration(resolved)
    resolved_device = resolve_device(device)
    model = build_model(resolved.model.to_model_kwargs()).to(resolved_device)
    feature = resolved.feature
    feature_extractor = AudioFeatureExtractor(
        sample_rate=feature.sample_rate,
        n_fft=feature.n_fft,
        win_length=feature.win_length,
        hop_length=feature.hop_length,
        n_mels=feature.n_mels,
        f_min=feature.f_min,
        f_max=feature.f_max,
        mode=cast(Literal["pcen", "logmel"], feature.kind),
        offset=feature.pcen_bias,
        gain=feature.pcen_gain,
        power=feature.pcen_power,
        eps=feature.pcen_eps,
        smoothing=feature.pcen_smoothing,
        repeat_channels=feature.repeat_channels,
        expected_num_samples=round(feature.sample_rate * feature.duration_seconds),
    ).to(resolved_device)
    feature_extractor.eval()
    feature_extractor.requires_grad_(False)
    return model, feature_extractor, resolved_device


def extract_fixed_features(
    feature_extractor: AudioFeatureExtractor,
    waveforms: torch.Tensor,
    device: DeviceLike,
) -> torch.Tensor:
    """Extract detached float32 features with autocast explicitly disabled."""

    resolved_device = resolve_device(device)
    with torch.no_grad():
        with torch.autocast(device_type=resolved_device.type, enabled=False):
            inputs = waveforms.to(
                device=resolved_device,
                dtype=torch.float32,
                non_blocking=True,
            )
            features = feature_extractor(inputs)
    return features.to(dtype=torch.float32)


def _amp_enabled(device: torch.device, requested: bool) -> bool:
    return bool(requested and device.type == "cuda")


def run_inference(
    model: NDDRMTL,
    feature_extractor: AudioFeatureExtractor,
    loader: DataLoader[Any],
    *,
    device: DeviceLike,
    criterion: nn.Module | None = None,
    amp: bool = False,
) -> InferenceResult:
    """Accumulate logits, probabilities, optional targets, names, and mean loss."""

    resolved_device = resolve_device(device)
    model.eval()
    feature_extractor.eval()
    use_amp = _amp_enabled(resolved_device, amp)
    logit_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    filenames: list[str] = []
    total_loss = 0.0
    loss_samples = 0
    target_presence: bool | None = None

    with torch.no_grad():
        for batch in loader:
            waveforms = cast(torch.Tensor, batch["waveform"])
            features = extract_fixed_features(
                feature_extractor,
                waveforms,
                resolved_device,
            )
            with torch.autocast(
                device_type=resolved_device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(features)
                batch_targets = batch.get("target")
                loss = None
                if criterion is not None and batch_targets is not None:
                    targets_for_loss = cast(torch.Tensor, batch_targets).to(
                        device=resolved_device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )
                    loss = criterion(logits, targets_for_loss)

            logits_float = logits.detach().float().cpu()
            logit_batches.append(logits_float.numpy())
            probability_batches.append(torch.sigmoid(logits_float).numpy())
            batch_size = logits.shape[0]
            if loss is not None:
                total_loss += float(loss.detach().float()) * batch_size
                loss_samples += batch_size

            has_targets = batch_targets is not None
            if target_presence is None:
                target_presence = has_targets
            elif target_presence != has_targets:
                raise EngineError("loader batches inconsistently include targets")
            if batch_targets is not None:
                target_batches.append(cast(torch.Tensor, batch_targets).detach().cpu().numpy())

            batch_filenames = batch["filename"]
            if isinstance(batch_filenames, str):
                filenames.append(batch_filenames)
            else:
                filenames.extend(str(name) for name in batch_filenames)

    if not logit_batches:
        raise EngineError("cannot run inference on an empty data loader")

    logits_array = np.concatenate(logit_batches, axis=0).astype(np.float32, copy=False)
    probabilities_array = np.concatenate(probability_batches, axis=0).astype(
        np.float32, copy=False
    )
    targets_array = (
        np.concatenate(target_batches, axis=0).astype(np.float32, copy=False)
        if target_batches
        else None
    )
    if len(filenames) != logits_array.shape[0]:
        raise EngineError("inference filename count does not match output rows")
    return InferenceResult(
        logits=logits_array,
        probabilities=probabilities_array,
        targets=targets_array,
        filenames=tuple(filenames),
        mean_loss=(total_loss / loss_samples if loss_samples else None),
    )


def _atomic_torch_save(payload: Mapping[str, Any], path: PathLike) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def save_checkpoint(
    path: PathLike,
    *,
    epoch: int,
    model: NDDRMTL,
    labels: Sequence[str],
    config: ExperimentConfig,
    best_map: float,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: Any | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save a complete, portable experiment checkpoint."""

    normalized_labels = tuple(str(label) for label in labels)
    if len(normalized_labels) != model.num_classes:
        raise ValueError("checkpoint label count does not match the model")
    if len(set(normalized_labels)) != len(normalized_labels):
        raise ValueError("checkpoint labels must be unique")
    payload: dict[str, Any] = {
        "format_version": 2,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_map": float(best_map),
        "best_mAP": float(best_map),
        "labels": list(normalized_labels),
        "config": serialize_config(config),
    }
    if extra:
        reserved = set(payload).intersection(extra)
        if reserved:
            raise ValueError(f"checkpoint extra values overwrite reserved keys: {sorted(reserved)}")
        payload.update(dict(extra))
    return _atomic_torch_save(payload, path)


def load_checkpoint(
    path: PathLike,
    *,
    device: DeviceLike = "cpu",
) -> dict[str, Any]:
    """Load and minimally validate a checkpoint produced by this engine."""

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise EngineError(f"checkpoint does not exist: {checkpoint_path}")
    resolved_device = resolve_device(device)
    try:
        loaded = torch.load(
            checkpoint_path,
            map_location=resolved_device,
            weights_only=False,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise EngineError(f"could not load checkpoint {checkpoint_path}: {error}") from error
    if not isinstance(loaded, dict):
        raise EngineError(f"checkpoint is not a mapping: {checkpoint_path}")
    required = {"epoch", "model_state_dict", "labels"}
    missing = sorted(required - set(loaded))
    if missing:
        raise EngineError(f"checkpoint is missing required keys: {missing}")
    if not isinstance(loaded["labels"], (list, tuple)):
        raise EngineError("checkpoint labels must be a list")
    normalized_labels = tuple(str(label) for label in loaded["labels"])
    if not normalized_labels or len(set(normalized_labels)) != len(normalized_labels):
        raise EngineError("checkpoint labels must be non-empty and unique")
    return cast(dict[str, Any], loaded)


def _validate_checkpoint_config(
    checkpoint: Mapping[str, Any],
    config: ExperimentConfig,
) -> None:
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is None:
        return
    if not isinstance(checkpoint_config, Mapping):
        raise EngineError("checkpoint config metadata must be a mapping")
    current_config = serialize_config(config)
    for section in ("feature", "model"):
        saved_section = checkpoint_config.get(section)
        if saved_section is not None and saved_section != current_config[section]:
            raise EngineError(
                f"checkpoint {section} configuration does not match the supplied config"
            )


def load_experiment_checkpoint(
    config: ConfigLike,
    checkpoint_path: PathLike | None = None,
    *,
    device: DeviceLike = "auto",
) -> LoadedExperiment:
    """Construct an experiment and restore its model state strictly."""

    resolved = _resolved_config(config)
    path = (
        Path(checkpoint_path).expanduser().resolve()
        if checkpoint_path is not None
        else (resolved.output_dir / resolved.training.checkpoint_filename).resolve()
    )
    model, feature_extractor, resolved_device = build_experiment_components(
        resolved, device
    )
    checkpoint = load_checkpoint(path, device=resolved_device)
    _validate_checkpoint_config(checkpoint, resolved)
    labels = tuple(str(label) for label in checkpoint["labels"])
    if len(labels) != resolved.model.num_classes:
        raise EngineError(
            f"checkpoint has {len(labels)} labels but the model expects "
            f"{resolved.model.num_classes}"
        )
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as error:
        raise EngineError(f"checkpoint model state is incompatible: {error}") from error
    return LoadedExperiment(
        model=model,
        feature_extractor=feature_extractor,
        checkpoint=checkpoint,
        labels=labels,
        device=resolved_device,
    )


def _atomic_write_text(path: Path, text: str) -> Path:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def _atomic_write_dataframe(frame: pd.DataFrame, path: Path) -> Path:
    return _atomic_write_text(path, frame.to_csv(index=False))


def _save_config_snapshot(config: ExperimentConfig, path: Path) -> Path:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(
        f".{destination.stem}.{os.getpid()}.tmp{destination.suffix}"
    )
    try:
        save_config(config, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def _history_from_disk(path: Path, *, before_epoch: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EngineError(f"could not load training history {path}: {error}") from error
    if not isinstance(loaded, list) or not all(isinstance(row, dict) for row in loaded):
        raise EngineError(f"training history must be a JSON list of objects: {path}")
    return [row for row in loaded if int(row.get("epoch", before_epoch)) < before_epoch]


def _save_history(
    history: Sequence[Mapping[str, Any]],
    csv_path: Path,
    json_path: Path,
) -> None:
    frame = pd.DataFrame(history)
    _atomic_write_dataframe(frame, csv_path)
    safe = metrics_to_jsonable({"history": list(history)})["history"]
    _atomic_write_text(
        json_path,
        json.dumps(safe, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )


def _cpu_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Clone a portable CPU snapshot without mutating the live model."""

    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _checkpoint_paths(config: ExperimentConfig) -> tuple[Path, Path]:
    best = (config.output_dir / config.training.checkpoint_filename).resolve()
    last = (config.output_dir / "last.pt").resolve()
    if best == last:
        raise EngineError("training.checkpoint_filename must differ from 'last.pt'")
    return best, last


def _restore_training_state(
    checkpoint: Mapping[str, Any],
    *,
    labels: Sequence[str],
    model: NDDRMTL,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
) -> tuple[int, float, float, int]:
    checkpoint_labels = tuple(str(label) for label in checkpoint["labels"])
    if checkpoint_labels != tuple(labels):
        raise EngineError("resume checkpoint label names/order do not match the split")
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer_state = checkpoint.get("optimizer_state_dict")
        scheduler_state = checkpoint.get("scheduler_state_dict")
        scaler_state = checkpoint.get("scaler_state_dict")
        if optimizer_state is None or scheduler_state is None or scaler_state is None:
            raise EngineError("resume checkpoint does not contain complete training state")
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)
        scaler.load_state_dict(scaler_state)
    except EngineError:
        raise
    except (KeyError, RuntimeError, ValueError) as error:
        raise EngineError(f"could not restore training state: {error}") from error

    epoch = int(checkpoint["epoch"])
    best_map = float(checkpoint.get("best_map", checkpoint.get("best_mAP", -math.inf)))
    early_best = float(checkpoint.get("early_stopping_best_map", best_map))
    stale_epochs = int(checkpoint.get("epochs_without_improvement", 0))
    return epoch + 1, best_map, early_best, stale_epochs


def train_experiment(
    config: ConfigLike,
    *,
    device: DeviceLike = "auto",
) -> TrainingSummary:
    """Explicitly train the configured NDDR-MTL model.

    This is the engine's only training entry point. It is never invoked during
    import, preparation, inspection, evaluation, or prediction.
    """

    resolved = _resolved_config(config)
    seed_everything(resolved.training.seed)
    resolved_device = resolve_device(device)
    resolved.output_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = _save_config_snapshot(
        resolved,
        resolved.output_dir / "config.resolved.yaml",
    )

    prepared = prepare_split(resolved)
    loaders = build_dataloaders(resolved, prepared=prepared, include_test=False)
    model, feature_extractor, _ = build_experiment_components(
        resolved, resolved_device
    )

    pos_weight, _support = calculate_pos_weight(
        prepared.train_metadata,
        prepared.labels,
        minimum=resolved.training.pos_weight_min,
        maximum=resolved.training.pos_weight_max,
        device=resolved_device,
    )
    if not resolved.training.use_pos_weight:
        pos_weight = torch.ones_like(pos_weight)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    parameter_groups = [
        dict(group)
        for group in model.optimizer_parameter_groups(
            base_weight_decay=resolved.training.base_weight_decay,
            nddr_weight_decay=resolved.training.nddr_weight_decay,
        )
    ]
    optimizer = torch.optim.Adam(
        parameter_groups,
        lr=resolved.training.learning_rate,
    )
    steps_per_decay = resolved.training.lr_decay_steps or max(
        1,
        round(len(loaders.train) * resolved.training.lr_decay_every_epochs),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: resolved.training.lr_decay
        ** (step // steps_per_decay),
    )
    use_amp = _amp_enabled(resolved_device, resolved.training.amp)
    scaler = torch.amp.GradScaler(resolved_device.type, enabled=use_amp)

    best_path, last_path = _checkpoint_paths(resolved)
    history_csv = (resolved.output_dir / "history.csv").resolve()
    history_json = (resolved.output_dir / "history.json").resolve()
    start_epoch = 1
    best_map = -math.inf
    early_best_map = -math.inf
    epochs_without_improvement = 0
    best_model_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    resume_history: list[dict[str, Any]] = []
    resume_checkpoint: dict[str, Any] | None = None

    if resolved.training.resume_from is not None:
        resume_checkpoint = load_checkpoint(
            resolved.training.resume_from,
            device=resolved_device,
        )
        _validate_checkpoint_config(resume_checkpoint, resolved)
        (
            start_epoch,
            best_map,
            early_best_map,
            epochs_without_improvement,
        ) = _restore_training_state(
            resume_checkpoint,
            labels=prepared.labels,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        saved_best_state = resume_checkpoint.get("best_model_state_dict")
        if not isinstance(saved_best_state, Mapping):
            raise EngineError(
                "resume checkpoint does not contain best_model_state_dict; "
                "resume from a format-version 2 last.pt checkpoint"
            )
        best_model_state = {
            str(name): cast(torch.Tensor, value).detach().cpu().clone()
            for name, value in saved_best_state.items()
        }
        best_epoch = int(resume_checkpoint.get("best_epoch", resume_checkpoint["epoch"]))
        saved_history = resume_checkpoint.get("history", [])
        if not isinstance(saved_history, list) or not all(
            isinstance(row, dict) for row in saved_history
        ):
            raise EngineError("resume checkpoint history must be a list of objects")
        resume_history = [
            dict(row)
            for row in saved_history
            if int(row.get("epoch", start_epoch)) < start_epoch
        ]

    disk_history = _history_from_disk(history_json, before_epoch=start_epoch)
    history = disk_history if disk_history else resume_history
    stopped_early = False
    epochs_completed = start_epoch - 1
    best_checkpoint_written = False
    if (
        resolved.training.save_checkpoints
        and resume_checkpoint is not None
        and best_model_state is not None
    ):
        best_payload = dict(resume_checkpoint)
        best_payload["model_state_dict"] = best_model_state
        best_payload["epoch"] = best_epoch
        best_payload["best_map"] = best_map
        best_payload["best_mAP"] = best_map
        best_payload["history"] = history
        _atomic_torch_save(best_payload, best_path)
        best_checkpoint_written = True

    for epoch in range(start_epoch, resolved.training.epochs + 1):
        train_generator = getattr(loaders.train, "generator", None)
        if isinstance(train_generator, torch.Generator):
            train_generator.manual_seed((resolved.training.seed + epoch - 1) % (2**63))

        model.train()
        feature_extractor.eval()
        total_train_loss = 0.0
        train_samples = 0
        learning_rate = float(optimizer.param_groups[0]["lr"])

        for batch in loaders.train:
            waveforms = cast(torch.Tensor, batch["waveform"])
            targets = cast(torch.Tensor, batch["target"]).to(
                device=resolved_device,
                dtype=torch.float32,
                non_blocking=True,
            )
            features = extract_fixed_features(
                feature_extractor,
                waveforms,
                resolved_device,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=resolved_device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(features)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            if resolved.training.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), resolved.training.grad_clip_norm
                )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_size = targets.shape[0]
            total_train_loss += float(loss.detach().float()) * batch_size
            train_samples += batch_size

        if train_samples == 0:
            raise EngineError("training data loader is empty")
        train_loss = total_train_loss / train_samples
        validation = run_inference(
            model,
            feature_extractor,
            loaders.validation,
            device=resolved_device,
            criterion=criterion,
            amp=resolved.training.amp,
        )
        if validation.targets is None:
            raise EngineError("validation data loader did not provide targets")
        validation_metrics = compute_multilabel_metrics(
            validation.targets,
            probabilities=validation.probabilities,
            threshold=0.5,
            class_names=prepared.labels,
        )
        validation_map = float(validation_metrics["mAP"])
        validation_loss = cast(float, validation.mean_loss)

        improved = math.isfinite(validation_map) and (
            not math.isfinite(best_map) or validation_map > best_map
        )
        if improved:
            best_map = validation_map
            best_epoch = epoch
            best_model_state = _cpu_model_state(model)

        stopping_improved = math.isfinite(validation_map) and (
            not math.isfinite(early_best_map)
            or validation_map
            > early_best_map + resolved.training.early_stopping_min_delta
        )
        if stopping_improved:
            early_best_map = validation_map
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_mAP": validation_map,
            "lr_decay_steps": steps_per_decay,
        }
        history.append(row)
        _save_history(history, history_csv, history_json)

        checkpoint_extra = {
            "early_stopping_best_map": early_best_map,
            "epochs_without_improvement": epochs_without_improvement,
            "best_epoch": best_epoch,
            "best_model_state_dict": best_model_state,
            "history": history,
        }
        patience = resolved.training.early_stopping_patience
        should_stop = (
            patience is not None and epochs_without_improvement >= patience
        )
        if resolved.training.save_checkpoints:
            if improved or not best_checkpoint_written:
                if best_model_state is None:
                    raise EngineError("cannot save a best checkpoint without a finite mAP")
                save_checkpoint(
                    best_path,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_map=best_map if math.isfinite(best_map) else validation_map,
                    labels=prepared.labels,
                    config=resolved,
                    extra=checkpoint_extra,
                )
                best_checkpoint_written = True
            if (
                epoch % resolved.training.checkpoint_every == 0
                or epoch == resolved.training.epochs
                or should_stop
            ):
                save_checkpoint(
                    last_path,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_map=best_map,
                    labels=prepared.labels,
                    config=resolved,
                    extra=checkpoint_extra,
                )

        epochs_completed = epoch
        print(
            f"epoch {epoch}/{resolved.training.epochs} "
            f"train_loss={train_loss:.5f} val_loss={validation_loss:.5f} "
            f"val_mAP={validation_map:.5f}"
        )

        if should_stop:
            stopped_early = True
            print(f"early stopping after epoch {epoch}")
            break

    return TrainingSummary(
        output_dir=resolved.output_dir,
        best_checkpoint=(
            best_path
            if resolved.training.save_checkpoints and best_path.is_file()
            else None
        ),
        last_checkpoint=(
            last_path
            if resolved.training.save_checkpoints and last_path.is_file()
            else None
        ),
        history_csv=history_csv,
        history_json=history_json,
        config_snapshot=config_snapshot,
        best_map=best_map,
        epochs_completed=epochs_completed,
        stopped_early=stopped_early,
    )


def _pos_weight_info_dict(info: PosWeightInfo) -> dict[str, Any]:
    return {
        "labels": list(info.labels),
        "positive_counts": list(info.positive_counts),
        "negative_counts": list(info.negative_counts),
        "raw_weights": list(info.raw_weights),
        "weights": list(info.weights),
        "supported_mask": list(info.supported_mask),
        "unsupported_classes": list(info.unsupported_classes),
    }


def _save_validation_probabilities(
    result: InferenceResult,
    labels: Sequence[str],
    path: Path,
) -> Path:
    if result.targets is None:
        raise ValueError("validation result does not contain targets")
    frame = pd.DataFrame({FILENAME_COLUMN: result.filenames})
    for index, label in enumerate(labels):
        frame[str(label)] = result.probabilities[:, index]
    for index, label in enumerate(labels):
        frame[f"target_{label}"] = result.targets[:, index].astype(np.int8)
    return _atomic_write_dataframe(frame, path)


def evaluate_checkpoint(
    config: ConfigLike,
    checkpoint_path: PathLike | None = None,
    *,
    device: DeviceLike = "auto",
) -> EvaluationSummary:
    """Evaluate a checkpoint and save metrics, thresholds, tables, and plots."""

    resolved = _resolved_config(config)
    seed_everything(resolved.training.seed)
    prepared = prepare_split(resolved)
    loaders = build_dataloaders(resolved, prepared=prepared, include_test=False)
    loaded = load_experiment_checkpoint(
        resolved,
        checkpoint_path,
        device=device,
    )
    if loaded.labels != prepared.labels:
        raise EngineError("checkpoint label names/order do not match validation metadata")

    pos_weight, support = calculate_pos_weight(
        prepared.train_metadata,
        prepared.labels,
        minimum=resolved.training.pos_weight_min,
        maximum=resolved.training.pos_weight_max,
        device=loaded.device,
    )
    if not resolved.training.use_pos_weight:
        pos_weight = torch.ones_like(pos_weight)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    result = run_inference(
        loaded.model,
        loaded.feature_extractor,
        loaders.validation,
        device=loaded.device,
        criterion=criterion,
        amp=resolved.training.amp,
    )
    if result.targets is None:
        raise EngineError("validation data loader did not provide targets")

    fixed = compute_multilabel_metrics(
        result.targets,
        probabilities=result.probabilities,
        threshold=0.5,
        class_names=loaded.labels,
    )
    threshold_optimization = optimize_per_class_thresholds(
        result.targets,
        result.probabilities,
        class_names=loaded.labels,
    )
    thresholds = cast(np.ndarray, threshold_optimization["thresholds"])
    optimized = compute_multilabel_metrics(
        result.targets,
        probabilities=result.probabilities,
        threshold=thresholds.tolist(),
        class_names=loaded.labels,
    )
    concurrency = {
        "fixed_0_5": metrics_by_concurrency(
            result.targets,
            result.probabilities,
            threshold=0.5,
            class_names=loaded.labels,
        ),
        "optimized": metrics_by_concurrency(
            result.targets,
            result.probabilities,
            threshold=thresholds.tolist(),
            class_names=loaded.labels,
        ),
    }
    zero_baseline = zero_vector_baseline_metrics(
        result.targets,
        class_names=loaded.labels,
    )

    output_dir = resolved.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    thresholds_json = output_dir / "thresholds.json"
    thresholds_csv = output_dir / "thresholds.csv"
    probabilities_csv = output_dir / "validation_probabilities.csv"
    precision_recall_plot = output_dir / "precision_recall.png"
    prevalence_ap_plot = output_dir / "prevalence_vs_ap.png"

    metrics_payload = {
        "checkpoint_epoch": int(loaded.checkpoint["epoch"]),
        "validation_loss": result.mean_loss,
        "pos_weight": _pos_weight_info_dict(support),
        "fixed_0_5": fixed,
        "optimized": optimized,
        "threshold_optimization": threshold_optimization,
        "concurrency": concurrency,
        "zero_baseline": zero_baseline,
    }
    save_metrics_json(metrics_payload, metrics_path)
    save_metrics_json(
        {
            "labels": list(loaded.labels),
            "class_names": list(loaded.labels),
            "thresholds": thresholds,
            "by_class": threshold_optimization["by_class"],
        },
        thresholds_json,
    )
    threshold_frame = pd.DataFrame(
        {
            "label": loaded.labels,
            "threshold": thresholds,
            "best_f1": threshold_optimization["best_f1"],
            "supported": threshold_optimization["supported_mask"],
        }
    )
    _atomic_write_dataframe(threshold_frame, thresholds_csv)
    _save_validation_probabilities(result, loaded.labels, probabilities_csv)

    pr_figure = plot_precision_recall_curves(
        result.targets,
        result.probabilities,
        class_names=loaded.labels,
        output_path=precision_recall_plot,
    )
    prevalence_figure = plot_prevalence_vs_ap(
        result.targets,
        result.probabilities,
        class_names=loaded.labels,
        output_path=prevalence_ap_plot,
    )
    import matplotlib.pyplot as plt

    plt.close(pr_figure)
    plt.close(prevalence_figure)

    return EvaluationSummary(
        metrics_path=metrics_path,
        thresholds_json=thresholds_json,
        thresholds_csv=thresholds_csv,
        validation_probabilities_csv=probabilities_csv,
        precision_recall_plot=precision_recall_plot,
        prevalence_ap_plot=prevalence_ap_plot,
        map_score=float(fixed["mAP"]),
        fixed_micro_f1=float(fixed["micro_f1"]),
        optimized_micro_f1=float(optimized["micro_f1"]),
        fixed_exact_match=float(fixed["exact_match"]),
        optimized_exact_match=float(optimized["exact_match"]),
    )


def load_thresholds_json(path: PathLike, labels: Sequence[str]) -> np.ndarray:
    """Load and validate a threshold JSON file against checkpoint label order."""

    threshold_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EngineError(f"could not load thresholds {threshold_path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise EngineError("threshold JSON must contain an object")

    expected_labels = tuple(str(label) for label in labels)
    payload_labels = payload.get("labels", payload.get("class_names"))
    if payload_labels is not None:
        if not isinstance(payload_labels, list) or tuple(map(str, payload_labels)) != expected_labels:
            raise EngineError("threshold label names/order do not match the checkpoint")

    values = payload.get("thresholds")
    if isinstance(values, Mapping):
        try:
            values = [values[label] for label in expected_labels]
        except KeyError as error:
            raise EngineError(f"threshold JSON is missing label {error.args[0]!r}") from error
    thresholds = np.asarray(values, dtype=np.float64)
    if thresholds.shape != (len(expected_labels),):
        raise EngineError(
            f"expected {len(expected_labels)} thresholds, found shape {thresholds.shape}"
        )
    if not np.all(np.isfinite(thresholds)) or np.any(
        (thresholds < 0.0) | (thresholds > 1.0)
    ):
        raise EngineError("thresholds must be finite values in [0, 1]")
    return thresholds


def save_prediction_tables(
    result: InferenceResult,
    labels: Sequence[str],
    thresholds: float | Sequence[float] | np.ndarray,
    output_directory: PathLike,
) -> PredictionSummary:
    """Save test probabilities and thresholded predictions in original label order."""

    normalized_labels = tuple(str(label) for label in labels)
    if result.probabilities.ndim != 2 or result.probabilities.shape[1] != len(
        normalized_labels
    ):
        raise ValueError("probability columns do not match labels")
    if result.probabilities.shape[0] != len(result.filenames):
        raise ValueError("probability rows do not match filenames")
    if not np.all(np.isfinite(result.probabilities)) or np.any(
        (result.probabilities < 0.0) | (result.probabilities > 1.0)
    ):
        raise ValueError("probabilities must be finite values in [0, 1]")

    threshold_array = np.asarray(thresholds, dtype=np.float64)
    if threshold_array.ndim == 0:
        threshold_array = np.full(len(normalized_labels), float(threshold_array))
    if threshold_array.shape != (len(normalized_labels),):
        raise ValueError("threshold count does not match labels")
    if not np.all(np.isfinite(threshold_array)) or np.any(
        (threshold_array < 0.0) | (threshold_array > 1.0)
    ):
        raise ValueError("thresholds must be finite values in [0, 1]")

    output_dir = Path(output_directory).expanduser().resolve()
    probabilities_path = output_dir / "test_probabilities.csv"
    predictions_path = output_dir / "test_predictions.csv"
    probability_frame = pd.DataFrame({FILENAME_COLUMN: result.filenames})
    prediction_frame = pd.DataFrame({FILENAME_COLUMN: result.filenames})
    binary = (result.probabilities >= threshold_array[np.newaxis, :]).astype(np.int8)
    for index, label in enumerate(normalized_labels):
        probability_frame[label] = result.probabilities[:, index]
        prediction_frame[label] = binary[:, index]
    _atomic_write_dataframe(probability_frame, probabilities_path)
    _atomic_write_dataframe(prediction_frame, predictions_path)
    return PredictionSummary(
        probabilities_csv=probabilities_path,
        predictions_csv=predictions_path,
        filenames=result.filenames,
        labels=normalized_labels,
        thresholds=tuple(float(value) for value in threshold_array),
    )


def predict_test(
    config: ConfigLike,
    checkpoint_path: PathLike | None = None,
    *,
    thresholds_path: PathLike | None = None,
    device: DeviceLike = "auto",
) -> PredictionSummary:
    """Run test inference and save probability and binary prediction CSV files."""

    resolved = _resolved_config(config)
    seed_everything(resolved.training.seed)
    loaded = load_experiment_checkpoint(
        resolved,
        checkpoint_path,
        device=device,
    )
    test_metadata = create_test_metadata(resolved.data.test_dir)
    test_dataset = AnimalAudioDataset(
        test_metadata,
        resolved.data.test_dir,
        labels=loaded.labels,
        has_targets=False,
        sample_rate=resolved.feature.sample_rate,
        duration_seconds=resolved.feature.duration_seconds,
    )
    test_loader = _make_loader(
        test_dataset,
        batch_size=resolved.training.batch_size,
        shuffle=False,
        num_workers=resolved.training.num_workers,
        pin_memory=resolved.training.pin_memory,
        seed=resolved.training.seed + 2,
    )
    result = run_inference(
        loaded.model,
        loaded.feature_extractor,
        test_loader,
        device=loaded.device,
        amp=resolved.training.amp,
    )
    thresholds = (
        load_thresholds_json(thresholds_path, loaded.labels)
        if thresholds_path is not None
        else np.full(len(loaded.labels), 0.5, dtype=np.float64)
    )
    return save_prediction_tables(
        result,
        loaded.labels,
        thresholds,
        resolved.output_dir,
    )


def benchmark_training_step(
    config: ConfigLike,
    *,
    device: DeviceLike = "auto",
) -> dict[str, Any]:
    """Measure one real forward/backward pass without updating model weights."""

    import time

    resolved = _resolved_config(config)
    seed_everything(resolved.training.seed)
    prepared = prepare_split(resolved)
    dataset = AnimalAudioDataset(
        prepared.train_metadata.iloc[: resolved.training.batch_size],
        resolved.data.train_dir,
        labels=prepared.labels,
        sample_rate=resolved.feature.sample_rate,
        duration_seconds=resolved.feature.duration_seconds,
    )
    loader = _make_loader(
        dataset,
        batch_size=resolved.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=resolved.training.pin_memory,
        seed=resolved.training.seed,
    )
    batch = next(iter(loader))
    model, feature_extractor, resolved_device = build_experiment_components(
        resolved, device
    )
    model.train()
    targets = cast(torch.Tensor, batch["target"]).to(resolved_device)
    waveforms = cast(torch.Tensor, batch["waveform"])
    use_amp = _amp_enabled(resolved_device, resolved.training.amp)
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
        torch.cuda.synchronize(resolved_device)
    started = time.perf_counter()
    features = extract_fixed_features(feature_extractor, waveforms, resolved_device)
    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    frontend_done = time.perf_counter()
    with torch.autocast(
        device_type=resolved_device.type,
        dtype=torch.float16,
        enabled=use_amp,
    ):
        logits = model(features)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    forward_done = time.perf_counter()
    loss.backward()
    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    backward_done = time.perf_counter()
    peak_memory = (
        torch.cuda.max_memory_allocated(resolved_device) / (1024**2)
        if resolved_device.type == "cuda"
        else 0.0
    )
    return {
        "updated_weights": False,
        "device": str(resolved_device),
        "batch_size": int(targets.shape[0]),
        "feature_shape": list(features.shape),
        "logit_shape": list(logits.shape),
        "loss": float(loss.detach()),
        "frontend_seconds": frontend_done - started,
        "forward_seconds": forward_done - frontend_done,
        "backward_seconds": backward_done - forward_done,
        "total_seconds": backward_done - started,
        "peak_memory_mib": peak_memory,
        "training_batches_per_epoch": math.ceil(
            len(prepared.train_metadata) / resolved.training.batch_size
        ),
    }


def inspect_model(
    config: ConfigLike,
    *,
    device: DeviceLike = "cpu",
    dry_forward: bool = False,
) -> dict[str, Any]:
    """Report model parameter counts and optionally run a tiny eval-mode forward."""

    resolved = _resolved_config(config)
    model, _feature_extractor, resolved_device = build_experiment_components(
        resolved, device
    )
    nddr_ids = {id(parameter) for parameter in model.nddr_parameters()}
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    nddr = sum(
        parameter.numel()
        for parameter in model.parameters()
        if id(parameter) in nddr_ids
    )
    report: dict[str, Any] = {
        "architecture": resolved.model.architecture,
        "device": str(resolved_device),
        "num_classes": model.num_classes,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "nddr_parameters": nddr,
        "base_parameters": total - nddr,
    }
    if dry_forward:
        time_frames = max(
            4,
            math.prod(pool[1] for pool in resolved.model.pool_sizes),
        )
        inputs = torch.zeros(
            1,
            resolved.model.input_channels,
            NDDRMTL.input_frequency_bins,
            time_frames,
            dtype=torch.float32,
            device=resolved_device,
        )
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
        report["dry_forward_input_shape"] = list(inputs.shape)
        report["dry_forward_output_shape"] = list(outputs.shape)
    return report


# Friendly API aliases for callers using alternate naming conventions.
prepare_data = prepare_split
build_data_loaders = build_dataloaders
compute_pos_weight = calculate_pos_weight
validate_or_create_split = prepare_split
