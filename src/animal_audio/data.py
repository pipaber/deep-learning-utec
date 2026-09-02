"""Data loading utilities for the animal-audio multilabel dataset."""

from __future__ import annotations

import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

PathLike: TypeAlias = str | Path

FILENAME_COLUMN = "filename"
SPLIT_COLUMN = "split"
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "val"
DEFAULT_SAMPLE_RATE = 22_050
DEFAULT_DURATION_SECONDS = 3.0
DEFAULT_LABEL_COUNT = 42


class MetadataValidationError(ValueError):
    """Raised when a metadata table does not match the expected schema."""


class AudioDecodingError(ValueError):
    """Raised when a WAV file is not supported or is malformed."""


@dataclass(frozen=True, slots=True)
class LabelSummary:
    """Positive-example counts and classes without enough positive support."""

    label_counts: dict[str, int]
    unsupported_classes: tuple[str, ...]

    @property
    def supported_classes(self) -> tuple[str, ...]:
        return tuple(
            label
            for label, count in self.label_counts.items()
            if label not in self.unsupported_classes and count > 0
        )


@dataclass(frozen=True, slots=True)
class SplitLabelSupport:
    """Positive support available globally and in each split."""

    labels: tuple[str, ...]
    global_counts: tuple[int, ...]
    positive_group_counts: tuple[int, ...]
    train_counts: tuple[int, ...]
    validation_counts: tuple[int, ...]
    globally_unsupported: tuple[str, ...]
    single_group_classes: tuple[str, ...]
    train_unsupported: tuple[str, ...]
    validation_unsupported: tuple[str, ...]


def label_columns(
    metadata: pd.DataFrame,
    *,
    filename_column: str = FILENAME_COLUMN,
    split_column: str = SPLIT_COLUMN,
) -> list[str]:
    """Return label columns in file order, excluding metadata-only columns."""

    return [
        str(column)
        for column in metadata.columns
        if column not in {filename_column, split_column}
    ]


def validate_metadata(
    metadata: pd.DataFrame,
    *,
    filename_column: str = FILENAME_COLUMN,
    expected_label_count: int | None = DEFAULT_LABEL_COUNT,
    require_labels: bool = True,
) -> pd.DataFrame:
    """Validate and normalize a train/test metadata table.

    Label values may be numeric strings in the input CSV, but the returned
    columns are always ``int8`` and contain only zero or one. Set
    ``expected_label_count=None`` for small fixtures or datasets whose class
    count is intentionally not fixed. With ``require_labels=False``, a
    filename-only test table is valid; any label columns that are present are
    still validated.
    """

    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame")
    if metadata.columns.has_duplicates:
        duplicated = metadata.columns[metadata.columns.duplicated()].tolist()
        raise MetadataValidationError(f"duplicate metadata columns: {duplicated}")
    if filename_column not in metadata.columns:
        raise MetadataValidationError(
            f"metadata is missing required column {filename_column!r}"
        )
    if metadata.empty:
        raise MetadataValidationError("metadata must contain at least one row")

    result = metadata.copy()
    filenames = result[filename_column]
    if filenames.isna().any():
        raise MetadataValidationError("filename values must not be missing")
    if not filenames.map(lambda value: isinstance(value, str)).all():
        raise MetadataValidationError("filename values must be strings")

    stripped_filenames = filenames.str.strip()
    if stripped_filenames.eq("").any():
        raise MetadataValidationError("filename values must not be empty")
    if stripped_filenames.duplicated().any():
        duplicates = stripped_filenames[stripped_filenames.duplicated()].unique()
        raise MetadataValidationError(
            f"duplicate filenames are not allowed: {duplicates.tolist()}"
        )
    result[filename_column] = stripped_filenames

    labels = label_columns(result, filename_column=filename_column)
    if require_labels and not labels:
        raise MetadataValidationError("metadata does not contain any label columns")
    if expected_label_count is not None and len(labels) != expected_label_count:
        raise MetadataValidationError(
            f"expected {expected_label_count} label columns, found {len(labels)}"
        )

    for column in labels:
        values = pd.to_numeric(result[column], errors="coerce")
        invalid = values.isna() | ~values.isin((0, 1))
        if invalid.any():
            bad_values = result.loc[invalid, column].head(5).tolist()
            raise MetadataValidationError(
                f"label column {column!r} must contain only 0 or 1; "
                f"invalid values include {bad_values}"
            )
        result[column] = values.astype(np.int8)

    return result


def load_metadata(
    csv_path: PathLike,
    *,
    filename_column: str = FILENAME_COLUMN,
    expected_label_count: int | None = DEFAULT_LABEL_COUNT,
    require_labels: bool = True,
) -> pd.DataFrame:
    """Load a metadata CSV and validate its filenames and binary labels."""

    path = Path(csv_path)
    try:
        metadata = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise MetadataValidationError(f"could not read metadata CSV {path}: {exc}") from exc
    return validate_metadata(
        metadata,
        filename_column=filename_column,
        expected_label_count=expected_label_count,
        require_labels=require_labels,
    )


def summarize_labels(
    metadata: pd.DataFrame,
    *,
    labels: Sequence[str] | None = None,
    minimum_positive_count: int = 1,
    filename_column: str = FILENAME_COLUMN,
) -> LabelSummary:
    """Count positive labels and report classes below a support threshold.

    A class is considered unsupported when it has fewer than
    ``minimum_positive_count`` positive recordings. This makes zero-positive
    columns explicit before fitting a multilabel model.
    """

    if minimum_positive_count < 1:
        raise ValueError("minimum_positive_count must be at least 1")
    selected = list(labels) if labels is not None else label_columns(
        metadata, filename_column=filename_column
    )
    missing = [column for column in selected if column not in metadata.columns]
    if missing:
        raise MetadataValidationError(f"unknown label columns: {missing}")

    counts: dict[str, int] = {}
    for column in selected:
        values = pd.to_numeric(metadata[column], errors="coerce")
        invalid = values.isna() | ~values.isin((0, 1))
        if invalid.any():
            raise MetadataValidationError(
                f"label column {column!r} must contain only 0 or 1"
            )
        counts[column] = int(values.sum())

    unsupported = tuple(
        column for column, count in counts.items() if count < minimum_positive_count
    )
    return LabelSummary(label_counts=counts, unsupported_classes=unsupported)


def derive_recording_group(filename: str) -> str:
    """Remove the final ``_start_end`` suffix from a WAV filename stem."""

    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename must be a non-empty string")
    stem = Path(filename.strip()).stem
    parts = stem.rsplit("_", maxsplit=2)
    return parts[0] if len(parts) == 3 and parts[0] else stem


def validate_group_disjointness(
    split_metadata: pd.DataFrame,
    *,
    filename_column: str = FILENAME_COLUMN,
    split_column: str = SPLIT_COLUMN,
) -> None:
    """Raise if a recording group occurs in both train and validation."""

    if filename_column not in split_metadata.columns:
        raise MetadataValidationError(
            f"split metadata is missing {filename_column!r}"
        )
    if split_column not in split_metadata.columns:
        raise MetadataValidationError(f"split metadata is missing {split_column!r}")

    split_values = split_metadata[split_column]
    if split_values.isna().any():
        raise MetadataValidationError("split values must not be missing")
    unknown = sorted(set(split_values.astype(str)) - {TRAIN_SPLIT, VALIDATION_SPLIT})
    if unknown:
        raise MetadataValidationError(f"unsupported split values: {unknown}")
    if not split_values.eq(TRAIN_SPLIT).any() or not split_values.eq(
        VALIDATION_SPLIT
    ).any():
        raise MetadataValidationError("both train and val splits must be non-empty")

    groups = split_metadata[filename_column].map(derive_recording_group)
    group_split_counts = (
        pd.DataFrame({"group": groups, "split": split_values})
        .drop_duplicates()
        .groupby("group", sort=False)["split"]
        .nunique()
    )
    overlapping = group_split_counts[group_split_counts > 1].index.tolist()
    if overlapping:
        preview = overlapping[:5]
        raise MetadataValidationError(
            f"recording groups occur in multiple splits: {preview}"
        )


def summarize_split_label_support(
    split_metadata: pd.DataFrame,
    *,
    labels: Sequence[str] | None = None,
    filename_column: str = FILENAME_COLUMN,
    split_column: str = SPLIT_COLUMN,
) -> SplitLabelSupport:
    """Summarize per-class support while respecting source-recording groups."""

    validate_group_disjointness(
        split_metadata,
        filename_column=filename_column,
        split_column=split_column,
    )
    selected = tuple(labels) if labels is not None else tuple(
        label_columns(
            split_metadata,
            filename_column=filename_column,
            split_column=split_column,
        )
    )
    if not selected:
        raise MetadataValidationError("at least one label column is required")
    targets = split_metadata[list(selected)].to_numpy(dtype=np.int64, copy=False)
    groups = split_metadata[filename_column].map(derive_recording_group).to_numpy()
    train_mask = split_metadata[split_column].eq(TRAIN_SPLIT).to_numpy()
    validation_mask = split_metadata[split_column].eq(VALIDATION_SPLIT).to_numpy()
    global_counts = targets.sum(axis=0)
    train_counts = targets[train_mask].sum(axis=0)
    validation_counts = targets[validation_mask].sum(axis=0)
    positive_group_counts = np.asarray(
        [np.unique(groups[targets[:, index] == 1]).size for index in range(len(selected))],
        dtype=np.int64,
    )

    def names(mask: np.ndarray) -> tuple[str, ...]:
        return tuple(label for label, included in zip(selected, mask, strict=True) if included)

    globally_supported = global_counts > 0
    return SplitLabelSupport(
        labels=selected,
        global_counts=tuple(int(value) for value in global_counts),
        positive_group_counts=tuple(int(value) for value in positive_group_counts),
        train_counts=tuple(int(value) for value in train_counts),
        validation_counts=tuple(int(value) for value in validation_counts),
        globally_unsupported=names(~globally_supported),
        single_group_classes=names(globally_supported & (positive_group_counts == 1)),
        train_unsupported=names(globally_supported & (train_counts == 0)),
        validation_unsupported=names(globally_supported & (validation_counts == 0)),
    )


def create_group_aware_split(
    metadata: pd.DataFrame,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
    n_candidates: int = 4096,
    labels: Sequence[str] | None = None,
    filename_column: str = FILENAME_COLUMN,
    split_column: str = SPLIT_COLUMN,
) -> pd.DataFrame:
    """Create a deterministic, approximately stratified group-aware split.

    ``GroupShuffleSplit`` generates candidate partitions. The selected
    candidate minimizes the mean train/validation per-class prevalence error
    relative to the full dataset, plus absolute validation-size error. Groups
    are derived from filenames, so overlapping windows stay together.
    """

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if n_candidates < 1:
        raise ValueError("n_candidates must be at least 1")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    validated = validate_metadata(
        metadata,
        filename_column=filename_column,
        expected_label_count=None,
        require_labels=True,
    )
    selected_labels = list(labels) if labels is not None else label_columns(
        validated, filename_column=filename_column, split_column=split_column
    )
    if not selected_labels:
        raise MetadataValidationError("at least one label column is required")
    missing = [column for column in selected_labels if column not in validated.columns]
    if missing:
        raise MetadataValidationError(f"unknown label columns: {missing}")

    groups = validated[filename_column].map(derive_recording_group).to_numpy()
    if np.unique(groups).size < 2:
        raise MetadataValidationError(
            "at least two recording groups are required to create a split"
        )

    targets = validated[selected_labels].to_numpy(dtype=np.float64, copy=True)
    overall_prevalence = targets.mean(axis=0)
    splitter = GroupShuffleSplit(
        n_splits=n_candidates,
        test_size=val_fraction,
        random_state=seed,
    )

    global_counts = targets.sum(axis=0)
    globally_supported = global_counts > 0
    positive_group_counts = np.asarray(
        [np.unique(groups[targets[:, index] == 1]).size for index in range(len(selected_labels))],
        dtype=np.int64,
    )
    validation_support_possible = positive_group_counts >= 2

    best_score = float("inf")
    best_indices: tuple[np.ndarray, np.ndarray] | None = None
    placeholders = np.zeros(len(validated), dtype=np.uint8)
    for train_indices, val_indices in splitter.split(
        placeholders, targets, groups=groups
    ):
        train_targets = targets[train_indices]
        val_targets = targets[val_indices]
        train_counts = train_targets.sum(axis=0)
        val_counts = val_targets.sum(axis=0)
        if np.any(globally_supported & (train_counts == 0)):
            continue
        if np.any(validation_support_possible & (val_counts == 0)):
            continue
        train_prevalence = train_targets.mean(axis=0)
        val_prevalence = val_targets.mean(axis=0)
        prevalence_error = float(
            np.mean(
                np.stack(
                    (
                        np.abs(train_prevalence - overall_prevalence),
                        np.abs(val_prevalence - overall_prevalence),
                    )
                )
            )
        )
        size_error = abs((len(val_indices) / len(validated)) - val_fraction)
        score = prevalence_error + size_error
        if score < best_score:
            best_score = score
            best_indices = (train_indices.copy(), val_indices.copy())

    if best_indices is None:
        raise MetadataValidationError(
            "could not create a group split with training positives for every "
            "globally supported class and validation positives for classes present "
            "in at least two groups; increase n_candidates or review rare-class groups"
        )

    split_values = np.full(len(validated), TRAIN_SPLIT, dtype=object)
    split_values[best_indices[1]] = VALIDATION_SPLIT
    result = validated.copy()
    result[split_column] = split_values
    validate_group_disjointness(
        result, filename_column=filename_column, split_column=split_column
    )
    return result


def save_split(
    split_metadata: pd.DataFrame,
    csv_path: PathLike,
    *,
    filename_column: str = FILENAME_COLUMN,
    split_column: str = SPLIT_COLUMN,
) -> Path:
    """Validate and save metadata containing a ``split`` column."""

    validate_group_disjointness(
        split_metadata, filename_column=filename_column, split_column=split_column
    )
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    split_metadata.to_csv(path, index=False)
    return path


def load_split(
    csv_path: PathLike,
    *,
    filename_column: str = FILENAME_COLUMN,
    split_column: str = SPLIT_COLUMN,
    expected_label_count: int | None = None,
) -> pd.DataFrame:
    """Load a saved split and verify its labels, values, and group isolation."""

    loaded = load_metadata(
        csv_path,
        filename_column=filename_column,
        expected_label_count=expected_label_count,
        require_labels=expected_label_count not in (None, 0),
    )
    validate_group_disjointness(
        loaded, filename_column=filename_column, split_column=split_column
    )
    return loaded


def _decode_pcm(raw_data: bytes, sample_width: int) -> np.ndarray:
    """Decode little-endian integer PCM bytes to normalized float32."""

    if sample_width == 1:
        samples = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32)
        return (samples - 128.0) / 128.0
    if sample_width == 2:
        samples = np.frombuffer(raw_data, dtype="<i2").astype(np.float32)
        return samples / 32_768.0
    if sample_width == 3:
        bytes_24 = np.frombuffer(raw_data, dtype=np.uint8)
        if bytes_24.size % 3:
            raise AudioDecodingError("24-bit PCM data has an incomplete sample")
        triples = bytes_24.reshape(-1, 3).astype(np.int32)
        samples = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        samples = np.where(samples & 0x800000, samples - 0x1000000, samples)
        return samples.astype(np.float32) / 8_388_608.0
    if sample_width == 4:
        samples = np.frombuffer(raw_data, dtype="<i4").astype(np.float32)
        return samples / 2_147_483_648.0
    raise AudioDecodingError(
        f"unsupported PCM sample width: {sample_width * 8} bits"
    )


def load_pcm_wav(
    wav_path: PathLike,
    *,
    expected_sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
) -> np.ndarray:
    """Load integer PCM WAV audio as fixed-length mono float32 samples.

    8-bit unsigned and 16/24/32-bit signed integer PCM are supported. Multiple
    channels are averaged, short clips are right-padded with zeros, and long
    clips are truncated.
    """

    if expected_sample_rate <= 0:
        raise ValueError("expected_sample_rate must be positive")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    path = Path(wav_path)
    try:
        with wave.open(str(path), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise AudioDecodingError(
                    f"compressed WAV is not supported: {wav_file.getcomptype()}"
                )
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            raw_data = wav_file.readframes(frame_count)
    except (OSError, EOFError, wave.Error) as exc:
        raise AudioDecodingError(f"could not decode WAV file {path}: {exc}") from exc

    if channels < 1:
        raise AudioDecodingError("WAV file must contain at least one channel")
    if sample_rate != expected_sample_rate:
        raise AudioDecodingError(
            f"expected sample rate {expected_sample_rate}, found {sample_rate}"
        )
    frame_width = channels * sample_width
    if frame_width <= 0 or len(raw_data) % frame_width:
        raise AudioDecodingError("PCM data does not contain complete audio frames")

    decoded = _decode_pcm(raw_data, sample_width)
    if channels > 1:
        decoded = decoded.reshape(-1, channels).mean(axis=1, dtype=np.float32)

    required_samples = int(round(expected_sample_rate * duration_seconds))
    if decoded.size >= required_samples:
        waveform = decoded[:required_samples]
    else:
        waveform = np.pad(decoded, (0, required_samples - decoded.size))
    return np.ascontiguousarray(waveform, dtype=np.float32)


def load_wav(
    wav_path: PathLike,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration: float = DEFAULT_DURATION_SECONDS,
) -> np.ndarray:
    """Convenience wrapper around :func:`load_pcm_wav`."""

    return load_pcm_wav(
        wav_path,
        expected_sample_rate=sample_rate,
        duration_seconds=duration,
    )


class AnimalAudioDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Dataset yielding collate-friendly waveform/target/filename mappings."""

    def __init__(
        self,
        metadata: pd.DataFrame | PathLike,
        audio_directory: PathLike,
        *,
        labels: Sequence[str] | None = None,
        has_targets: bool | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        duration_seconds: float = DEFAULT_DURATION_SECONDS,
        filename_column: str = FILENAME_COLUMN,
    ) -> None:
        raw_metadata = (
            pd.read_csv(metadata) if isinstance(metadata, (str, Path)) else metadata
        )
        inferred_labels = label_columns(raw_metadata, filename_column=filename_column)
        if labels is not None:
            selected_labels = list(labels)
        else:
            selected_labels = inferred_labels
        if has_targets is None:
            has_targets = bool(selected_labels)
        if not has_targets:
            selected_labels = []

        self.metadata = validate_metadata(
            raw_metadata,
            filename_column=filename_column,
            expected_label_count=(len(inferred_labels) if inferred_labels else None),
            require_labels=has_targets,
        ).reset_index(drop=True)
        missing = [
            column for column in selected_labels if column not in self.metadata.columns
        ]
        if missing:
            raise MetadataValidationError(f"unknown label columns: {missing}")

        self.audio_directory = Path(audio_directory)
        self.label_columns = tuple(selected_labels)
        self.has_targets = has_targets
        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds
        self.filename_column = filename_column
        self._targets = (
            self.metadata[list(self.label_columns)].to_numpy(dtype=np.float32, copy=True)
            if self.has_targets
            else None
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        filename = str(self.metadata.iloc[index][self.filename_column])
        waveform = load_pcm_wav(
            self.audio_directory / filename,
            expected_sample_rate=self.sample_rate,
            duration_seconds=self.duration_seconds,
        )
        sample: dict[str, torch.Tensor | str] = {
            "waveform": torch.from_numpy(waveform),
            "filename": filename,
        }
        if self._targets is not None:
            sample["target"] = torch.from_numpy(self._targets[index])
        return sample


# A concise alias for callers that prefer a verb-first split API.
split_metadata = create_group_aware_split
