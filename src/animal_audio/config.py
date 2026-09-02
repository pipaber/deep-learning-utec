"""Typed configuration for the animal-audio NDDR-MTL experiment.

Relative paths are resolved against an explicit ``project_root`` when supplied.
Otherwise, loading or saving searches upward from the config file for the nearest
``pyproject.toml``; if none is found, the config file's directory is used. This
makes a config in ``<project>/configs`` resolve ``train`` to ``<project>/train``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from types import UnionType
from typing import Any, Mapping, TypeVar, Union, cast, get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    """Raised when a configuration is malformed or contains unknown keys."""


@dataclass
class DataConfig:
    """Input locations.

    ``train_dir`` and ``test_dir`` are the root extraction destinations for
    their corresponding archives; no nested ``train/train`` or ``test/test``
    layout is assumed.
    """

    train_dir: Path = Path("train")
    test_dir: Path = Path("test")
    metadata_csv: Path = Path("train.csv")
    split_csv: Path = Path("artifacts/split.csv")
    train_archive: Path | None = None
    test_archive: Path | None = None

    def __post_init__(self) -> None:
        for name in (
            "train_dir",
            "test_dir",
            "metadata_csv",
            "split_csv",
            "train_archive",
            "test_archive",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                setattr(self, name, Path(value))

    @property
    def metadata_path(self) -> Path:
        """Alias useful to consumers that do not assume a CSV implementation."""

        return self.metadata_csv

    @property
    def split_path(self) -> Path:
        """Alias useful to consumers that do not assume a CSV implementation."""

        return self.split_csv


@dataclass
class FeatureConfig:
    """MS-PCEN extraction settings adapted to 22.05 kHz audio."""

    sample_rate: int = 22_050
    duration_seconds: float = 3.0
    n_fft: int = 512
    win_length: int = 512
    hop_length: int = 128
    n_mels: int = 64
    f_min: float = 0.0
    f_max: float | None = None
    kind: str = "pcen"
    repeat_channels: int = 3
    pcen_gain: float = 0.98
    pcen_bias: float = 0.05
    pcen_power: float = 0.5
    pcen_eps: float = 1e-4
    pcen_smoothing: float = 0.967

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("feature.sample_rate must be positive")
        if self.duration_seconds <= 0:
            raise ValueError("feature.duration_seconds must be positive")
        if min(self.n_fft, self.win_length, self.hop_length, self.n_mels) <= 0:
            raise ValueError("FFT, window, hop, and mel sizes must be positive")
        if self.win_length > self.n_fft:
            raise ValueError("feature.win_length cannot exceed feature.n_fft")
        if self.f_max is not None and not self.f_min < self.f_max <= self.sample_rate / 2:
            raise ValueError("feature frequencies must satisfy f_min < f_max <= sample_rate / 2")
        if self.kind not in {"pcen", "logmel"}:
            raise ValueError("feature.kind must be 'pcen' or 'logmel'")
        if self.repeat_channels not in {1, 3}:
            raise ValueError("feature.repeat_channels must be 1 or 3")
        if not 0.0 <= self.pcen_gain <= 1.0:
            raise ValueError("feature.pcen_gain must be in [0, 1]")
        if self.pcen_bias < 0.0 or self.pcen_power <= 0.0 or self.pcen_eps <= 0.0:
            raise ValueError("PCEN bias must be non-negative; power and eps must be positive")
        if not 0.0 < self.pcen_smoothing < 1.0:
            raise ValueError("feature.pcen_smoothing must be in (0, 1)")

    @property
    def sr(self) -> int:
        """Short alias commonly used by audio libraries."""

        return self.sample_rate

    @property
    def duration(self) -> float:
        """Duration in seconds."""

        return self.duration_seconds

    def to_extractor_kwargs(self) -> dict[str, object]:
        """Return arguments accepted by ``AudioFeatureExtractor``."""

        return {
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "win_length": self.win_length,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "f_min": self.f_min,
            "f_max": self.f_max,
            "mode": self.kind,
            "offset": self.pcen_bias,
            "gain": self.pcen_gain,
            "power": self.pcen_power,
            "eps": self.pcen_eps,
            "smoothing": self.pcen_smoothing,
            "repeat_channels": self.repeat_channels,
            "expected_num_samples": round(self.sample_rate * self.duration_seconds),
        }


@dataclass
class ModelConfig:
    """Compact NDDR-MTL CRNN settings for 42 binary label paths."""

    architecture: str = "nddr_mtl"
    num_classes: int = 42
    input_channels: int = 3
    conv_channels: int = 4
    conv_layers: int = 3
    kernel_size: tuple[int, int] = (5, 5)
    pool_sizes: tuple[tuple[int, int], ...] = ((5, 1), (4, 1), (2, 1))
    conv_dropout: float = 0.25
    gru_hidden_size: int = 16
    gru_layers: int = 3
    gru_dropout: float = 0.2
    bidirectional: bool = True
    nddr_self_weight: float = 0.6
    nddr_cross_weight: float | None = None
    nddr_skip_weights: tuple[float, ...] = (0.2, 0.2, 0.6)

    def __post_init__(self) -> None:
        self.kernel_size = tuple(self.kernel_size)
        self.pool_sizes = tuple(tuple(pool) for pool in self.pool_sizes)
        self.nddr_skip_weights = tuple(self.nddr_skip_weights)
        if self.architecture != "nddr_mtl":
            raise ValueError("only architecture='nddr_mtl' is supported")
        if min(
            self.num_classes,
            self.input_channels,
            self.conv_channels,
            self.conv_layers,
            self.gru_hidden_size,
            self.gru_layers,
        ) <= 0:
            raise ValueError("model dimensions and layer counts must be positive")
        if len(self.kernel_size) != 2 or any(size <= 0 for size in self.kernel_size):
            raise ValueError("model.kernel_size must contain two positive integers")
        if len(self.pool_sizes) != self.conv_layers:
            raise ValueError("model.pool_sizes must have one entry per convolutional layer")
        if any(len(pool) != 2 or any(size <= 0 for size in pool) for pool in self.pool_sizes):
            raise ValueError("each model.pool_sizes entry must contain two positive integers")
        if not 0.0 <= self.conv_dropout < 1.0 or not 0.0 <= self.gru_dropout < 1.0:
            raise ValueError("model dropouts must be in [0, 1)")
        if not 0.0 <= self.nddr_self_weight <= 1.0:
            raise ValueError("model.nddr_self_weight must be in [0, 1]")
        if self.nddr_cross_weight is not None and not 0.0 <= self.nddr_cross_weight <= 1.0:
            raise ValueError("model.nddr_cross_weight must be null or in [0, 1]")
        if len(self.nddr_skip_weights) != self.conv_layers:
            raise ValueError("model.nddr_skip_weights must have one entry per convolutional layer")
        if any(weight < 0.0 for weight in self.nddr_skip_weights):
            raise ValueError("model.nddr_skip_weights cannot be negative")
        if abs(sum(self.nddr_skip_weights) - 1.0) > 1e-6:
            raise ValueError("model.nddr_skip_weights must sum to 1")

    @property
    def gru_hidden(self) -> int:
        """Short alias for the per-direction GRU hidden size."""

        return self.gru_hidden_size

    @property
    def resolved_cross_weight(self) -> float:
        """Cross-task initializer preserving a total fusion weight of one.

        The paper's 0.1 value is exact for five tasks. With 42 tasks, distributing
        the remaining 0.4 across all other branches avoids a 4.7x initial sum.
        Set ``nddr_cross_weight`` explicitly to reproduce another initializer.
        """

        if self.nddr_cross_weight is not None:
            return self.nddr_cross_weight
        if self.num_classes == 1:
            return 0.0
        return (1.0 - self.nddr_self_weight) / (self.num_classes - 1)

    def to_model_kwargs(self) -> dict[str, object]:
        """Return arguments accepted by ``NDDRMTL``/``NDDRMTLConfig``."""

        return {
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "conv_channels": self.conv_channels,
            "gru_hidden_size": self.gru_hidden_size,
            "num_gru_layers": self.gru_layers,
            "pool_sizes": self.pool_sizes,
            "cnn_dropout": self.conv_dropout,
            "gru_dropout": self.gru_dropout,
            "nddr_self_weight": self.nddr_self_weight,
            "nddr_cross_weight": self.resolved_cross_weight,
            "nddr_skip_weights": self.nddr_skip_weights,
        }


@dataclass
class TrainingConfig:
    """Training hyperparameters; importing this module never starts training."""

    seed: int = 42
    batch_size: int = 8
    epochs: int = 100
    learning_rate: float = 1e-3
    lr_decay: float = 0.75
    lr_decay_steps: int | None = None
    lr_decay_every_epochs: float = 0.5
    use_pos_weight: bool = True
    pos_weight_min: float = 1.0
    pos_weight_max: float = 20.0
    amp: bool = True
    grad_clip_norm: float | None = 5.0
    base_weight_decay: float = 1e-4
    nddr_weight_decay: float = 1e-2
    num_workers: int = 0
    pin_memory: bool = True
    save_checkpoints: bool = True
    checkpoint_filename: str = "best.pt"
    checkpoint_every: int = 1
    resume_from: Path | None = None
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.resume_from is not None and not isinstance(self.resume_from, Path):
            self.resume_from = Path(self.resume_from)
        if self.seed < 0 or self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("seed must be non-negative; batch size and epochs must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("training.learning_rate must be positive")
        if self.lr_decay_steps is not None and self.lr_decay_steps <= 0:
            raise ValueError("training.lr_decay_steps must be positive or null")
        if not 0.0 < self.lr_decay <= 1.0 or self.lr_decay_every_epochs <= 0.0:
            raise ValueError("learning-rate decay must be in (0, 1] with a positive interval")
        if self.pos_weight_min <= 0.0 or self.pos_weight_max < self.pos_weight_min:
            raise ValueError("invalid positive-weight clamp")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0.0:
            raise ValueError("training.grad_clip_norm must be positive or null")
        if self.base_weight_decay < 0.0 or self.nddr_weight_decay < 0.0:
            raise ValueError("weight decay cannot be negative")
        if self.num_workers < 0:
            raise ValueError("training.num_workers cannot be negative")
        if not self.checkpoint_filename:
            raise ValueError("training.checkpoint_filename cannot be empty")
        if self.checkpoint_every <= 0:
            raise ValueError("training.checkpoint_every must be positive")
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            raise ValueError("early-stopping patience must be positive or null")
        if self.early_stopping_min_delta < 0.0:
            raise ValueError("early-stopping minimum delta cannot be negative")

    @property
    def pos_weight_clip(self) -> tuple[float, float]:
        """Clamp applied to BCE ``pos_weight`` values derived from training data."""

        return (self.pos_weight_min, self.pos_weight_max)

    @property
    def weight_decay(self) -> float:
        """Alias for the non-NDDR/base parameter-group weight decay."""

        return self.base_weight_decay


@dataclass
class ExperimentConfig:
    """Complete, serializable NDDR-PCEN experiment configuration."""

    name: str = "nddr_pcen"
    output_dir: Path = Path("artifacts/experiments/nddr_pcen")
    data: DataConfig = field(default_factory=DataConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)
        if not self.name:
            raise ValueError("experiment.name cannot be empty")
        if self.feature.repeat_channels != self.model.input_channels:
            raise ValueError(
                "feature.repeat_channels must equal model.input_channels "
                f"({self.feature.repeat_channels} != {self.model.input_channels})"
            )

    def resolved(self, project_root: str | Path) -> ExperimentConfig:
        """Return a copy with all filesystem paths made absolute."""

        root = Path(project_root).expanduser().resolve()

        def resolve_path(value: Path | None) -> Path | None:
            if value is None:
                return None
            expanded = value.expanduser()
            return expanded.resolve() if expanded.is_absolute() else (root / expanded).resolve()

        data = replace(
            self.data,
            train_dir=resolve_path(self.data.train_dir),
            test_dir=resolve_path(self.data.test_dir),
            metadata_csv=resolve_path(self.data.metadata_csv),
            split_csv=resolve_path(self.data.split_csv),
            train_archive=resolve_path(self.data.train_archive),
            test_archive=resolve_path(self.data.test_archive),
        )
        training = replace(self.training, resume_from=resolve_path(self.training.resume_from))
        return replace(
            self,
            output_dir=resolve_path(self.output_dir),
            data=data,
            training=training,
        )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        project_root: str | Path | None = None,
    ) -> ExperimentConfig:
        """Load a YAML config, reject unknown keys, and resolve its paths."""

        return load_config(path, project_root=project_root)

    def save(
        self,
        path: str | Path,
        *,
        project_root: str | Path | None = None,
    ) -> Path:
        """Save a resolved snapshot as YAML (or JSON when suffixed ``.json``)."""

        return save_config(self, path, project_root=project_root)


T = TypeVar("T")


def _coerce_value(value: Any, annotation: Any, location: str) -> Any:
    origin = get_origin(annotation)
    arguments = get_args(annotation)

    if annotation is Any:
        return value
    if annotation is Path:
        if not isinstance(value, (str, Path)):
            raise ConfigError(f"{location} must be a path string")
        return Path(value)
    if isinstance(annotation, type) and is_dataclass(annotation):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{location} must be a mapping")
        return _construct_dataclass(annotation, value, location)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{location} must be a YAML list")
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(
                _coerce_value(item, arguments[0], f"{location}[{index}]")
                for index, item in enumerate(value)
            )
        if len(value) != len(arguments):
            raise ConfigError(f"{location} must contain {len(arguments)} items")
        return tuple(
            _coerce_value(item, item_type, f"{location}[{index}]")
            for index, (item, item_type) in enumerate(zip(value, arguments, strict=True))
        )
    if origin in (Union, UnionType):
        if value is None and type(None) in arguments:
            return None
        errors: list[str] = []
        for option in arguments:
            if option is type(None):
                continue
            try:
                return _coerce_value(value, option, location)
            except ConfigError as error:
                errors.append(str(error))
        raise ConfigError(errors[0] if errors else f"invalid value for {location}")
    if annotation is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{location} must be a boolean")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{location} must be an integer")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{location} must be a number")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{location} must be a string")
        return value
    return value


def _construct_dataclass(cls: type[T], values: Mapping[str, Any], location: str) -> T:
    field_names = {field.name for field in fields(cast(Any, cls))}
    unknown = sorted(set(values) - field_names)
    if unknown:
        qualified = ", ".join(f"{location}.{key}" for key in unknown)
        raise ConfigError(f"unknown configuration key(s): {qualified}")

    hints = get_type_hints(cls)
    converted = {
        name: _coerce_value(value, hints[name], f"{location}.{name}")
        for name, value in values.items()
    }
    try:
        return cls(**converted)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"invalid {location}: {error}") from error


def _find_project_root(start: Path) -> Path:
    current = start.expanduser().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return current


def load_config(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> ExperimentConfig:
    """Load strict YAML and resolve relative paths against the project root.

    The root is explicit when ``project_root`` is provided. Otherwise it is the
    nearest ancestor of the YAML file containing ``pyproject.toml``, falling
    back to the YAML file's parent directory.
    """

    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"could not load config {config_path}: {error}") from error

    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ConfigError("the YAML document root must be a mapping")

    config = _construct_dataclass(ExperimentConfig, raw, "experiment")
    root = Path(project_root).expanduser().resolve() if project_root else _find_project_root(config_path.parent)
    return config.resolved(root)


def _serialize(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _serialize(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    return value


def save_config(
    config: ExperimentConfig,
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> Path:
    """Save a fully resolved config snapshot to ``.yaml``, ``.yml``, or ``.json``."""

    output_path = Path(path).expanduser().resolve()
    root = Path(project_root).expanduser().resolve() if project_root else _find_project_root(output_path.parent)
    payload = _serialize(config.resolved(root))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if output_path.suffix.lower() == ".json":
            with output_path.open("w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, ensure_ascii=False)
                file.write("\n")
        elif output_path.suffix.lower() in {".yaml", ".yml"}:
            with output_path.open("w", encoding="utf-8") as file:
                yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=True)
        else:
            raise ConfigError("config output must use a .yaml, .yml, or .json suffix")
    except OSError as error:
        raise ConfigError(f"could not save config {output_path}: {error}") from error

    return output_path


load_yaml = load_config
