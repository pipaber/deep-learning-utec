"""Neural discriminative dimensionality reduction for multi-task learning."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, TypedDict, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


PoolSize = tuple[int, int]
_DEFAULT_POOL_SIZES: tuple[PoolSize, PoolSize, PoolSize] = ((5, 1), (4, 1), (2, 1))


class OptimizerParameterGroup(TypedDict):
    """A parameter group accepted by PyTorch optimizers."""

    name: str
    params: list[nn.Parameter]
    weight_decay: float


@dataclass(frozen=True, slots=True)
class NDDRMTLConfig:
    """Configuration for :class:`NDDRMTL`."""

    num_classes: int
    input_channels: int = 1
    conv_channels: int = 64
    gru_hidden_size: int = 128
    num_gru_layers: int = 3
    pool_sizes: Sequence[PoolSize] = _DEFAULT_POOL_SIZES
    cnn_dropout: float = 0.25
    gru_dropout: float = 0.2
    nddr_self_weight: float = 0.6
    nddr_cross_weight: float = 0.1
    nddr_skip_weights: Sequence[float] = (0.2, 0.2, 0.6)


class _CNNBlock(nn.Module):
    """One task-specific convolutional feature-extraction block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: PoolSize,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
        self.dropout = nn.Dropout2d(dropout)

    def finish(self, convolved: Tensor) -> Tensor:
        """Apply the non-convolutional part after a grouped task convolution."""

        features = self.batch_norm(convolved)
        features = self.activation(features)
        features = self.pool(features)
        return self.dropout(features)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.finish(self.conv(inputs))


class _NDDRLayer(nn.Module):
    """Fuse corresponding feature maps across all task branches."""

    def __init__(
        self,
        num_tasks: int,
        channels: int,
        *,
        self_weight: float,
        cross_weight: float,
    ) -> None:
        super().__init__()
        fused_channels = num_tasks * channels
        self.batch_norm = nn.BatchNorm2d(fused_channels)
        self.projections = nn.ModuleList(
            nn.Conv2d(fused_channels, channels, kernel_size=1)
            for _ in range(num_tasks)
        )
        self._initialize_projections(
            num_tasks,
            channels,
            self_weight=self_weight,
            cross_weight=cross_weight,
        )

    def _initialize_projections(
        self,
        num_tasks: int,
        channels: int,
        *,
        self_weight: float,
        cross_weight: float,
    ) -> None:
        with torch.no_grad():
            for output_task, module in enumerate(self.projections):
                projection = cast(nn.Conv2d, module)
                weight = projection.weight
                bias = cast(Tensor, projection.bias)
                weight.zero_()
                bias.zero_()
                for source_task in range(num_tasks):
                    value = self_weight if source_task == output_task else cross_weight
                    for channel in range(channels):
                        weight[channel, source_task * channels + channel, 0, 0] = value

    def forward(self, task_features: Sequence[Tensor]) -> list[Tensor]:
        fused = self.batch_norm(torch.cat(tuple(task_features), dim=1))
        projections = [cast(nn.Conv2d, module) for module in self.projections]
        weight = torch.cat([projection.weight for projection in projections], dim=0)
        bias = torch.cat([cast(Tensor, projection.bias) for projection in projections])
        projected = F.conv2d(fused, weight, bias)
        return list(projected.chunk(len(projections), dim=1))


class NDDRMTL(nn.Module):
    """NDDR multi-task CNN-BiGRU network returning one logit per task."""

    input_frequency_bins = 64

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 1,
        conv_channels: int = 64,
        gru_hidden_size: int = 128,
        num_gru_layers: int = 3,
        pool_sizes: Sequence[PoolSize] = _DEFAULT_POOL_SIZES,
        cnn_dropout: float = 0.25,
        gru_dropout: float = 0.2,
        nddr_self_weight: float = 0.6,
        nddr_cross_weight: float = 0.1,
        nddr_skip_weights: Sequence[float] = (0.2, 0.2, 0.6),
    ) -> None:
        super().__init__()
        self.num_classes = _positive_int("num_classes", num_classes)
        self.input_channels = _positive_int("input_channels", input_channels)
        self.conv_channels = _positive_int("conv_channels", conv_channels)
        self.gru_hidden_size = _positive_int("gru_hidden_size", gru_hidden_size)
        self.num_gru_layers = _positive_int("num_gru_layers", num_gru_layers)
        self.pool_sizes = _normalize_pool_sizes(pool_sizes)
        self.cnn_dropout = _dropout_probability("cnn_dropout", cnn_dropout)
        self.gru_dropout = _dropout_probability("gru_dropout", gru_dropout)
        self.nddr_self_weight = _non_negative_finite(
            "nddr_self_weight", nddr_self_weight
        )
        self.nddr_cross_weight = _non_negative_finite(
            "nddr_cross_weight", nddr_cross_weight
        )
        self.nddr_skip_weights = _normalize_skip_weights(nddr_skip_weights)
        self.final_frequency_bins = _pooled_frequency_bins(
            self.input_frequency_bins, self.pool_sizes
        )

        self.cnn_branches = nn.ModuleList(
            nn.ModuleList(
                _CNNBlock(
                    self.input_channels if stage == 0 else self.conv_channels,
                    self.conv_channels,
                    self.pool_sizes[stage],
                    self.cnn_dropout,
                )
                for stage in range(3)
            )
            for _ in range(self.num_classes)
        )
        self.nddr_layers = nn.ModuleList(
            _NDDRLayer(
                self.num_classes,
                self.conv_channels,
                self_weight=self.nddr_self_weight,
                cross_weight=self.nddr_cross_weight,
            )
            for _ in range(3)
        )

        skip_channels = 3 * self.conv_channels
        self.skip_batch_norms = nn.ModuleList(
            nn.BatchNorm2d(skip_channels) for _ in range(self.num_classes)
        )
        self.skip_fusion_convs = nn.ModuleList(
            nn.Conv2d(skip_channels, self.conv_channels, kernel_size=1)
            for _ in range(self.num_classes)
        )

        first_gru_input_size = self.conv_channels * self.final_frequency_bins
        self.gru_layers = nn.ModuleList(
            nn.ModuleList(
                nn.GRU(
                    input_size=first_gru_input_size if layer == 0 else self.gru_hidden_size,
                    hidden_size=self.gru_hidden_size,
                    batch_first=True,
                    bidirectional=True,
                )
                for layer in range(self.num_gru_layers)
            )
            for _ in range(self.num_classes)
        )
        self.classifiers = nn.ModuleList(
            nn.Linear(self.gru_hidden_size, 1) for _ in range(self.num_classes)
        )

        self._initialize_ordinary_layers()
        self._initialize_skip_fusion()

    def _initialize_ordinary_layers(self) -> None:
        for branch_module in self.cnn_branches:
            branch = cast(nn.ModuleList, branch_module)
            for block_module in branch:
                block = cast(_CNNBlock, block_module)
                nn.init.xavier_uniform_(block.conv.weight)
                nn.init.zeros_(cast(Tensor, block.conv.bias))
        for task_grus_module in self.gru_layers:
            task_grus = cast(nn.ModuleList, task_grus_module)
            for gru_module in task_grus:
                gru = cast(nn.GRU, gru_module)
                for name, parameter in gru.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(parameter)
                    elif "bias" in name:
                        nn.init.zeros_(parameter)
        for module in self.classifiers:
            classifier = cast(nn.Linear, module)
            nn.init.xavier_uniform_(classifier.weight)
            nn.init.zeros_(cast(Tensor, classifier.bias))

    def _initialize_skip_fusion(self) -> None:
        stage_weights = self.nddr_skip_weights
        with torch.no_grad():
            for module in self.skip_fusion_convs:
                projection = cast(nn.Conv2d, module)
                weight = projection.weight
                bias = cast(Tensor, projection.bias)
                weight.zero_()
                bias.zero_()
                for stage, value in enumerate(stage_weights):
                    for channel in range(self.conv_channels):
                        weight[
                            channel, stage * self.conv_channels + channel, 0, 0
                        ] = value

    def _run_cnn_stage(
        self,
        task_features: Sequence[Tensor],
        stage: int,
    ) -> list[Tensor]:
        """Run independent task convolutions as one grouped convolution."""

        blocks = [
            cast(_CNNBlock, cast(nn.ModuleList, self.cnn_branches[task])[stage])
            for task in range(self.num_classes)
        ]
        grouped_inputs = torch.cat(tuple(task_features), dim=1)
        weight = torch.cat([block.conv.weight for block in blocks], dim=0)
        bias = torch.cat([cast(Tensor, block.conv.bias) for block in blocks])
        convolved = F.conv2d(
            grouped_inputs,
            weight,
            bias,
            padding=2,
            groups=self.num_classes,
        )
        return [
            block.finish(features)
            for block, features in zip(
                blocks,
                convolved.chunk(self.num_classes, dim=1),
                strict=True,
            )
        ]

    def forward(self, inputs: Tensor) -> Tensor:
        """Compute logits with shape ``[batch, num_classes]``."""
        self._validate_input(inputs)
        self._validate_temporal_pooling(inputs.shape[3])

        task_features: list[Tensor] = [inputs for _ in range(self.num_classes)]
        stage_outputs: list[list[Tensor]] = []
        for stage, nddr_module in enumerate(self.nddr_layers):
            nddr_layer = cast(_NDDRLayer, nddr_module)
            next_features = self._run_cnn_stage(task_features, stage)
            task_features = nddr_layer(next_features)
            stage_outputs.append(task_features)

        target_size = stage_outputs[-1][0].shape[-2:]
        skip_inputs: list[Tensor] = []
        for task in range(self.num_classes):
            resized_stages = [
                features[task]
                if features[task].shape[-2:] == target_size
                else F.interpolate(
                    features[task],
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
                for features in stage_outputs
            ]
            skip_features = torch.cat(resized_stages, dim=1)
            batch_norm = cast(nn.BatchNorm2d, self.skip_batch_norms[task])
            skip_inputs.append(batch_norm(skip_features))

        skip_projections = [
            cast(nn.Conv2d, module) for module in self.skip_fusion_convs
        ]
        skip_weight = torch.cat(
            [projection.weight for projection in skip_projections], dim=0
        )
        skip_bias = torch.cat(
            [cast(Tensor, projection.bias) for projection in skip_projections]
        )
        grouped_skip_inputs = torch.cat(skip_inputs, dim=1)
        grouped_skip_outputs = F.conv2d(
            grouped_skip_inputs,
            skip_weight,
            skip_bias,
            groups=self.num_classes,
        )
        fused_features = list(
            grouped_skip_outputs.chunk(self.num_classes, dim=1)
        )

        logits: list[Tensor] = []
        for task, features in enumerate(fused_features):
            sequence = features.permute(0, 3, 1, 2).flatten(start_dim=2)
            task_grus = cast(nn.ModuleList, self.gru_layers[task])
            for gru_module in task_grus:
                gru = cast(nn.GRU, gru_module)
                sequence = F.dropout(
                    sequence, p=self.gru_dropout, training=self.training
                )
                sequence, _ = gru(sequence)
                batch, time, _ = sequence.shape
                sequence = sequence.reshape(
                    batch, time, 2, self.gru_hidden_size
                ).mean(dim=2)
            pooled = torch.amax(sequence, dim=1)
            classifier = cast(nn.Linear, self.classifiers[task])
            logits.append(classifier(pooled).squeeze(-1))

        return torch.stack(logits, dim=1)

    def _validate_input(self, inputs: Tensor) -> None:
        if not isinstance(inputs, Tensor):
            raise TypeError(f"inputs must be a torch.Tensor, got {type(inputs).__name__}")
        if inputs.ndim != 4:
            raise ValueError(
                "inputs must have shape [batch, channels, 64, time]; "
                f"received {tuple(inputs.shape)}"
            )
        if inputs.shape[0] < 1:
            raise ValueError("inputs must contain at least one sample")
        if inputs.shape[1] != self.input_channels:
            raise ValueError(
                f"expected {self.input_channels} input channels, got {inputs.shape[1]}"
            )
        if inputs.shape[2] != self.input_frequency_bins:
            raise ValueError(
                f"expected 64 frequency bins, got {inputs.shape[2]}"
            )
        if inputs.shape[3] < 1:
            raise ValueError("inputs must contain at least one time frame")
        if not inputs.is_floating_point():
            raise TypeError(f"inputs must use a floating-point dtype, got {inputs.dtype}")

    def _validate_temporal_pooling(self, time_frames: int) -> None:
        current = time_frames
        for stage, (_, temporal_pool) in enumerate(self.pool_sizes, start=1):
            if temporal_pool > current:
                raise ValueError(
                    f"pool size {self.pool_sizes[stage - 1]} at stage {stage} is too "
                    f"large for temporal dimension {current}"
                )
            current //= temporal_pool

    def nddr_parameters(self) -> Iterator[nn.Parameter]:
        """Yield 1x1 fusion weights regularized by the paper's NDDR L2 term."""
        for layer_module in self.nddr_layers:
            layer = cast(_NDDRLayer, layer_module)
            for projection_module in layer.projections:
                projection = cast(nn.Conv2d, projection_module)
                yield cast(nn.Parameter, projection.weight)
        for projection_module in self.skip_fusion_convs:
            projection = cast(nn.Conv2d, projection_module)
            yield cast(nn.Parameter, projection.weight)

    def base_parameters(self) -> Iterator[nn.Parameter]:
        """Yield all trainable parameters outside the NDDR fusion layers."""
        nddr_parameter_ids = {id(parameter) for parameter in self.nddr_parameters()}
        yield from (
            parameter
            for parameter in self.parameters()
            if id(parameter) not in nddr_parameter_ids
        )

    def optimizer_parameter_groups(
        self,
        base_weight_decay: float,
        nddr_weight_decay: float = 0.01,
    ) -> list[OptimizerParameterGroup]:
        """Return disjoint optimizer groups with NDDR-specific weight decay."""
        base_decay = _weight_decay("base_weight_decay", base_weight_decay)
        nddr_decay = _weight_decay("nddr_weight_decay", nddr_weight_decay)
        return [
            {
                "name": "base",
                "params": list(self.base_parameters()),
                "weight_decay": base_decay,
            },
            {
                "name": "nddr",
                "params": list(self.nddr_parameters()),
                "weight_decay": nddr_decay,
            },
        ]


def build_model(config: NDDRMTLConfig | Mapping[str, object]) -> NDDRMTL:
    """Build an :class:`NDDRMTL` from a dataclass or configuration mapping."""
    if isinstance(config, Mapping):
        try:
            config = NDDRMTLConfig(**cast(dict[str, Any], dict(config)))
        except TypeError as error:
            raise ValueError(f"invalid NDDRMTL configuration: {error}") from error
    if not isinstance(config, NDDRMTLConfig):
        raise TypeError("config must be an NDDRMTLConfig or a mapping")
    return NDDRMTL(
        num_classes=config.num_classes,
        input_channels=config.input_channels,
        conv_channels=config.conv_channels,
        gru_hidden_size=config.gru_hidden_size,
        num_gru_layers=config.num_gru_layers,
        pool_sizes=config.pool_sizes,
        cnn_dropout=config.cnn_dropout,
        gru_dropout=config.gru_dropout,
        nddr_self_weight=config.nddr_self_weight,
        nddr_cross_weight=config.nddr_cross_weight,
        nddr_skip_weights=config.nddr_skip_weights,
    )


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _dropout_probability(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number in [0, 1), got {value!r}")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability < 1.0:
        raise ValueError(f"{name} must be a number in [0, 1), got {value!r}")
    return probability


def _normalize_pool_sizes(pool_sizes: Sequence[PoolSize]) -> tuple[PoolSize, ...]:
    if (
        not isinstance(pool_sizes, Sequence)
        or isinstance(pool_sizes, (str, bytes))
        or len(pool_sizes) != 3
    ):
        raise ValueError("pool_sizes must contain exactly three (frequency, time) pairs")

    normalized: list[PoolSize] = []
    for stage, pool_size in enumerate(pool_sizes, start=1):
        if (
            not isinstance(pool_size, Sequence)
            or isinstance(pool_size, (str, bytes))
            or len(pool_size) != 2
        ):
            raise ValueError(
                f"pool size at stage {stage} must be a (frequency, time) pair"
            )
        frequency, time = pool_size
        normalized.append(
            (
                _positive_int(f"pool_sizes[{stage - 1}][0]", frequency),
                _positive_int(f"pool_sizes[{stage - 1}][1]", time),
            )
        )
    return tuple(normalized)


def _pooled_frequency_bins(
    frequency_bins: int, pool_sizes: Sequence[PoolSize]
) -> int:
    current = frequency_bins
    for stage, pool_size in enumerate(pool_sizes, start=1):
        frequency_pool = pool_size[0]
        if frequency_pool > current:
            raise ValueError(
                f"pool size {pool_size} at stage {stage} is too large for "
                f"frequency dimension {current}"
            )
        current //= frequency_pool
    return current


def _non_negative_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
    return normalized


def _normalize_skip_weights(values: Sequence[float]) -> tuple[float, float, float]:
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or len(values) != 3
    ):
        raise ValueError("nddr_skip_weights must contain exactly three values")
    normalized = tuple(
        _non_negative_finite(f"nddr_skip_weights[{index}]", value)
        for index, value in enumerate(values)
    )
    if not math.isclose(sum(normalized), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("nddr_skip_weights must sum to 1")
    return cast(tuple[float, float, float], normalized)


def _weight_decay(name: str, value: float) -> float:
    return _non_negative_finite(name, value)
