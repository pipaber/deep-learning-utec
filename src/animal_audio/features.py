"""Dependency-light mel-spectrogram frontends for animal audio."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

FeatureMode = Literal["pcen", "logmel"]


def _hz_to_mel(frequency: torch.Tensor) -> torch.Tensor:
    """Convert Hz to the HTK mel scale."""

    return 2595.0 * torch.log10(1.0 + frequency / 700.0)


def _mel_to_hz(mels: torch.Tensor) -> torch.Tensor:
    """Convert HTK mel values to Hz."""

    return 700.0 * (torch.pow(10.0, mels / 2595.0) - 1.0)


def create_mel_filterbank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    """Construct a unit-height triangular mel filterbank from scratch.

    The returned float32 tensor has shape ``[n_mels, n_fft // 2 + 1]`` and is
    suitable for multiplication with a one-sided power spectrogram.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if n_fft < 2:
        raise ValueError("n_fft must be at least 2")
    if n_mels < 1:
        raise ValueError("n_mels must be at least 1")

    nyquist = sample_rate / 2.0
    upper_frequency = nyquist if f_max is None else float(f_max)
    if not 0.0 <= f_min < upper_frequency <= nyquist:
        raise ValueError("frequencies must satisfy 0 <= f_min < f_max <= sr / 2")

    frequency_bounds = torch.tensor(
        [float(f_min), upper_frequency], dtype=torch.float64
    )
    mel_bounds = _hz_to_mel(frequency_bounds)
    mel_points = torch.linspace(
        mel_bounds[0], mel_bounds[1], n_mels + 2, dtype=torch.float64
    )
    hz_points = _mel_to_hz(mel_points)
    fft_frequencies = torch.linspace(
        0.0, nyquist, n_fft // 2 + 1, dtype=torch.float64
    )

    lower = hz_points[:-2].unsqueeze(1)
    center = hz_points[1:-1].unsqueeze(1)
    upper = hz_points[2:].unsqueeze(1)
    rising = (fft_frequencies.unsqueeze(0) - lower) / (center - lower)
    falling = (upper - fft_frequencies.unsqueeze(0)) / (upper - center)
    filters = torch.minimum(rising, falling).clamp_min(0.0)

    if torch.any(filters.sum(dim=1) == 0):
        raise ValueError(
            "mel configuration creates empty filters; increase n_fft or reduce n_mels"
        )
    return filters.to(dtype=torch.float32)


class AudioFeatureExtractor(nn.Module):
    """Fixed STFT -> mel -> PCEN/log-mel audio frontend.

    Input is ``[N]`` or ``[B, N]`` and output is ``[B, C, n_mels, T]``. ``C``
    is one by default and can be set to three with ``repeat_channels=3`` for
    image backbones. PCEN smoothing uses ``M_0 = E_0`` followed by
    ``M_t = a M_(t-1) + (1-a) E_t``.
    """

    window: torch.Tensor
    mel_filterbank: torch.Tensor

    def __init__(
        self,
        *,
        sample_rate: int = 22_050,
        n_fft: int = 512,
        win_length: int = 512,
        hop_length: int = 128,
        n_mels: int = 64,
        f_min: float = 0.0,
        f_max: float | None = None,
        mode: FeatureMode = "pcen",
        offset: float = 0.05,
        gain: float = 0.98,
        power: float = 0.5,
        eps: float = 1e-4,
        smoothing: float = 0.967,
        log_eps: float = 1e-10,
        repeat_channels: int = 1,
        expected_num_samples: int | None = None,
    ) -> None:
        super().__init__()
        normalized_mode = mode.lower().replace("-", "").replace("_", "")
        if normalized_mode not in {"pcen", "logmel"}:
            raise ValueError("mode must be 'pcen' or 'logmel'")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not 0 < win_length <= n_fft:
            raise ValueError("win_length must satisfy 0 < win_length <= n_fft")
        if hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if offset < 0.0:
            raise ValueError("offset must be non-negative")
        if gain < 0.0:
            raise ValueError("gain must be non-negative")
        if power <= 0.0:
            raise ValueError("power must be positive")
        if eps <= 0.0 or log_eps <= 0.0:
            raise ValueError("eps and log_eps must be positive")
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must satisfy 0 <= smoothing < 1")
        if repeat_channels not in {1, 3}:
            raise ValueError("repeat_channels must be 1 or 3")
        if expected_num_samples is not None and expected_num_samples <= 0:
            raise ValueError("expected_num_samples must be positive when configured")

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.mode: FeatureMode = normalized_mode  # type: ignore[assignment]
        self.offset = float(offset)
        self.gain = float(gain)
        self.power = float(power)
        self.eps = float(eps)
        self.smoothing = float(smoothing)
        self.log_eps = float(log_eps)
        self.repeat_channels = repeat_channels
        self.expected_num_samples = expected_num_samples

        self.register_buffer(
            "window", torch.hann_window(win_length, periodic=True, dtype=torch.float32)
        )
        self.register_buffer(
            "mel_filterbank",
            create_mel_filterbank(
                sample_rate=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                f_min=f_min,
                f_max=f_max,
            ),
        )

    def _smooth_energy(self, energy: torch.Tensor) -> torch.Tensor:
        """Apply the previous-state exponential smoother without a Python loop.

        For ``M_t = a M_(t-1) + (1-a) E_t`` and ``M_0 = E_0``, multiplying
        by ``a^-t`` turns the recurrence into a cumulative sum. Float64 is used
        internally to keep the rescaling stable across the 513 time frames.
        """

        if self.smoothing == 0.0 or energy.shape[-1] == 1:
            return energy
        work = energy.to(dtype=torch.float64)
        time = torch.arange(
            energy.shape[-1],
            device=energy.device,
            dtype=torch.float64,
        )
        powers = torch.pow(self.smoothing, time)
        scaled = work / powers
        scaled = torch.cat((torch.zeros_like(scaled[..., :1]), scaled[..., 1:]), dim=-1)
        cumulative = torch.cumsum(scaled, dim=-1)
        smoothed = powers * (
            work[..., :1] + (1.0 - self.smoothing) * cumulative
        )
        return smoothed.to(dtype=energy.dtype)

    def _pcen(self, energy: torch.Tensor) -> torch.Tensor:
        smoother = self._smooth_energy(energy)
        denominator = (self.eps + smoother).pow(self.gain)
        stabilized = energy / denominator + self.offset
        return stabilized.clamp_min(0.0).pow(self.power) - math.pow(
            self.offset, self.power
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract float32 PCEN or log-mel features while preserving batch size."""

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim != 2:
            raise ValueError("waveform must have shape [N] or [B, N]")
        if waveform.shape[-1] == 0:
            raise ValueError("waveform must contain at least one sample")
        if (
            self.expected_num_samples is not None
            and waveform.shape[-1] != self.expected_num_samples
        ):
            raise ValueError(
                f"expected {self.expected_num_samples} samples, "
                f"found {waveform.shape[-1]}"
            )

        waveform = waveform.to(dtype=torch.float32)
        waveform = torch.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        power_spectrogram = spectrum.abs().square()
        power_spectrogram = torch.nan_to_num(
            power_spectrogram,
            nan=0.0,
            posinf=torch.finfo(torch.float32).max,
            neginf=0.0,
        )
        mel_energy = torch.einsum(
            "mf,bft->bmt", self.mel_filterbank, power_spectrogram
        ).clamp_min(0.0)

        if self.mode == "pcen":
            features = self._pcen(mel_energy)
        else:
            features = torch.log(mel_energy.clamp_min(self.log_eps))
        features = torch.nan_to_num(features)

        features = features.unsqueeze(1)
        if self.repeat_channels == 3:
            features = features.repeat(1, 3, 1, 1)
        return features
