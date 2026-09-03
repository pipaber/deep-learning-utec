"""Generate waveform, log-Mel, and PCEN figures for representative clips."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from animal_audio.config import ExperimentConfig, load_config
from animal_audio.data import FILENAME_COLUMN, label_columns, load_metadata, load_wav
from animal_audio.features import AudioFeatureExtractor, FeatureMode

CATEGORY_ORDER: Final = ("none", "single", "multiple")
CATEGORY_TITLES: Final = {
    "none": "Sin especies etiquetadas",
    "single": "Una especie etiquetada",
    "multiple": "Múltiples especies etiquetadas",
}


@dataclass(frozen=True, slots=True)
class SelectedExample:
    """Metadata needed to render one representative audio clip."""

    category: str
    filename: str
    labels: tuple[str, ...]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate representative waveform/log-Mel/PCEN figures",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/nddr_pcen.yaml"),
        help="Experiment YAML used to reproduce frontend parameters",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/figures/audio_examples"),
        help="Directory receiving PNG figures and manifest.csv",
    )
    parser.add_argument(
        "--examples-per-category",
        type=int,
        default=1,
        help="Examples selected for each of: no labels, one label, multiple labels",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()



def select_examples(
    metadata: object,
    *,
    labels: list[str],
    examples_per_category: int,
    seed: int,
) -> list[SelectedExample]:
    """Select deterministic examples from the three label-concurrency groups."""

    import pandas as pd

    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame")
    if examples_per_category <= 0:
        raise ValueError("examples_per_category must be positive")

    label_counts = metadata[labels].sum(axis=1)
    masks = {
        "none": label_counts == 0,
        "single": label_counts == 1,
        "multiple": label_counts >= 2,
    }
    selected: list[SelectedExample] = []
    for offset, category in enumerate(CATEGORY_ORDER):
        candidates = metadata.loc[masks[category]]
        if candidates.empty:
            raise ValueError(f"dataset has no examples for category {category!r}")
        count = min(examples_per_category, len(candidates))
        sampled = candidates.sample(n=count, random_state=seed + offset).sort_values(
            FILENAME_COLUMN
        )
        for _, row in sampled.iterrows():
            positive_labels = tuple(label for label in labels if int(row[label]) == 1)
            selected.append(
                SelectedExample(
                    category=category,
                    filename=str(row[FILENAME_COLUMN]),
                    labels=positive_labels,
                )
            )
    return selected



def build_frontends(
    config: ExperimentConfig,
) -> tuple[AudioFeatureExtractor, AudioFeatureExtractor]:
    """Build CPU frontends that differ only in PCEN versus log-Mel compression."""

    feature = config.feature

    def build(mode: FeatureMode) -> AudioFeatureExtractor:
        return AudioFeatureExtractor(
            sample_rate=feature.sample_rate,
            n_fft=feature.n_fft,
            win_length=feature.win_length,
            hop_length=feature.hop_length,
            n_mels=feature.n_mels,
            f_min=feature.f_min,
            f_max=feature.f_max,
            mode=mode,
            offset=feature.pcen_bias,
            gain=feature.pcen_gain,
            power=feature.pcen_power,
            eps=feature.pcen_eps,
            smoothing=feature.pcen_smoothing,
            repeat_channels=1,
            expected_num_samples=round(
                feature.sample_rate * feature.duration_seconds
            ),
        ).eval()

    return build("pcen"), build("logmel")



def robust_limits(values: np.ndarray) -> tuple[float, float]:
    """Return non-degenerate percentile limits for a spectrogram color scale."""

    lower, upper = np.percentile(values, (1.0, 99.0))
    if not np.isfinite(lower) or not np.isfinite(upper):
        return 0.0, 1.0
    if upper <= lower:
        upper = lower + 1.0
    return float(lower), float(upper)



def safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")
    return normalized or "audio"



def plot_example(
    example: SelectedExample,
    waveform: np.ndarray,
    logmel: np.ndarray,
    pcen: np.ndarray,
    *,
    sample_rate: int,
    output_path: Path,
) -> None:
    """Save one three-panel comparison figure."""

    duration = waveform.size / sample_rate
    time = np.arange(waveform.size, dtype=np.float64) / sample_rate
    label_text = ", ".join(example.labels) if example.labels else "ninguna"

    figure, axes = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)
    figure.suptitle(
        f"{CATEGORY_TITLES[example.category]} · {example.filename}\nEtiquetas: {label_text}",
        fontsize=14,
        fontweight="bold",
    )

    axes[0].plot(time, waveform, color="steelblue", linewidth=0.65)
    axes[0].set(title="Forma de onda", ylabel="Amplitud", xlim=(0.0, duration))
    axes[0].grid(alpha=0.2)

    for axis, values, title, cmap in (
        (axes[1], logmel, "Espectrograma log-Mel", "magma"),
        (axes[2], pcen, "Representación PCEN usada por el modelo", "viridis"),
    ):
        vmin, vmax = robust_limits(values)
        image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=(0.0, duration, 0, values.shape[0] - 1),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        axis.set(title=title, ylabel="Índice Mel", xlim=(0.0, duration))
        figure.colorbar(image, ax=axis, pad=0.01, fraction=0.025)

    axes[-1].set_xlabel("Tiempo (s)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)



def write_manifest(examples: list[SelectedExample], paths: list[Path], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("category", "filename", "label_count", "labels", "figure"),
        )
        writer.writeheader()
        for example, path in zip(examples, paths, strict=True):
            writer.writerow(
                {
                    "category": example.category,
                    "filename": example.filename,
                    "label_count": len(example.labels),
                    "labels": ";".join(example.labels),
                    "figure": path.name,
                }
            )



def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    metadata = load_metadata(config.data.metadata_csv)
    labels = label_columns(metadata)
    examples = select_examples(
        metadata,
        labels=labels,
        examples_per_category=args.examples_per_category,
        seed=args.seed,
    )
    pcen_frontend, logmel_frontend = build_frontends(config)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    with torch.inference_mode():
        for index, example in enumerate(examples, start=1):
            wav_path = config.data.train_dir / example.filename
            waveform = load_wav(
                wav_path,
                sample_rate=config.feature.sample_rate,
                duration=config.feature.duration_seconds,
            )
            waveform_tensor = torch.from_numpy(waveform)
            pcen = pcen_frontend(waveform_tensor)[0, 0].cpu().numpy()
            logmel = logmel_frontend(waveform_tensor)[0, 0].cpu().numpy()
            output_path = output_dir / (
                f"{index:02d}_{example.category}_{safe_stem(example.filename)}.png"
            )
            plot_example(
                example,
                waveform,
                logmel,
                pcen,
                sample_rate=config.feature.sample_rate,
                output_path=output_path,
            )
            output_paths.append(output_path)
            print(output_path)

    manifest_path = output_dir / "manifest.csv"
    write_manifest(examples, output_paths, manifest_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
