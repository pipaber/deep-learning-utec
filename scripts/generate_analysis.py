"""Generate final comparison plots and error analysis for PCEN versus log-Mel."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from animal_audio.config import ExperimentConfig, load_config
from animal_audio.data import load_wav
from animal_audio.features import AudioFeatureExtractor

MODEL_COLORS = {"PCEN": "#4472C4", "log-Mel": "#ED7D31", "Baseline": "#A5A5A5"}
CONCURRENCY_ORDER = ("0", "1", "2", "3+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate comparison plots and error analysis",
    )
    parser.add_argument(
        "--pcen-dir",
        type=Path,
        default=Path("artifacts/experiments/nddr_pcen"),
    )
    parser.add_argument(
        "--logmel-dir",
        type=Path,
        default=Path("artifacts/experiments/nddr_logmel"),
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=Path("artifacts/split_seed42.csv"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/nddr_logmel.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/analysis"),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fixed evaluation threshold; only 0.5 is supported for consistency",
    )
    parser.add_argument("--error-examples-per-type", type=int, default=3)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def save_json(value: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def fixed_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("fixed_0_5")
    if not isinstance(value, dict):
        raise ValueError("metrics payload is missing fixed_0_5")
    return value


def load_inputs(
    pcen_dir: Path,
    logmel_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pcen_metrics = load_json(pcen_dir / "metrics.json")
    logmel_metrics = load_json(logmel_dir / "metrics.json")
    pcen_history = pd.read_csv(pcen_dir / "history.csv")
    logmel_history = pd.read_csv(logmel_dir / "history.csv")
    validation = pd.read_csv(logmel_dir / "validation_probabilities.csv")
    pcen_labels = fixed_metrics(pcen_metrics)["class_names"]
    logmel_labels = fixed_metrics(logmel_metrics)["class_names"]
    if pcen_labels != logmel_labels:
        raise ValueError("PCEN and log-Mel label order differs")
    labels = [str(label) for label in logmel_labels]
    required = ["filename", *labels, *(f"target_{label}" for label in labels)]
    missing = [column for column in required if column not in validation.columns]
    if missing:
        raise ValueError(f"validation probabilities are missing columns: {missing}")
    return pcen_metrics, logmel_metrics, pcen_history, logmel_history, validation


def plot_training_curves(
    pcen_history: pd.DataFrame,
    logmel_history: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for name, history in (("PCEN", pcen_history), ("log-Mel", logmel_history)):
        color = MODEL_COLORS[name]
        axes[0, 0].plot(history["epoch"], history["train_loss"], label=name, color=color)
        axes[0, 1].plot(
            history["epoch"], history["validation_loss"], label=name, color=color
        )
        axes[1, 0].plot(
            history["epoch"], history["validation_mAP"], label=name, color=color
        )
        axes[1, 1].semilogy(
            history["epoch"], history["learning_rate"], label=name, color=color
        )
    titles = (
        "Pérdida de entrenamiento",
        "Pérdida de validación",
        "mAP de validación",
        "Learning rate staircase",
    )
    ylabels = ("BCE ponderada", "BCE ponderada", "mAP", "Learning rate")
    for axis, title, ylabel in zip(axes.flat, titles, ylabels, strict=True):
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
    save_figure(figure, output_path)


def plot_model_comparison(
    pcen_metrics: dict[str, Any],
    logmel_metrics: dict[str, Any],
    output_path: Path,
) -> None:
    fixed_pcen = fixed_metrics(pcen_metrics)
    fixed_logmel = fixed_metrics(logmel_metrics)
    baseline = logmel_metrics["zero_baseline"]
    metric_keys = ("mAP", "micro_f1", "macro_f1", "exact_match")
    metric_names = ("mAP", "Micro F1", "Macro F1", "Exact match")
    models = ("Baseline", "PCEN", "log-Mel")
    sources = (baseline, fixed_pcen, fixed_logmel)
    positions = np.arange(len(metric_keys), dtype=np.float64)
    width = 0.24
    figure, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for index, (name, source) in enumerate(zip(models, sources, strict=True)):
        values = [float(source.get(key, 0.0) or 0.0) for key in metric_keys]
        offset = (index - 1) * width
        bars = axis.bar(
            positions + offset,
            values,
            width,
            label=name,
            color=MODEL_COLORS[name],
        )
        axis.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    axis.set(
        title="Comparación en validación con umbral 0.5",
        ylabel="Valor",
        xticks=positions,
        xticklabels=metric_names,
        ylim=(0.0, 0.8),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    save_figure(figure, output_path)


def build_per_class_table(
    pcen_metrics: dict[str, Any],
    logmel_metrics: dict[str, Any],
) -> pd.DataFrame:
    pcen = fixed_metrics(pcen_metrics)["per_class"]
    logmel = fixed_metrics(logmel_metrics)["per_class"]
    rows: list[dict[str, Any]] = []
    for label in fixed_metrics(logmel_metrics)["class_names"]:
        p = pcen[label]
        l = logmel[label]
        rows.append(
            {
                "label": label,
                "supported": bool(l["supported"]),
                "positive_count": int(l["positive_count"]),
                "prevalence": l["prevalence"],
                "pcen_ap": p["average_precision"],
                "logmel_ap": l["average_precision"],
                "delta_ap": (
                    None
                    if p["average_precision"] is None or l["average_precision"] is None
                    else float(l["average_precision"]) - float(p["average_precision"])
                ),
                "pcen_f1": p["f1"],
                "logmel_f1": l["f1"],
                "delta_f1": float(l["f1"]) - float(p["f1"]),
                "logmel_precision": l["precision"],
                "logmel_recall": l["recall"],
            }
        )
    return pd.DataFrame(rows)


def plot_per_class(
    table: pd.DataFrame,
    *,
    pcen_column: str,
    logmel_column: str,
    title: str,
    xlabel: str,
    output_path: Path,
) -> None:
    supported = table.loc[table["supported"]].sort_values(logmel_column)
    positions = np.arange(len(supported))
    height = max(8.0, len(supported) * 0.28)
    figure, axis = plt.subplots(figsize=(11, height), constrained_layout=True)
    axis.barh(
        positions - 0.18,
        supported[pcen_column],
        0.36,
        label="PCEN",
        color=MODEL_COLORS["PCEN"],
    )
    axis.barh(
        positions + 0.18,
        supported[logmel_column],
        0.36,
        label="log-Mel",
        color=MODEL_COLORS["log-Mel"],
    )
    axis.set(
        title=title,
        xlabel=xlabel,
        yticks=positions,
        yticklabels=supported["label"],
        xlim=(0.0, 1.0),
    )
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="lower right")
    save_figure(figure, output_path)


def plot_concurrency(
    pcen_metrics: dict[str, Any],
    logmel_metrics: dict[str, Any],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    metric_specs = (("micro_f1", "Micro F1"), ("exact_match", "Exact match"))
    x = np.arange(len(CONCURRENCY_ORDER))
    width = 0.35
    for axis, (metric, title) in zip(axes, metric_specs, strict=True):
        for index, (name, payload) in enumerate(
            (("PCEN", pcen_metrics), ("log-Mel", logmel_metrics))
        ):
            groups = payload["concurrency"]["fixed_0_5"]
            values = [float(groups[group][metric]) for group in CONCURRENCY_ORDER]
            offset = (index - 0.5) * width
            bars = axis.bar(
                x + offset,
                values,
                width,
                label=name,
                color=MODEL_COLORS[name],
            )
            axis.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
        axis.set(
            title=f"{title} por concurrencia",
            xlabel="Especies verdaderas por clip",
            ylabel=title,
            xticks=x,
            xticklabels=CONCURRENCY_ORDER,
            ylim=(0.0, 1.0),
        )
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    save_figure(figure, output_path)


def plot_class_distribution(split_csv: Path, labels: list[str], output_path: Path) -> None:
    split = pd.read_csv(split_csv, usecols=["split", *labels])
    train = split.loc[split["split"] == "train", labels].sum(axis=0)
    validation = split.loc[split["split"] == "val", labels].sum(axis=0)
    order = (train + validation).sort_values().index
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(11, 11), constrained_layout=True)
    axis.barh(
        positions - 0.18,
        train.loc[order],
        0.36,
        label="Train",
        color="#5B9BD5",
    )
    axis.barh(
        positions + 0.18,
        validation.loc[order],
        0.36,
        label="Validation",
        color="#70AD47",
    )
    axis.set(
        title="Distribución de positivos por especie",
        xlabel="Cantidad de clips positivos (escala log)",
        yticks=positions,
        yticklabels=order,
        xscale="symlog",
    )
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    save_figure(figure, output_path)


def error_rankings(
    validation: pd.DataFrame,
    labels: list[str],
    threshold: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    probabilities = validation[labels].to_numpy(dtype=np.float64)
    targets = validation[[f"target_{label}" for label in labels]].to_numpy(dtype=np.int8)
    predictions = probabilities >= threshold
    false_positive = (predictions == 1) & (targets == 0)
    false_negative = (predictions == 0) & (targets == 1)
    positive = targets.sum(axis=0)
    negative = targets.shape[0] - positive
    rows = []
    for index, label in enumerate(labels):
        fp = int(false_positive[:, index].sum())
        fn = int(false_negative[:, index].sum())
        rows.append(
            {
                "label": label,
                "positive_count": int(positive[index]),
                "negative_count": int(negative[index]),
                "false_positives": fp,
                "false_negatives": fn,
                "false_positive_rate": fp / negative[index] if negative[index] else None,
                "false_negative_rate": fn / positive[index] if positive[index] else None,
            }
        )
    return pd.DataFrame(rows), probabilities, targets, predictions


def plot_error_rankings(table: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    specifications = (
        ("false_positive_rate", "Mayor tasa de falsos positivos", "#C55A11"),
        ("false_negative_rate", "Mayor tasa de falsos negativos", "#A64D79"),
    )
    for axis, (column, title, color) in zip(axes, specifications, strict=True):
        selected = table.dropna(subset=[column]).nlargest(12, column).sort_values(column)
        axis.barh(selected["label"], selected[column], color=color)
        axis.set(title=title, xlabel="Tasa", xlim=(0.0, 1.0))
        axis.grid(axis="x", alpha=0.25)
    save_figure(figure, output_path)


def label_combination_table(
    targets: np.ndarray,
    predictions: np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    def combination(row: np.ndarray) -> str:
        selected = [labels[index] for index in np.flatnonzero(row)]
        return "+".join(selected) if selected else "ninguna"

    true_combinations = [combination(row) for row in targets]
    predicted_combinations = [combination(row) for row in predictions]
    frame = pd.DataFrame(
        {"true_combination": true_combinations, "predicted_combination": predicted_combinations}
    )
    frame["exact"] = frame["true_combination"] == frame["predicted_combination"]
    rows: list[dict[str, Any]] = []
    for true_combination, group in frame.groupby("true_combination", sort=False):
        mistakes = group.loc[~group["exact"], "predicted_combination"]
        common_error = Counter(mistakes).most_common(1)
        rows.append(
            {
                "true_combination": true_combination,
                "count": len(group),
                "exact_match": float(group["exact"].mean()),
                "most_common_wrong_prediction": (
                    common_error[0][0] if common_error else "—"
                ),
                "wrong_prediction_count": common_error[0][1] if common_error else 0,
            }
        )
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def plot_label_combinations(table: pd.DataFrame, output_path: Path) -> None:
    selected = table.head(12).sort_values("count")
    labels = [value if len(value) <= 42 else f"{value[:39]}..." for value in selected["true_combination"]]
    figure, axis = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    bars = axis.barh(labels, selected["exact_match"], color="#8064A2")
    axis.bar_label(
        bars,
        labels=[f"n={count}" for count in selected["count"]],
        fontsize=8,
        padding=3,
    )
    axis.set(
        title="Exact match de las combinaciones reales más frecuentes",
        xlabel="Exact match con umbral 0.5",
        xlim=(0.0, 1.08),
    )
    axis.grid(axis="x", alpha=0.25)
    save_figure(figure, output_path)


def build_logmel_frontend(config: ExperimentConfig) -> AudioFeatureExtractor:
    feature = config.feature
    return AudioFeatureExtractor(
        sample_rate=feature.sample_rate,
        n_fft=feature.n_fft,
        win_length=feature.win_length,
        hop_length=feature.hop_length,
        n_mels=feature.n_mels,
        f_min=feature.f_min,
        f_max=feature.f_max,
        mode="logmel",
        offset=feature.pcen_bias,
        gain=feature.pcen_gain,
        power=feature.pcen_power,
        eps=feature.pcen_eps,
        smoothing=feature.pcen_smoothing,
        repeat_channels=1,
        expected_num_samples=round(feature.sample_rate * feature.duration_seconds),
    ).eval()


def select_error_examples(
    validation: pd.DataFrame,
    labels: list[str],
    probabilities: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    count_per_type: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    used_filenames: set[str] = set()
    masks = {
        "FP": (predictions == 1) & (targets == 0),
        "FN": (predictions == 0) & (targets == 1),
    }
    scores = {"FP": probabilities, "FN": 1.0 - probabilities}
    for error_type in ("FP", "FN"):
        candidates = np.argwhere(masks[error_type])
        ordering = np.argsort(scores[error_type][masks[error_type]])[::-1]
        selected = 0
        for candidate_index in ordering:
            sample_index, class_index = candidates[candidate_index]
            filename = str(validation.iloc[sample_index]["filename"])
            if filename in used_filenames:
                continue
            true_labels = [
                labels[index] for index in np.flatnonzero(targets[sample_index])
            ]
            predicted_labels = [
                labels[index] for index in np.flatnonzero(predictions[sample_index])
            ]
            rows.append(
                {
                    "error_type": error_type,
                    "filename": filename,
                    "label": labels[class_index],
                    "probability": float(probabilities[sample_index, class_index]),
                    "true_labels": ";".join(true_labels),
                    "predicted_labels": ";".join(predicted_labels),
                }
            )
            used_filenames.add(filename)
            selected += 1
            if selected == count_per_type:
                break
    return pd.DataFrame(rows)


def plot_error_examples(
    examples: pd.DataFrame,
    config: ExperimentConfig,
    output_path: Path,
) -> None:
    frontend = build_logmel_frontend(config)
    columns = 2
    rows = max(1, int(np.ceil(len(examples) / columns)))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(14, rows * 3.0),
        constrained_layout=True,
        squeeze=False,
    )
    with torch.inference_mode():
        for axis, (_, example) in zip(axes.flat, examples.iterrows(), strict=False):
            waveform = load_wav(
                config.data.train_dir / str(example["filename"]),
                sample_rate=config.feature.sample_rate,
                duration=config.feature.duration_seconds,
            )
            logmel = frontend(torch.from_numpy(waveform))[0, 0].numpy()
            lower, upper = np.percentile(logmel, (1.0, 99.0))
            axis.imshow(
                logmel,
                origin="lower",
                aspect="auto",
                extent=(0.0, config.feature.duration_seconds, 0, logmel.shape[0] - 1),
                cmap="magma",
                vmin=lower,
                vmax=upper,
                interpolation="nearest",
            )
            axis.set(
                title=(
                    f"{example['error_type']} {example['label']} · "
                    f"p={example['probability']:.3f}\n{example['filename']}"
                ),
                xlabel="Tiempo (s)",
                ylabel="Índice Mel",
            )
        for axis in axes.flat[len(examples) :]:
            axis.set_visible(False)
    figure.suptitle("Errores de mayor confianza del modelo log-Mel", fontweight="bold")
    save_figure(figure, output_path)


def build_summary(
    pcen_metrics: dict[str, Any],
    logmel_metrics: dict[str, Any],
    per_class: pd.DataFrame,
    pcen_history: pd.DataFrame,
    logmel_history: pd.DataFrame,
) -> dict[str, Any]:
    keys = (
        "mAP",
        "label_accuracy",
        "exact_match",
        "hamming_loss",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    )
    pcen = fixed_metrics(pcen_metrics)
    logmel = fixed_metrics(logmel_metrics)
    supported = per_class.loc[per_class["supported"]].copy()
    return {
        "threshold": 0.5,
        "pcen_best_epoch": int(pcen_metrics["checkpoint_epoch"]),
        "logmel_best_epoch": int(logmel_metrics["checkpoint_epoch"]),
        "pcen": {key: float(pcen[key]) for key in keys},
        "logmel": {key: float(logmel[key]) for key in keys},
        "delta_logmel_minus_pcen": {
            key: float(logmel[key]) - float(pcen[key]) for key in keys
        },
        "zero_baseline": {
            key: float(logmel_metrics["zero_baseline"][key]) for key in keys
        },
        "supported_classes": int(logmel["supported_count"]),
        "unsupported_classes": per_class.loc[~per_class["supported"], "label"].tolist(),
        "top_logmel_ap_gains": supported.nlargest(5, "delta_ap")[
            ["label", "delta_ap"]
        ].to_dict(orient="records"),
        "top_logmel_ap_losses": supported.nsmallest(5, "delta_ap")[
            ["label", "delta_ap"]
        ].to_dict(orient="records"),
        "pcen_epochs": int(len(pcen_history)),
        "logmel_epochs": int(len(logmel_history)),
    }


def main() -> None:
    args = parse_args()
    if not np.isclose(args.threshold, 0.5):
        raise ValueError(
            "threshold must be 0.5 because metrics.json and concurrency artifacts "
            "use the paper's fixed threshold"
        )
    if args.error_examples_per_type <= 0:
        raise ValueError("error_examples_per_type must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pcen_metrics, logmel_metrics, pcen_history, logmel_history, validation = load_inputs(
        args.pcen_dir.expanduser().resolve(), args.logmel_dir.expanduser().resolve()
    )
    labels = [str(label) for label in fixed_metrics(logmel_metrics)["class_names"]]

    plot_training_curves(
        pcen_history, logmel_history, output_dir / "training_curves.png"
    )
    plot_model_comparison(
        pcen_metrics, logmel_metrics, output_dir / "model_comparison.png"
    )
    per_class = build_per_class_table(pcen_metrics, logmel_metrics)
    per_class.to_csv(output_dir / "per_class_metrics.csv", index=False)
    plot_per_class(
        per_class,
        pcen_column="pcen_ap",
        logmel_column="logmel_ap",
        title="Average Precision por especie",
        xlabel="Average Precision",
        output_path=output_dir / "per_class_ap.png",
    )
    plot_per_class(
        per_class,
        pcen_column="pcen_f1",
        logmel_column="logmel_f1",
        title="F1 por especie con umbral 0.5",
        xlabel="F1",
        output_path=output_dir / "per_class_f1.png",
    )
    plot_concurrency(pcen_metrics, logmel_metrics, output_dir / "concurrency.png")
    plot_class_distribution(args.split_csv, labels, output_dir / "class_distribution.png")

    rankings, probabilities, targets, predictions = error_rankings(
        validation, labels, args.threshold
    )
    rankings.to_csv(output_dir / "error_rankings.csv", index=False)
    plot_error_rankings(rankings, output_dir / "error_rankings.png")

    combinations = label_combination_table(targets, predictions, labels)
    combinations.to_csv(output_dir / "label_combination_errors.csv", index=False)
    plot_label_combinations(combinations, output_dir / "label_combinations.png")

    error_examples = select_error_examples(
        validation,
        labels,
        probabilities,
        targets,
        predictions,
        args.error_examples_per_type,
    )
    error_examples.to_csv(output_dir / "error_examples.csv", index=False)
    config = load_config(args.config)
    plot_error_examples(error_examples, config, output_dir / "error_examples.png")

    summary = build_summary(
        pcen_metrics, logmel_metrics, per_class, pcen_history, logmel_history
    )
    save_json(summary, output_dir / "summary.json")
    for path in sorted(output_dir.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
