"""Evaluation utilities for multi-label animal-audio classification."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
)

ArrayLike = Any


def _to_numpy(value: ArrayLike, *, name: str) -> np.ndarray:
    """Convert NumPy/torch-like inputs without retaining gradients."""

    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} cannot be converted to a NumPy array") from error


def _validate_targets(targets: ArrayLike) -> np.ndarray:
    array = _to_numpy(targets, name="targets")
    if array.ndim != 2:
        raise ValueError(f"targets must have shape (samples, classes), got {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("targets must contain at least one sample and one class")
    if not np.all(np.isfinite(array)):
        raise ValueError("targets must contain only finite values")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("targets must be binary (0 or 1)")
    return array.astype(np.int8, copy=False)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits.astype(np.float64, copy=False), -709.0, 709.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _resolve_probabilities(
    targets: np.ndarray,
    *,
    probabilities: ArrayLike | None,
    logits: ArrayLike | None,
) -> np.ndarray:
    if (probabilities is None) == (logits is None):
        raise ValueError("provide exactly one of probabilities or logits")

    source_name = "logits" if logits is not None else "probabilities"
    source = logits if logits is not None else probabilities
    scores = _to_numpy(source, name=source_name).astype(np.float64, copy=False)
    if scores.shape != targets.shape:
        raise ValueError(
            f"{source_name} shape {scores.shape} does not match targets shape {targets.shape}"
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"{source_name} must contain only finite values")
    if logits is not None:
        return _sigmoid(scores)
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("probabilities must be in [0, 1]")
    return scores


def _resolve_thresholds(threshold: float | Sequence[float], num_classes: int) -> np.ndarray:
    values = np.asarray(threshold, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(num_classes, float(values), dtype=np.float64)
    elif values.shape != (num_classes,):
        raise ValueError(f"threshold must be a scalar or have shape ({num_classes},)")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("thresholds must be finite and in [0, 1]")
    return values


def _resolve_class_names(class_names: Sequence[str] | None, num_classes: int) -> list[str]:
    names = (
        [f"class_{index}" for index in range(num_classes)]
        if class_names is None
        else [str(name) for name in class_names]
    )
    if len(names) != num_classes:
        raise ValueError(f"expected {num_classes} class names, got {len(names)}")
    if len(set(names)) != len(names):
        raise ValueError("class names must be unique")
    return names


def compute_multilabel_metrics(
    targets: ArrayLike,
    probabilities: ArrayLike | None = None,
    *,
    logits: ArrayLike | None = None,
    threshold: float | Sequence[float] = 0.5,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute fixed-threshold metrics for multi-label predictions.

    ``label_accuracy`` follows the paper's formula: the proportion of all
    sample/label decisions that are correct. Consequently, it equals one minus
    Hamming loss. Macro metrics and mAP average only classes with at least one
    positive target; unsupported classes have AP set to ``NaN``.
    """

    target_array = _validate_targets(targets)
    probability_array = _resolve_probabilities(
        target_array,
        probabilities=probabilities,
        logits=logits,
    )
    num_samples, num_classes = target_array.shape
    thresholds = _resolve_thresholds(threshold, num_classes)
    names = _resolve_class_names(class_names, num_classes)
    predictions = (probability_array >= thresholds[np.newaxis, :]).astype(np.int8)

    label_accuracy = float(np.mean(predictions == target_array))
    exact_match = float(np.mean(np.all(predictions == target_array, axis=1)))
    hamming_loss = float(np.mean(predictions != target_array))

    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        target_array.ravel(),
        predictions.ravel(),
        average="binary",
        zero_division=0,
    )
    per_precision, per_recall, per_f1, _ = precision_recall_fscore_support(
        target_array,
        predictions,
        average=None,
        zero_division=0,
    )

    positives = target_array.sum(axis=0, dtype=np.int64)
    negatives = num_samples - positives
    supported_mask = positives > 0
    supported_count = int(supported_mask.sum())
    prevalence = positives.astype(np.float64) / num_samples

    average_precision = np.full(num_classes, np.nan, dtype=np.float64)
    for class_index in np.flatnonzero(supported_mask):
        average_precision[class_index] = average_precision_score(
            target_array[:, class_index],
            probability_array[:, class_index],
        )

    if supported_count:
        macro_precision = float(np.mean(per_precision[supported_mask]))
        macro_recall = float(np.mean(per_recall[supported_mask]))
        macro_f1 = float(np.mean(per_f1[supported_mask]))
        mean_average_precision = float(np.mean(average_precision[supported_mask]))
    else:
        macro_precision = math.nan
        macro_recall = math.nan
        macro_f1 = math.nan
        mean_average_precision = math.nan

    per_class = {
        name: {
            "precision": float(per_precision[index]),
            "recall": float(per_recall[index]),
            "f1": float(per_f1[index]),
            "average_precision": float(average_precision[index]),
            "prevalence": float(prevalence[index]),
            "positive_count": int(positives[index]),
            "negative_count": int(negatives[index]),
            "supported": bool(supported_mask[index]),
            "threshold": float(thresholds[index]),
        }
        for index, name in enumerate(names)
    }

    return {
        "num_samples": num_samples,
        "num_classes": num_classes,
        "class_names": names,
        "thresholds": thresholds,
        "label_accuracy": label_accuracy,
        "exact_match": exact_match,
        "hamming_loss": hamming_loss,
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_precision": np.asarray(per_precision, dtype=np.float64),
        "per_class_recall": np.asarray(per_recall, dtype=np.float64),
        "per_class_f1": np.asarray(per_f1, dtype=np.float64),
        "per_class_average_precision": average_precision,
        "prevalence": prevalence,
        "supported_mask": supported_mask,
        "supported_count": supported_count,
        "map": mean_average_precision,
        "mAP": mean_average_precision,
        "per_class": per_class,
    }


def compute_metrics(
    outputs: ArrayLike,
    targets: ArrayLike,
    *,
    threshold: float | Sequence[float] = 0.5,
    class_names: Sequence[str] | None = None,
    from_logits: bool = True,
) -> dict[str, Any]:
    """Training-loop-friendly wrapper accepting outputs before targets."""

    if from_logits:
        return compute_multilabel_metrics(
            targets,
            logits=outputs,
            threshold=threshold,
            class_names=class_names,
        )
    return compute_multilabel_metrics(
        targets,
        probabilities=outputs,
        threshold=threshold,
        class_names=class_names,
    )


def zero_vector_baseline_metrics(
    targets: ArrayLike,
    *,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate the all-absent prediction baseline."""

    target_array = _validate_targets(targets)
    probabilities = np.zeros(target_array.shape, dtype=np.float64)
    return compute_multilabel_metrics(
        target_array,
        probabilities=probabilities,
        threshold=0.5,
        class_names=class_names,
    )


def _empty_metrics(num_classes: int, names: list[str], thresholds: np.ndarray) -> dict[str, Any]:
    nan_values = np.full(num_classes, np.nan, dtype=np.float64)
    supported = np.zeros(num_classes, dtype=bool)
    return {
        "num_samples": 0,
        "num_classes": num_classes,
        "class_names": names,
        "thresholds": thresholds.copy(),
        "label_accuracy": math.nan,
        "exact_match": math.nan,
        "hamming_loss": math.nan,
        "micro_precision": math.nan,
        "micro_recall": math.nan,
        "micro_f1": math.nan,
        "macro_precision": math.nan,
        "macro_recall": math.nan,
        "macro_f1": math.nan,
        "per_class_precision": nan_values.copy(),
        "per_class_recall": nan_values.copy(),
        "per_class_f1": nan_values.copy(),
        "per_class_average_precision": nan_values.copy(),
        "prevalence": nan_values.copy(),
        "supported_mask": supported,
        "supported_count": 0,
        "map": math.nan,
        "mAP": math.nan,
        "per_class": {
            name: {
                "precision": math.nan,
                "recall": math.nan,
                "f1": math.nan,
                "average_precision": math.nan,
                "prevalence": math.nan,
                "positive_count": 0,
                "negative_count": 0,
                "supported": False,
                "threshold": float(thresholds[index]),
            }
            for index, name in enumerate(names)
        },
    }


def metrics_by_concurrency(
    targets: ArrayLike,
    probabilities: ArrayLike | None = None,
    *,
    logits: ArrayLike | None = None,
    threshold: float | Sequence[float] = 0.5,
    class_names: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute metrics for samples containing 0, 1, 2, or 3+ positives."""

    target_array = _validate_targets(targets)
    probability_array = _resolve_probabilities(
        target_array,
        probabilities=probabilities,
        logits=logits,
    )
    num_classes = target_array.shape[1]
    thresholds = _resolve_thresholds(threshold, num_classes)
    names = _resolve_class_names(class_names, num_classes)
    concurrency = target_array.sum(axis=1)
    masks = {
        "0": concurrency == 0,
        "1": concurrency == 1,
        "2": concurrency == 2,
        "3+": concurrency >= 3,
    }

    results: dict[str, dict[str, Any]] = {}
    for level, mask in masks.items():
        if np.any(mask):
            results[level] = compute_multilabel_metrics(
                target_array[mask],
                probabilities=probability_array[mask],
                threshold=thresholds.tolist(),
                class_names=names,
            )
        else:
            results[level] = _empty_metrics(num_classes, names, thresholds)
    return results


def optimize_per_class_thresholds(
    targets: ArrayLike,
    probabilities: ArrayLike | None = None,
    *,
    logits: ArrayLike | None = None,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Choose per-class validation thresholds that maximize binary F1.

    Candidate thresholds are the observed probabilities, which cover every
    distinct prediction set under the module's ``>= threshold`` convention.
    When multiple thresholds attain the same F1, the lowest one is chosen to
    make the tie break deterministic and recall-favoring. Classes without a
    positive target are unsupported and receive threshold ``1.0``.
    """

    target_array = _validate_targets(targets)
    probability_array = _resolve_probabilities(
        target_array,
        probabilities=probabilities,
        logits=logits,
    )
    num_classes = target_array.shape[1]
    names = _resolve_class_names(class_names, num_classes)
    positives = target_array.sum(axis=0, dtype=np.int64)
    supported_mask = positives > 0
    thresholds = np.ones(num_classes, dtype=np.float64)
    best_f1 = np.full(num_classes, np.nan, dtype=np.float64)

    for class_index in np.flatnonzero(supported_mask):
        target = target_array[:, class_index]
        scores = probability_array[:, class_index]
        order = np.argsort(-scores, kind="stable")
        sorted_scores = scores[order]
        sorted_targets = target[order]
        true_positives = np.cumsum(sorted_targets, dtype=np.int64)
        predicted_positives = np.arange(1, len(sorted_scores) + 1, dtype=np.int64)
        false_positives = predicted_positives - true_positives
        false_negatives = positives[class_index] - true_positives
        denominators = 2 * true_positives + false_positives + false_negatives
        f1_at_rank = np.divide(
            2.0 * true_positives,
            denominators,
            out=np.zeros_like(denominators, dtype=np.float64),
            where=denominators != 0,
        )

        group_ends = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
        candidate_thresholds = sorted_scores[group_ends]
        candidate_f1 = f1_at_rank[group_ends]
        maximum = float(np.max(candidate_f1))
        tied = np.isclose(candidate_f1, maximum, rtol=0.0, atol=1e-12)
        thresholds[class_index] = float(np.min(candidate_thresholds[tied]))
        best_f1[class_index] = maximum

    by_class = {
        name: {
            "threshold": float(thresholds[index]),
            "f1": float(best_f1[index]),
            "supported": bool(supported_mask[index]),
        }
        for index, name in enumerate(names)
    }
    return {
        "class_names": names,
        "thresholds": thresholds,
        "best_f1": best_f1,
        "supported_mask": supported_mask,
        "supported_count": int(supported_mask.sum()),
        "by_class": by_class,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def metrics_to_jsonable(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Convert NumPy values and non-finite floats into strict JSON values."""

    return _json_safe(metrics)


def save_metrics_json(
    metrics: Mapping[str, Any],
    path: str | Path,
    *,
    indent: int = 2,
) -> Path:
    """Write metrics as strict JSON, representing undefined values as ``null``."""

    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics_to_jsonable(metrics), file, indent=indent, ensure_ascii=False, allow_nan=False)
        file.write("\n")
    return output_path


def _save_figure(figure: Any, output_path: str | Path | None) -> None:
    if output_path is None:
        return
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")


def plot_precision_recall_curves(
    targets: ArrayLike,
    probabilities: ArrayLike | None = None,
    *,
    logits: ArrayLike | None = None,
    class_names: Sequence[str] | None = None,
    output_path: str | Path | None = None,
    max_classes: int | None = None,
) -> Any:
    """Plot one PR curve per supported class and optionally save the figure."""

    import matplotlib.pyplot as plt

    target_array = _validate_targets(targets)
    probability_array = _resolve_probabilities(
        target_array,
        probabilities=probabilities,
        logits=logits,
    )
    names = _resolve_class_names(class_names, target_array.shape[1])
    supported_indices = np.flatnonzero(target_array.sum(axis=0) > 0)
    if max_classes is not None:
        if max_classes <= 0:
            raise ValueError("max_classes must be positive or null")
        supported_indices = supported_indices[:max_classes]

    figure, axis = plt.subplots(figsize=(8, 6))
    for class_index in supported_indices:
        precision, recall, _ = precision_recall_curve(
            target_array[:, class_index],
            probability_array[:, class_index],
        )
        ap = average_precision_score(
            target_array[:, class_index],
            probability_array[:, class_index],
        )
        axis.plot(recall, precision, label=f"{names[class_index]} (AP={ap:.3f})")

    axis.set(xlabel="Recall", ylabel="Precision", title="Per-class precision-recall curves")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.grid(alpha=0.25)
    if supported_indices.size:
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="small")
    else:
        axis.text(0.5, 0.5, "No supported classes", ha="center", va="center")
    _save_figure(figure, output_path)
    return figure


def plot_prevalence_vs_ap(
    targets: ArrayLike,
    probabilities: ArrayLike | None = None,
    *,
    logits: ArrayLike | None = None,
    class_names: Sequence[str] | None = None,
    output_path: str | Path | None = None,
    annotate: bool = True,
) -> Any:
    """Plot validation prevalence against AP for supported classes."""

    import matplotlib.pyplot as plt

    target_array = _validate_targets(targets)
    probability_array = _resolve_probabilities(
        target_array,
        probabilities=probabilities,
        logits=logits,
    )
    names = _resolve_class_names(class_names, target_array.shape[1])
    prevalence = target_array.mean(axis=0)
    supported_indices = np.flatnonzero(target_array.sum(axis=0) > 0)
    average_precision = np.array(
        [
            average_precision_score(target_array[:, index], probability_array[:, index])
            for index in supported_indices
        ],
        dtype=np.float64,
    )

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(prevalence[supported_indices], average_precision, alpha=0.8)
    if annotate:
        for point_index, class_index in enumerate(supported_indices):
            axis.annotate(
                names[class_index],
                (prevalence[class_index], average_precision[point_index]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize="small",
            )
    axis.set(
        xlabel="Positive prevalence",
        ylabel="Average precision",
        title="Class prevalence vs. average precision",
    )
    axis.set_xlim(left=0.0)
    axis.set_ylim(0.0, 1.05)
    axis.grid(alpha=0.25)
    _save_figure(figure, output_path)
    return figure


multilabel_metrics = compute_multilabel_metrics
zero_vector_baseline = zero_vector_baseline_metrics
optimize_thresholds = optimize_per_class_thresholds
plot_pr_curves = plot_precision_recall_curves
