import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from animal_audio.metrics import (
    compute_multilabel_metrics,
    metrics_by_concurrency,
    optimize_per_class_thresholds,
    save_metrics_json,
    zero_vector_baseline_metrics,
)


class MetricsTests(unittest.TestCase):
    def test_known_multilabel_metrics_match_paper_accuracy_formula(self) -> None:
        targets = np.array(
            [
                [1, 0, 1],
                [0, 1, 0],
                [0, 0, 0],
            ]
        )
        probabilities = np.array(
            [
                [0.9, 0.1, 0.4],
                [0.6, 0.8, 0.2],
                [0.1, 0.2, 0.3],
            ]
        )

        metrics = compute_multilabel_metrics(targets, probabilities, threshold=0.5)

        self.assertAlmostEqual(metrics["label_accuracy"], 7 / 9)
        self.assertAlmostEqual(metrics["exact_match"], 1 / 3)
        self.assertAlmostEqual(metrics["hamming_loss"], 2 / 9)
        self.assertAlmostEqual(metrics["label_accuracy"], 1 - metrics["hamming_loss"])
        self.assertAlmostEqual(metrics["micro_precision"], 2 / 3)
        self.assertAlmostEqual(metrics["micro_recall"], 2 / 3)
        self.assertAlmostEqual(metrics["micro_f1"], 2 / 3)
        self.assertAlmostEqual(metrics["macro_precision"], 0.5)
        self.assertAlmostEqual(metrics["macro_recall"], 2 / 3)
        self.assertAlmostEqual(metrics["macro_f1"], 5 / 9)
        np.testing.assert_allclose(
            metrics["per_class_average_precision"],
            [1.0, 1.0, 1.0],
        )
        self.assertAlmostEqual(metrics["map"], 1.0)

    def test_unsupported_class_has_nan_ap_and_is_excluded_from_macro(self) -> None:
        targets = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [0, 1, 0],
            ]
        )
        probabilities = np.array(
            [
                [0.9, 0.1, 0.9],
                [0.2, 0.8, 0.8],
                [0.7, 0.3, 0.7],
                [0.1, 0.6, 0.6],
            ]
        )

        metrics = compute_multilabel_metrics(
            targets,
            probabilities,
            class_names=["calls", "noise", "unsupported"],
        )

        np.testing.assert_array_equal(metrics["supported_mask"], [True, True, False])
        self.assertEqual(metrics["supported_count"], 2)
        self.assertTrue(np.isnan(metrics["per_class_average_precision"][2]))
        self.assertAlmostEqual(metrics["map"], 1.0)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0)
        self.assertTrue(np.isnan(metrics["per_class"]["unsupported"]["average_precision"]))

    def test_zero_vector_baseline(self) -> None:
        targets = np.array([[0, 0], [1, 0], [1, 1]])

        metrics = zero_vector_baseline_metrics(targets)

        self.assertAlmostEqual(metrics["label_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["exact_match"], 1 / 3)
        self.assertAlmostEqual(metrics["hamming_loss"], 0.5)
        self.assertEqual(metrics["micro_precision"], 0.0)
        self.assertEqual(metrics["micro_recall"], 0.0)
        self.assertEqual(metrics["micro_f1"], 0.0)

    def test_threshold_optimization_is_per_class_and_marks_unsupported(self) -> None:
        targets = np.array(
            [
                [1, 1, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ]
        )
        probabilities = np.array(
            [
                [0.40, 0.90, 0.9],
                [0.35, 0.80, 0.7],
                [0.30, 0.70, 0.5],
                [0.10, 0.10, 0.3],
            ]
        )

        result = optimize_per_class_thresholds(
            targets,
            probabilities,
            class_names=["a", "b", "missing"],
        )

        np.testing.assert_allclose(result["thresholds"], [0.35, 0.70, 1.0])
        np.testing.assert_allclose(result["best_f1"][:2], [1.0, 0.8])
        self.assertTrue(np.isnan(result["best_f1"][2]))
        np.testing.assert_array_equal(result["supported_mask"], [True, True, False])
        self.assertEqual(result["by_class"]["missing"]["threshold"], 1.0)
        self.assertTrue(np.isnan(result["by_class"]["missing"]["f1"]))
        self.assertFalse(result["by_class"]["missing"]["supported"])

    def test_metrics_by_concurrency_uses_required_groups(self) -> None:
        targets = np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
            ]
        )
        probabilities = np.where(targets == 1, 0.9, 0.1)

        grouped = metrics_by_concurrency(targets, probabilities)

        self.assertEqual(list(grouped), ["0", "1", "2", "3+"])
        self.assertEqual(
            [grouped[level]["num_samples"] for level in grouped],
            [1, 1, 1, 2],
        )
        for group_metrics in grouped.values():
            self.assertAlmostEqual(group_metrics["label_accuracy"], 1.0)
            self.assertAlmostEqual(group_metrics["exact_match"], 1.0)
        self.assertEqual(grouped["0"]["supported_count"], 0)
        self.assertTrue(np.isnan(grouped["0"]["map"]))

    def test_save_metrics_json_converts_numpy_and_nan(self) -> None:
        metrics = {
            "array": np.array([1.0, np.nan]),
            "count": np.int64(2),
            "supported": np.bool_(True),
        }
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "metrics.json"

            save_metrics_json(metrics, output_path)

            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                {
                    "array": [1.0, None],
                    "count": 2,
                    "supported": True,
                },
            )


if __name__ == "__main__":
    unittest.main()
