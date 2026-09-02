from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from animal_audio.config import (
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    TrainingConfig,
)
from animal_audio.engine import (
    InferenceResult,
    build_experiment_components,
    calculate_pos_weight,
    create_test_metadata,
    load_experiment_checkpoint,
    predict_test,
    prepare_split,
    save_checkpoint,
    save_prediction_tables,
)


def _write_wav(path: Path, *, sample_rate: int = 8_000, samples: int = 512) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = np.zeros(samples, dtype="<i2").tobytes()
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(payload)


def _tiny_config(root: Path) -> ExperimentConfig:
    return ExperimentConfig(
        name="tiny_nddr",
        output_dir=root / "output",
        data=DataConfig(
            train_dir=root / "train",
            test_dir=root / "test",
            metadata_csv=root / "train.csv",
            split_csv=root / "split.csv",
        ),
        feature=FeatureConfig(
            sample_rate=8_000,
            duration_seconds=512 / 8_000,
            n_fft=512,
            win_length=512,
            hop_length=128,
            n_mels=64,
            kind="logmel",
            repeat_channels=1,
        ),
        model=ModelConfig(
            num_classes=2,
            input_channels=1,
            conv_channels=1,
            conv_dropout=0.0,
            gru_hidden_size=2,
            gru_layers=1,
            gru_dropout=0.0,
        ),
        training=TrainingConfig(
            batch_size=2,
            epochs=1,
            amp=False,
            num_workers=0,
            pin_memory=False,
        ),
    ).resolved(root)


def _write_training_fixture(root: Path) -> None:
    rows: list[dict[str, int | str]] = []
    for index in range(10):
        filename = f"recording_{index:02d}_0_3.wav"
        _write_wav(root / "train" / filename)
        rows.append(
            {
                "filename": filename,
                "zebra": index % 2,
                "ant": int(index % 3 == 0),
            }
        )
    pd.DataFrame(rows).to_csv(root / "train.csv", index=False)


class PosWeightTests(unittest.TestCase):
    def test_weights_are_clamped_and_zero_support_stays_one(self) -> None:
        metadata = pd.DataFrame(
            {
                "filename": ["a.wav", "b.wav", "c.wav", "d.wav"],
                "common": [1, 1, 0, 0],
                "rare": [1, 0, 0, 0],
                "missing": [0, 0, 0, 0],
            }
        )

        weights, support = calculate_pos_weight(
            metadata,
            ["common", "rare", "missing"],
            minimum=1.5,
            maximum=2.0,
        )

        torch.testing.assert_close(weights, torch.tensor([1.5, 2.0, 1.0]))
        self.assertEqual(support.positive_counts, (2, 1, 0))
        self.assertEqual(support.negative_counts, (2, 3, 4))
        self.assertEqual(support.raw_weights, (1.0, 3.0, 1.0))
        self.assertEqual(support.supported_mask, (True, True, False))
        self.assertEqual(support.unsupported_classes, ("missing",))


class SplitTests(unittest.TestCase):
    def test_prepare_split_preserves_label_order_and_validates_wavs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _tiny_config(root)
            _write_training_fixture(root)

            first = prepare_split(config)
            second = prepare_split(config)

            self.assertEqual(first.labels, ("zebra", "ant"))
            self.assertEqual(first.labels, second.labels)
            self.assertEqual(len(first.train_metadata), 8)
            self.assertEqual(len(first.validation_metadata), 2)
            pd.testing.assert_frame_equal(first.split_metadata, second.split_metadata)
            self.assertTrue(config.data.split_csv.is_file())

    def test_test_metadata_is_sorted_by_wav_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_wav(root / "z.wav")
            _write_wav(root / "a.wav")
            (root / "ignore.txt").write_text("not audio", encoding="utf-8")

            metadata = create_test_metadata(root)

            self.assertEqual(metadata["filename"].tolist(), ["a.wav", "z.wav"])


class CheckpointAndPredictionTests(unittest.TestCase):
    def test_checkpoint_round_trip_restores_model_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _tiny_config(root)
            model, _extractor, _device = build_experiment_components(config, "cpu")
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.fill_(0.125)
            checkpoint_path = root / "roundtrip.pt"

            save_checkpoint(
                checkpoint_path,
                epoch=3,
                model=model,
                labels=("zebra", "ant"),
                config=config,
                best_map=0.75,
            )
            loaded = load_experiment_checkpoint(
                config,
                checkpoint_path,
                device="cpu",
            )

            self.assertEqual(loaded.labels, ("zebra", "ant"))
            self.assertEqual(loaded.checkpoint["epoch"], 3)
            self.assertEqual(loaded.checkpoint["best_mAP"], 0.75)
            self.assertIsInstance(loaded.checkpoint["config"]["output_dir"], str)
            for expected, actual in zip(
                model.parameters(), loaded.model.parameters(), strict=True
            ):
                torch.testing.assert_close(expected, actual)

    def test_prediction_table_helper_preserves_filename_then_label_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = InferenceResult(
                logits=np.zeros((2, 2), dtype=np.float32),
                probabilities=np.array([[0.7, 0.2], [0.3, 0.9]], dtype=np.float32),
                targets=None,
                filenames=("a.wav", "z.wav"),
                mean_loss=None,
            )

            summary = save_prediction_tables(
                result,
                ("zebra", "ant"),
                (0.5, 0.8),
                directory,
            )

            probabilities = pd.read_csv(summary.probabilities_csv)
            predictions = pd.read_csv(summary.predictions_csv)
            self.assertEqual(
                probabilities.columns.tolist(), ["filename", "zebra", "ant"]
            )
            self.assertEqual(
                predictions.columns.tolist(), ["filename", "zebra", "ant"]
            )
            self.assertEqual(predictions["filename"].tolist(), ["a.wav", "z.wav"])
            np.testing.assert_array_equal(
                predictions[["zebra", "ant"]].to_numpy(), [[1, 0], [0, 1]]
            )

    def test_predict_test_runs_one_tiny_cpu_batch_in_sorted_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _tiny_config(root)
            _write_wav(root / "test" / "z.wav")
            _write_wav(root / "test" / "a.wav")
            model, _extractor, _device = build_experiment_components(config, "cpu")
            checkpoint_path = config.output_dir / config.training.checkpoint_filename
            save_checkpoint(
                checkpoint_path,
                epoch=1,
                model=model,
                labels=("zebra", "ant"),
                config=config,
                best_map=0.5,
            )

            summary = predict_test(config, device="cpu")

            self.assertEqual(summary.filenames, ("a.wav", "z.wav"))
            self.assertEqual(summary.labels, ("zebra", "ant"))
            self.assertEqual(summary.thresholds, (0.5, 0.5))
            self.assertEqual(
                pd.read_csv(summary.probabilities_csv).columns.tolist(),
                ["filename", "zebra", "ant"],
            )
            self.assertEqual(
                pd.read_csv(summary.predictions_csv)["filename"].tolist(),
                ["a.wav", "z.wav"],
            )


if __name__ == "__main__":
    unittest.main()
