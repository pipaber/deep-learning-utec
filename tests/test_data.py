from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from animal_audio.data import (
    AnimalAudioDataset,
    AudioDecodingError,
    MetadataValidationError,
    create_group_aware_split,
    derive_recording_group,
    load_pcm_wav,
    load_split,
    save_split,
    summarize_labels,
    summarize_split_label_support,
    validate_group_disjointness,
    validate_metadata,
)


def _write_pcm_wav(
    path: Path,
    samples: np.ndarray,
    *,
    sample_rate: int,
    sample_width: int,
    channels: int = 1,
) -> None:
    flat_samples = np.asarray(samples).reshape(-1)
    if sample_width == 1:
        payload = flat_samples.astype(np.uint8).tobytes()
    elif sample_width == 2:
        payload = flat_samples.astype("<i2").tobytes()
    elif sample_width == 3:
        payload = b"".join(
            (int(value) & 0xFFFFFF).to_bytes(3, "little", signed=False)
            for value in flat_samples
        )
    elif sample_width == 4:
        payload = flat_samples.astype("<i4").tobytes()
    else:
        raise AssertionError("unsupported fixture width")

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(payload)


class MetadataTests(unittest.TestCase):
    def _metadata(self) -> pd.DataFrame:
        rows: list[dict[str, int | str]] = []
        for group_index in range(20):
            for window_index in range(2):
                start = window_index * 2
                rows.append(
                    {
                        "filename": (
                            f"recording_{group_index:02d}_{start}_{start + 3}.wav"
                        ),
                        "bird": group_index % 2,
                        "frog": (group_index // 2) % 2,
                        "insect": int(group_index % 5 == 0),
                    }
                )
        return pd.DataFrame(rows)

    def test_group_split_is_reproducible_and_has_no_leakage(self) -> None:
        metadata = self._metadata()
        first = create_group_aware_split(metadata, seed=42, n_candidates=80)
        second = create_group_aware_split(metadata, seed=42, n_candidates=80)

        pd.testing.assert_frame_equal(first, second)
        validate_group_disjointness(first)
        train_groups = {
            derive_recording_group(filename)
            for filename in first.loc[first["split"] == "train", "filename"]
        }
        val_groups = {
            derive_recording_group(filename)
            for filename in first.loc[first["split"] == "val", "filename"]
        }
        self.assertTrue(train_groups.isdisjoint(val_groups))
        self.assertAlmostEqual((first["split"] == "val").mean(), 0.2, places=7)

    def test_single_group_positive_is_kept_in_training_and_reported(self) -> None:
        metadata = self._metadata()
        metadata["singleton"] = 0
        metadata.loc[metadata["filename"].str.startswith("recording_00_"), "singleton"] = 1

        split = create_group_aware_split(metadata, seed=42, n_candidates=160)
        support = summarize_split_label_support(split)

        self.assertNotIn("singleton", support.train_unsupported)
        self.assertIn("singleton", support.single_group_classes)
        self.assertIn("singleton", support.validation_unsupported)

    def test_split_round_trip_preserves_validated_metadata(self) -> None:
        split = create_group_aware_split(self._metadata(), n_candidates=24)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.csv"
            self.assertEqual(save_split(split, path), path)
            loaded = load_split(path)
        pd.testing.assert_frame_equal(split, loaded)

    def test_malformed_binary_labels_are_rejected(self) -> None:
        metadata = pd.DataFrame(
            {
                "filename": ["a_0_3.wav", "b_0_3.wav"],
                "bird": [0, 2],
                "frog": [1, 0],
            }
        )
        with self.assertRaisesRegex(MetadataValidationError, "only 0 or 1"):
            validate_metadata(metadata, expected_label_count=2)

    def test_label_summary_reports_zero_support(self) -> None:
        metadata = pd.DataFrame(
            {
                "filename": ["a_0_3.wav", "b_0_3.wav"],
                "bird": [1, 0],
                "frog": [0, 0],
            }
        )
        report = summarize_labels(metadata)
        self.assertEqual(report.label_counts, {"bird": 1, "frog": 0})
        self.assertEqual(report.unsupported_classes, ("frog",))


class WavAndDatasetTests(unittest.TestCase):
    def test_all_supported_pcm_widths_decode_to_float32(self) -> None:
        sample_rate = 8_000
        fixtures = {
            1: (np.array([128, 255, 0]), np.array([0.0, 127 / 128, -1.0])),
            2: (
                np.array([0, 32_767, -32_768]),
                np.array([0.0, 32_767 / 32_768, -1.0]),
            ),
            3: (
                np.array([0, 8_388_607, -8_388_608]),
                np.array([0.0, 8_388_607 / 8_388_608, -1.0]),
            ),
            4: (
                np.array([0, 2_147_483_647, -2_147_483_648]),
                np.array([0.0, 2_147_483_647 / 2_147_483_648, -1.0]),
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            for sample_width, (samples, expected) in fixtures.items():
                with self.subTest(sample_width=sample_width):
                    path = Path(directory) / f"pcm_{sample_width}.wav"
                    _write_pcm_wav(
                        path,
                        samples,
                        sample_rate=sample_rate,
                        sample_width=sample_width,
                    )
                    waveform = load_pcm_wav(
                        path,
                        expected_sample_rate=sample_rate,
                        duration_seconds=3 / sample_rate,
                    )
                    self.assertEqual(waveform.dtype, np.float32)
                    np.testing.assert_allclose(waveform, expected, atol=1e-7)

    def test_stereo_is_averaged_and_short_audio_is_padded(self) -> None:
        sample_rate = 8_000
        stereo_frames = np.array([[32_000, 0], [-16_000, 16_000]], dtype=np.int16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            _write_pcm_wav(
                path,
                stereo_frames,
                sample_rate=sample_rate,
                sample_width=2,
                channels=2,
            )
            waveform = load_pcm_wav(
                path,
                expected_sample_rate=sample_rate,
                duration_seconds=5 / sample_rate,
            )

        self.assertEqual(waveform.shape, (5,))
        np.testing.assert_allclose(waveform[:2], [16_000 / 32_768, 0.0])
        np.testing.assert_array_equal(waveform[2:], np.zeros(3, dtype=np.float32))

    def test_wrong_sample_rate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            _write_pcm_wav(
                path,
                np.zeros(4, dtype=np.int16),
                sample_rate=8_000,
                sample_width=2,
            )
            with self.assertRaisesRegex(AudioDecodingError, "sample rate"):
                load_pcm_wav(path, expected_sample_rate=22_050)

    def test_dataset_train_and_test_items_are_collate_friendly(self) -> None:
        sample_rate = 8_000
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_pcm_wav(
                root / "clip_0_3.wav",
                np.zeros(4, dtype=np.int16),
                sample_rate=sample_rate,
                sample_width=2,
            )
            train_metadata = pd.DataFrame(
                {"filename": ["clip_0_3.wav"], "bird": [1], "frog": [0]}
            )
            train_dataset = AnimalAudioDataset(
                train_metadata,
                root,
                sample_rate=sample_rate,
                duration_seconds=4 / sample_rate,
            )
            test_dataset = AnimalAudioDataset(
                train_metadata[["filename"]],
                root,
                has_targets=False,
                sample_rate=sample_rate,
                duration_seconds=4 / sample_rate,
            )

            train_item = train_dataset[0]
            test_item = test_dataset[0]

        self.assertEqual(set(train_item), {"waveform", "target", "filename"})
        self.assertEqual(set(test_item), {"waveform", "filename"})
        self.assertEqual(train_item["filename"], "clip_0_3.wav")
        target = train_item["target"]
        self.assertIsInstance(target, torch.Tensor)
        if not isinstance(target, torch.Tensor):
            self.fail("training target must be a tensor")
        self.assertTrue(torch.equal(target, torch.tensor([1.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
