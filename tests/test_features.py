from __future__ import annotations

import unittest

import torch

from animal_audio.features import AudioFeatureExtractor, create_mel_filterbank


class MelFilterbankTests(unittest.TestCase):
    def test_triangular_filterbank_has_expected_shape_and_support(self) -> None:
        filters = create_mel_filterbank(
            sample_rate=22_050,
            n_fft=512,
            n_mels=64,
        )
        self.assertEqual(filters.shape, (64, 257))
        self.assertEqual(filters.dtype, torch.float32)
        self.assertTrue(torch.all(filters >= 0))
        self.assertTrue(torch.all(filters.sum(dim=1) > 0))


class AudioFeatureExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.waveform = torch.randn(2, 4_096) * 0.05

    def test_default_pcen_shape_is_finite_and_buffers_are_registered(self) -> None:
        extractor = AudioFeatureExtractor()
        features = extractor(self.waveform)

        self.assertEqual(features.shape[:3], (2, 1, 64))
        self.assertEqual(features.shape[-1], 1 + (4_096 - 512) // 128)
        self.assertEqual(features.dtype, torch.float32)
        self.assertTrue(torch.isfinite(features).all())
        buffers = dict(extractor.named_buffers())
        self.assertIn("window", buffers)
        self.assertIn("mel_filterbank", buffers)

    def test_single_waveform_preserves_batch_dimension_and_can_repeat(self) -> None:
        extractor = AudioFeatureExtractor(
            n_fft=256,
            win_length=256,
            hop_length=64,
            n_mels=32,
            repeat_channels=3,
        )
        features = extractor(self.waveform[0])
        self.assertEqual(features.shape, (1, 3, 32, 61))

    def test_vectorized_smoother_matches_reference_recurrence(self) -> None:
        extractor = AudioFeatureExtractor(
            n_fft=256,
            win_length=256,
            hop_length=64,
            n_mels=32,
            smoothing=0.967,
        )
        energy = torch.rand(2, 3, 17)
        state = energy[..., 0]
        expected = [state]
        for index in range(1, energy.shape[-1]):
            state = 0.967 * state + 0.033 * energy[..., index]
            expected.append(state)

        actual = extractor._smooth_energy(energy)

        torch.testing.assert_close(actual, torch.stack(expected, dim=-1), rtol=1e-5, atol=1e-6)

    def test_pcen_and_logmel_are_distinct_and_finite(self) -> None:
        options = {
            "n_fft": 256,
            "win_length": 256,
            "hop_length": 64,
            "n_mels": 32,
        }
        pcen = AudioFeatureExtractor(mode="pcen", **options)(self.waveform)
        logmel = AudioFeatureExtractor(mode="logmel", **options)(self.waveform)

        self.assertEqual(pcen.shape, logmel.shape)
        self.assertTrue(torch.isfinite(pcen).all())
        self.assertTrue(torch.isfinite(logmel).all())
        self.assertFalse(torch.allclose(pcen, logmel))

    def test_zero_waveform_remains_finite_in_both_modes(self) -> None:
        waveform = torch.zeros(1_024)
        for mode in ("pcen", "logmel"):
            with self.subTest(mode=mode):
                features = AudioFeatureExtractor(
                    mode=mode,
                    n_fft=256,
                    win_length=256,
                    hop_length=64,
                    n_mels=32,
                )(waveform)
                self.assertTrue(torch.isfinite(features).all())

    def test_sample_length_is_checked_only_when_configured(self) -> None:
        AudioFeatureExtractor(n_fft=256, win_length=256, n_mels=32)(
            torch.zeros(1_000)
        )
        configured = AudioFeatureExtractor(
            n_fft=256,
            win_length=256,
            n_mels=32,
            expected_num_samples=2_000,
        )
        with self.assertRaisesRegex(ValueError, "expected 2000 samples"):
            configured(torch.zeros(1_000))


if __name__ == "__main__":
    unittest.main()
