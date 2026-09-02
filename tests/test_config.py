from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from animal_audio.config import ConfigError, ExperimentConfig, ModelConfig, load_config


class ConfigTests(unittest.TestCase):
    def test_default_42_class_cross_weight_preserves_unit_total(self) -> None:
        config = ModelConfig(num_classes=42, nddr_cross_weight=None)

        self.assertAlmostEqual(config.resolved_cross_weight, 0.4 / 41)
        total = config.nddr_self_weight + 41 * config.resolved_cross_weight
        self.assertAlmostEqual(total, 1.0)

    def test_logmel_ablation_loads_and_unknown_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0'\n", encoding="utf-8")
            config_path = root / "config.yaml"
            config_path.write_text("feature:\n  kind: logmel\n", encoding="utf-8")

            config = load_config(config_path)
            self.assertIsInstance(config, ExperimentConfig)
            self.assertEqual(config.feature.kind, "logmel")
            self.assertEqual(config.output_dir, (root / "artifacts/experiments/nddr_pcen").resolve())

            config_path.write_text("unknown: true\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "unknown configuration key"):
                load_config(config_path)


if __name__ == "__main__":
    unittest.main()
