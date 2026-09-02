"""Tests for the NDDR multi-task audio model."""

from __future__ import annotations

import unittest
from typing import Any, cast

import torch
from torch import Tensor, nn

from animal_audio.model import NDDRMTL, NDDRMTLConfig, build_model


class NDDRMTLTest(unittest.TestCase):
    def make_model(self, num_classes: int = 2) -> NDDRMTL:
        return NDDRMTL(
            num_classes=num_classes,
            input_channels=1,
            conv_channels=2,
            gru_hidden_size=3,
            num_gru_layers=1,
            cnn_dropout=0.0,
            gru_dropout=0.0,
        )

    def test_output_shape_and_backward(self) -> None:
        torch.manual_seed(0)
        model = self.make_model()
        inputs = torch.randn(2, 1, 64, 4, requires_grad=True)

        logits = model(inputs)
        logits.sum().backward()

        self.assertEqual(logits.shape, (2, 2))
        input_gradient = inputs.grad
        self.assertIsNotNone(input_gradient)
        assert input_gradient is not None
        self.assertGreater(float(input_gradient.abs().sum()), 0.0)
        nddr_layer = cast(Any, model.nddr_layers[0])
        projection = cast(nn.Conv2d, nddr_layer.projections[0])
        projection_gradient = projection.weight.grad
        self.assertIsNotNone(projection_gradient)
        assert projection_gradient is not None
        self.assertGreater(float(projection_gradient.abs().sum()), 0.0)

    def test_special_initialization(self) -> None:
        model = self.make_model(num_classes=3)
        channels = model.conv_channels

        nddr_layer = cast(Any, model.nddr_layers[0])
        for output_task, module in enumerate(nddr_layer.projections):
            projection = cast(nn.Conv2d, module)
            weight = projection.weight
            bias = cast(Tensor, projection.bias)
            expected = torch.zeros_like(weight)
            for source_task in range(model.num_classes):
                value = 0.6 if source_task == output_task else 0.1
                for channel in range(channels):
                    expected[channel, source_task * channels + channel, 0, 0] = value
            torch.testing.assert_close(weight, expected)
            torch.testing.assert_close(bias, torch.zeros_like(bias))

        for module in model.skip_fusion_convs:
            projection = cast(nn.Conv2d, module)
            weight = projection.weight
            bias = cast(Tensor, projection.bias)
            expected = torch.zeros_like(weight)
            for stage, value in enumerate((0.2, 0.2, 0.6)):
                for channel in range(channels):
                    expected[channel, stage * channels + channel, 0, 0] = value
            torch.testing.assert_close(weight, expected)
            torch.testing.assert_close(bias, torch.zeros_like(bias))

    def test_optimizer_groups_are_disjoint_and_complete(self) -> None:
        config = NDDRMTLConfig(
            num_classes=2,
            conv_channels=2,
            gru_hidden_size=3,
            num_gru_layers=1,
        )
        model = build_model(config)
        groups = model.optimizer_parameter_groups(
            base_weight_decay=0.001,
            nddr_weight_decay=0.01,
        )

        self.assertEqual([group["weight_decay"] for group in groups], [0.001, 0.01])
        base_ids = {id(parameter) for parameter in groups[0]["params"]}
        nddr_ids = {id(parameter) for parameter in groups[1]["params"]}
        all_ids = {id(parameter) for parameter in model.parameters()}
        self.assertFalse(base_ids & nddr_ids)
        self.assertEqual(base_ids | nddr_ids, all_ids)

    def test_build_model_accepts_mapping(self) -> None:
        model = build_model(
            {
                "num_classes": 2,
                "input_channels": 1,
                "conv_channels": 2,
                "gru_hidden_size": 3,
                "num_gru_layers": 1,
            }
        )

        self.assertIsInstance(model, NDDRMTL)
        self.assertEqual(model.num_classes, 2)

    def test_rejects_non_64_frequency_input(self) -> None:
        model = self.make_model()

        with self.assertRaisesRegex(ValueError, "expected 64 frequency bins, got 63"):
            model(torch.randn(2, 1, 63, 4))


if __name__ == "__main__":
    unittest.main()
