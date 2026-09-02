"""Command-line interface for NDDR-MTL animal-audio experiments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import ConfigError, load_config
from .engine import (
    EngineError,
    evaluate_checkpoint,
    extract_configured_archives,
    inspect_model,
    predict_test,
    prepare_split,
    train_experiment,
)
from .metrics import metrics_to_jsonable


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the experiment YAML configuration",
    )


def _add_device_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device (default: auto)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without executing any experiment work."""

    parser = argparse.ArgumentParser(
        prog="animal-audio",
        description="NDDR-MTL animal-audio experiment engine",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Create or validate the group-aware train/validation split",
    )
    _add_config_argument(prepare_parser)
    prepare_parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate the split even when one already exists",
    )
    prepare_parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract configured train/test 7z archives before validation",
    )
    prepare_parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Overwrite archive members while extracting (implies --extract)",
    )

    inspect_parser = subparsers.add_parser(
        "inspect-model",
        help="Report NDDR-MTL parameter counts without training",
    )
    _add_config_argument(inspect_parser)
    _add_device_argument(inspect_parser)
    inspect_parser.add_argument(
        "--dry-forward",
        action="store_true",
        help="Run one tiny eval-mode model forward pass",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Explicitly train the configured NDDR-MTL experiment",
    )
    _add_config_argument(train_parser)
    _add_device_argument(train_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a checkpoint on the validation split",
    )
    _add_config_argument(evaluate_parser)
    _add_device_argument(evaluate_parser)
    evaluate_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (default: configured best checkpoint)",
    )

    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict sorted test WAV files with a checkpoint",
    )
    _add_config_argument(predict_parser)
    _add_device_argument(predict_parser)
    predict_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (default: configured best checkpoint)",
    )
    predict_parser.add_argument(
        "--thresholds",
        type=Path,
        default=None,
        help="Optional threshold JSON from evaluate (default: 0.5)",
    )
    return parser


def _run_command(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)

    if args.command == "prepare":
        extraction = None
        if args.extract or args.force_extract:
            extraction = extract_configured_archives(
                config,
                force=args.force_extract,
            )
        prepared = prepare_split(config, force=args.force)
        return {
            "extraction": extraction,
            "split": str(prepared.split_path),
            "labels": len(prepared.labels),
            "train_samples": len(prepared.train_metadata),
            "validation_samples": len(prepared.validation_metadata),
            "globally_unsupported_classes": list(
                prepared.label_support.globally_unsupported
            ),
            "single_group_classes": list(
                prepared.label_support.single_group_classes
            ),
            "validation_unsupported_classes": list(
                prepared.label_support.validation_unsupported
            ),
        }

    if args.command == "inspect-model":
        return inspect_model(
            config,
            device=args.device,
            dry_forward=args.dry_forward,
        )

    if args.command == "train":
        summary = train_experiment(config, device=args.device)
        return {
            "output_dir": str(summary.output_dir),
            "best_checkpoint": (
                str(summary.best_checkpoint) if summary.best_checkpoint else None
            ),
            "last_checkpoint": (
                str(summary.last_checkpoint) if summary.last_checkpoint else None
            ),
            "best_mAP": summary.best_map,
            "epochs_completed": summary.epochs_completed,
            "stopped_early": summary.stopped_early,
        }

    if args.command == "evaluate":
        summary = evaluate_checkpoint(
            config,
            args.checkpoint,
            device=args.device,
        )
        return {
            "metrics": str(summary.metrics_path),
            "thresholds": str(summary.thresholds_json),
            "mAP": summary.map_score,
            "fixed_micro_f1": summary.fixed_micro_f1,
            "optimized_micro_f1": summary.optimized_micro_f1,
            "fixed_exact_match": summary.fixed_exact_match,
            "optimized_exact_match": summary.optimized_exact_match,
        }

    if args.command == "predict":
        summary = predict_test(
            config,
            args.checkpoint,
            thresholds_path=args.thresholds,
            device=args.device,
        )
        return {
            "probabilities": str(summary.probabilities_csv),
            "predictions": str(summary.predictions_csv),
            "samples": len(summary.filenames),
            "labels": len(summary.labels),
        }

    raise AssertionError(f"unhandled command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = _run_command(args)
    except (ConfigError, EngineError, FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            metrics_to_jsonable(result),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
