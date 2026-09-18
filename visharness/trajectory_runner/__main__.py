"""CLI entry point for VisHarness trajectory inference/data generation."""

from __future__ import annotations

import argparse
import logging
import random

from .config import load_config
from .evaluator import TrajectoryEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VisHarness model trajectories.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a JSON or YAML trajectory-runner configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(42)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    config = load_config(args.config)
    evaluator = TrajectoryEvaluator(config)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
