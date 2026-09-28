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
        default="/vepfs-dev/metro/hantao/nwp_bench/fyw/code/fyw/code/VisHarness-public/recipe/visharness/configs/trajectory_runner/Kimi_2.5_online_config.yaml",
        help="Path to a json/yaml config. The old tf_eval config shape is supported.",
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
