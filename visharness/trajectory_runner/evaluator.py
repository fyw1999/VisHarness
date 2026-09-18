"""Evaluator-style wrapper for the VisHarness trajectory runner."""

from __future__ import annotations

import logging
from typing import Any

from .config import as_plain_dict
from .dataset import TrajectoryDataset
from .inferencer import BaseTrajectoryInferencer
from .model_client import OnlineVllmModelClient
from .serializer import TrajectorySerializer

logger = logging.getLogger(__name__)


class TrajectoryEvaluator:
    def __init__(self, config: Any):
        self.config = config
        mode = str(config.get("mode", "inference")).lower()
        if mode not in {"data_generation", "inference"}:
            raise ValueError("mode should be either data_generation or inference")

        model_args = as_plain_dict(config.get("model_args", {}))
        self.model = OnlineVllmModelClient(mode, **model_args)
        self.model.set_generation_config(config.get("generation_args", {}))
        benchmark_config = as_plain_dict(config.get("benchmark", {}))
        if not benchmark_config.get("model_config_path"):
            benchmark_config["model_config_path"] = (
                model_args.get("tokenizer_path")
                or model_args.get("model_path")
                or model_args.get("local_model_path")
            )
        if (
            benchmark_config.get("enabled")
            and not benchmark_config.get("vllm_metrics_url")
        ):
            base_url = str(
                model_args.get("base_url", "http://localhost:8000/v1")
            ).rstrip("/")
            if base_url.endswith("/v1"):
                base_url = base_url[:-3]
            benchmark_config["vllm_metrics_url"] = (
                f"{base_url.rstrip('/')}/metrics"
            )

        dataset_args = config.get("dataset_args", {})
        save_path = dataset_args.get("save_path") if isinstance(dataset_args, dict) else getattr(dataset_args, "save_path", None)
        if not save_path:
            raise ValueError(
                "dataset_args.save_path is required so trajectory-runner results can be checkpointed."
            )
        save_trajectory = (
            dataset_args.get("save_trajectory", False)
            if isinstance(dataset_args, dict)
            else getattr(dataset_args, "save_trajectory", False)
        )
        self.serializer = TrajectorySerializer(
            save_path=save_path,
            save_trajectory=save_trajectory,
        )

        self.inferencer = BaseTrajectoryInferencer(
            tp_model=self.model,
            batch_size=int(config.get("batch_size", 1)),
            max_rounds=int(config.get("max_rounds", 3)),
            mode=mode,
            model_family=config.get("model_family", "qwen3_vl"),
            archive_previous_images=bool(config.get("archive_previous_images", True)),
            serializer=self.serializer,
            controller_url_location=config.get("controller_url_location", None),
            max_consecutive_oom=int(config.get("max_consecutive_oom", 5)),
            benchmark_config=benchmark_config,
        )

    def evaluate(self):
        dataset = TrajectoryDataset(dataset_args=self.config.get("dataset_args", {}))
        logger.info("Loaded %d samples for VisHarness trajectory runner", len(dataset))
        return self.inferencer.parallel_batch_inference(dataset)
