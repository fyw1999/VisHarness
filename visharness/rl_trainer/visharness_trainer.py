"""Custom trainer that preserves trajectory-level GRPO and trains per turn."""

import hashlib
import json
import math
import os
import shutil
import time
import uuid
from collections.abc import Mapping, Sequence
from collections import Counter, defaultdict
from copy import deepcopy
from functools import partial
from pprint import pprint
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import kl_penalty
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import AdvantageEstimator
from verl.trainer.ppo.reward import extract_reward
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip
from verl.workers.utils.padding import left_right_2_no_padding

from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer

from visharness.data.task_balanced_sampler import (
    CyclingTaskSourcePool,
    TaskSamplingPlan,
    build_task_sampling_plan,
)
from visharness.agent_loop.dynamic_validation_manager import (
    _normalize_worker_output_fields,
)

from .checkpoint_retention import prune_resume_checkpoints
from .per_turn import (
    NoTrainablePerTurnSamplesError,
    SUPPORTED_LOCAL_COST_MODES,
    build_per_turn_grpo_batch,
    compute_turn_advantage_shaping,
    pad_per_turn_batch_to_divisor,
    plan_fixed_per_turn_mini_batches,
)
from .per_turn_loss import (
    annotate_effective_global_batch_size,
    rescale_trajectory_loss_weights_for_minibatches,
    visharness_ppo_loss,
)
from .validation_metrics import summarize_validation, write_validation_artifacts


def _align_dynamic_validation_reward_info(
    completed_results: dict[int, dict[str, Any]],
    *,
    total_jobs: int,
) -> dict[str, list[Any]]:
    """Align singleton validation reward fields by original sample index."""

    reward_extra_keys = sorted(
        {
            key
            for result in completed_results.values()
            for key in result["reward_extra_info"]
            if key != "reward"
        }
    )
    aligned: dict[str, list[Any]] = {
        "reward": [
            completed_results[sequence_index]["score"]
            for sequence_index in range(total_jobs)
        ]
    }
    for key in reward_extra_keys:
        aligned_values: list[Any] = []
        for sequence_index in range(total_jobs):
            raw_values = completed_results[sequence_index]["reward_extra_info"].get(key)
            if raw_values is None:
                values = []
            elif isinstance(raw_values, np.ndarray):
                values = raw_values.tolist()
            elif isinstance(raw_values, list):
                values = raw_values
            else:
                values = [raw_values]
            if len(values) > 1:
                raise RuntimeError(
                    "Each dynamic-validation job contains one trajectory, but "
                    f"reward field {key!r} returned {len(values)} values for "
                    f"sequence_index={sequence_index}"
                )
            aligned_values.append(values[0] if values else None)
        aligned[key] = aligned_values
    return aligned


class VisHarnessTrainer(RayDAPOTrainer):
    """Compute reward/GRPO advantages by trajectory, then update actor per turn."""

    def __init__(self, *args, **kwargs):
        self._visharness_collate_fn = kwargs.get("collate_fn")
        super().__init__(*args, **kwargs)
        if self._visharness_collate_fn is None:
            self._visharness_collate_fn = self.train_dataloader.collate_fn
        if str(self.config.algorithm.adv_estimator).lower() != "grpo":
            raise ValueError("VisHarness per-turn training currently supports algorithm.adv_estimator=grpo only")
        if self.use_critic:
            raise ValueError("VisHarness per-turn GRPO does not use a critic")
        if self.config.algorithm.use_kl_in_reward:
            raise ValueError(
                "VisHarness per-turn training cannot use trajectory-level KL-in-reward; "
                "use actor.use_kl_loss if KL regularization is needed"
            )
        if self.config.get("distillation", {}).get("enabled", False):
            raise ValueError("VisHarness per-turn training does not yet preserve per-turn teacher outputs")
        if self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False):
            raise ValueError("Disable actor.use_prefix_grouper because turns sharing a uid have different prompts")
        loss_agg_mode = str(self.config.actor_rollout_ref.actor.loss_agg_mode)
        supported_loss_agg_modes = {"token-mean", "seq-mean-token-mean"}
        if loss_agg_mode not in supported_loss_agg_modes:
            raise ValueError(
                "VisHarness per-turn training supports actor.loss_agg_mode in "
                f"{sorted(supported_loss_agg_modes)}, got {loss_agg_mode!r}"
            )
        if loss_agg_mode == "seq-mean-token-mean" and bool(self.config.actor_rollout_ref.actor.shuffle):
            raise ValueError(
                "VisHarness seq-mean DP normalization requires actor.shuffle=false so precomputed "
                "optimizer mini-batch denominators remain aligned after DP dispatch"
            )
        requires_sequence_mean = self._per_turn_loss_weight_correction_enabled()
        if requires_sequence_mean and loss_agg_mode != "seq-mean-token-mean":
            raise ValueError(
                "VisHarness trajectory-equal weighting and fixed per-event local costs require "
                "actor.loss_agg_mode=seq-mean-token-mean so every turn, rather than every token, "
                "has a stable scale"
            )
        # Validate hard-error costs at startup. A zero/non-finite cost would turn
        # a positive task advantage into zero rather than a strictly negative
        # hard-error update.
        self._get_per_turn_advantage_settings()
        task_sampling_config = self.config.get("visharness", {}).get("task_sampling", {})
        task_sampling_requested = bool(task_sampling_config.get("enable", False))
        self._get_per_turn_num_mini_batches()
        if self._group_filter_enabled():
            metric_name = str(self.config.algorithm.filter_groups.metric or "")
            if metric_name != "trajectory_reward":
                raise ValueError(
                    "VisHarness GRPO group filtering must use "
                    "algorithm.filter_groups.metric=trajectory_reward so the filter "
                    "uses task_reward minus any enabled trajectory step cost; "
                    f"got {metric_name!r}"
                )
            self._get_min_reward_std()
            if bool(
                self.config.actor_rollout_ref.rollout.get("skip_rollout", False)
            ):
                raise ValueError(
                    "Streaming group-filter rollout is incompatible with "
                    "actor_rollout_ref.rollout.skip_rollout=true because cached "
                    "fixed batches cannot be refilled one prompt group at a time"
                )
        if self._group_filter_enabled() or task_sampling_requested:
            min_trajectories = self._get_min_trajectories_after_rollout_filter()
            rollout_group_size = int(self.config.actor_rollout_ref.rollout.n)
            if min_trajectories > rollout_group_size:
                raise ValueError(
                    "visharness.filter_groups.min_trajectories_after_rollout_filter "
                    f"({min_trajectories}) cannot exceed actor_rollout_ref.rollout.n "
                    f"({rollout_group_size})"
                )
        rollout_correction = self.config.algorithm.get("rollout_correction", None)
        rollout_correction_enabled = rollout_correction is not None and (
            rollout_correction.get("rollout_is", None) is not None
            or rollout_correction.get("rollout_rs", None) is not None
            or rollout_correction.get("bypass_mode", False)
        )
        if rollout_correction_enabled:
            raise ValueError("Disable algorithm.rollout_correction until it is computed on per-turn samples")

        self._task_sampling_plan: TaskSamplingPlan | None = None
        if task_sampling_requested:
            train_batch_size = int(self.config.data.train_batch_size)
            configured_gen_batch_size = self.config.data.get("gen_batch_size", None)
            sampler_batch_size = int(
                configured_gen_batch_size
                if configured_gen_batch_size is not None
                else train_batch_size
            )
            if sampler_batch_size != train_batch_size:
                raise ValueError(
                    "Task-balanced sampling currently requires data.gen_batch_size "
                    "to equal data.train_batch_size so each optimizer update has "
                    f"one explicit source quota, got {sampler_batch_size} and "
                    f"{train_batch_size}"
                )
            self._task_sampling_plan = build_task_sampling_plan(
                self.train_dataset,
                alpha=float(task_sampling_config.get("alpha", 0.5)),
                batch_size=sampler_batch_size,
                epoch_size_mode="all_sources_once",
            )
            metric_source_names = [
                self._metric_source_name(source)
                for source in self._task_sampling_plan.sources
            ]
            if len(metric_source_names) != len(set(metric_source_names)):
                raise ValueError(
                    "Task data sources collide after metric-name normalization: "
                    f"{dict(zip(self._task_sampling_plan.sources, metric_source_names, strict=True))}"
                )
            if train_batch_size <= 0:
                raise ValueError("data.train_batch_size must be positive for task-balanced sampling")
            max_no_progress_cycles = int(
                task_sampling_config.get("max_no_progress_cycles", 3)
            )
            if max_no_progress_cycles <= 0:
                raise ValueError(
                    "visharness.task_sampling.max_no_progress_cycles must be positive, "
                    f"got {max_no_progress_cycles}"
                )
            if int(self.config.algorithm.filter_groups.max_num_gen_batches) > 0:
                raise ValueError(
                    "Task-balanced sampling requires "
                    "algorithm.filter_groups.max_num_gen_batches=0. Source-specific "
                    "top-up continues until the actor quota is filled; "
                    "max_no_progress_cycles provides the dead-loop guard."
                )
            task_sampling_seed = int(self.config.data.get("seed", 0))
            self._task_source_pool = CyclingTaskSourcePool(
                self._task_sampling_plan,
                seed=task_sampling_seed,
            )

    def init_workers(self):
        """Initialize Verl workers and install the VisHarness PPO loss wrapper."""

        super().init_workers()

        from verl.utils.config import omega_conf_to_dataclass

        actor_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.actor)
        self.actor_rollout_wg.set_loss_fn(partial(visharness_ppo_loss, config=actor_config))

    def _task_sampling_enabled(self) -> bool:
        return getattr(self, "_task_sampling_plan", None) is not None

    def _task_sampling_config(self):
        config = getattr(self, "config", {})
        return config.get("visharness", {}).get("task_sampling", {})

    def _task_sampling_max_no_progress_cycles(self) -> int:
        max_cycles = int(
            self._task_sampling_config().get("max_no_progress_cycles", 3)
        )
        if max_cycles <= 0:
            raise ValueError(
                "visharness.task_sampling.max_no_progress_cycles must be positive, "
                f"got {max_cycles}"
            )
        return max_cycles

    @staticmethod
    def _metric_source_name(source: str) -> str:
        source = str(source).strip().lower()
        if "/" in source:
            source = source.rsplit("/", 1)[-1]
        sanitized = "".join(
            character if character.isalnum() or character in "_-" else "_"
            for character in source
        )
        return sanitized or "unknown"

    def _task_quota_for_update(self, update_index: int) -> dict[str, int]:
        if not self._task_sampling_enabled():
            return {}
        return self._task_sampling_plan.quota_for_update(update_index)

    @staticmethod
    def _task_quota_deficits(
        target_counts: dict[str, int],
        kept_counts: dict[str, int],
    ) -> dict[str, int]:
        return {
            source: max(int(target) - int(kept_counts.get(source, 0)), 0)
            for source, target in target_counts.items()
            if int(target) > int(kept_counts.get(source, 0))
        }

    def _record_task_sampling_filter_progress(self, sampling_state: dict) -> None:
        """Raise when several source-size-equivalent attempts add no kept prompt."""

        max_cycles = self._task_sampling_max_no_progress_cycles()
        for source in self._task_sampling_plan.sources:
            attempted_since_progress = int(
                sampling_state["attempted_since_progress"].get(source, 0)
            )
            if attempted_since_progress <= 0:
                continue
            kept = int(sampling_state["kept"].get(source, 0))
            previous_kept = int(
                sampling_state["kept_at_progress_check"].get(source, 0)
            )
            if kept > previous_kept:
                sampling_state["attempted_since_progress"][source] = 0
                attempted_since_progress = 0
            sampling_state["kept_at_progress_check"][source] = kept
            source_size = int(self._task_sampling_plan.source_counts[source])
            no_progress_cycles = attempted_since_progress // source_size
            sampling_state["no_progress_cycles"][source] = no_progress_cycles
            if no_progress_cycles >= max_cycles:
                deficit = max(
                    int(sampling_state["original_target"].get(source, 0)) - kept,
                    0,
                )
                if self._group_filter_enabled():
                    selection_description = (
                        f"reward_std_filter=enabled, "
                        f"min_reward_std={self._get_min_reward_std()}, "
                        f"min_surviving_trajectories="
                        f"{self._get_min_trajectories_after_rollout_filter()}"
                    )
                else:
                    selection_description = (
                        "reward_std_filter=disabled, "
                        f"min_surviving_trajectories="
                        f"{self._get_min_trajectories_after_rollout_filter()}"
                    )
                raise RuntimeError(
                    "Task-balanced sampling made no selectable-group progress for "
                    f"{max_cycles} complete cycle(s) of source {source!r}. "
                    f"target={sampling_state['original_target'].get(source, 0)}, "
                    f"kept={kept}, remaining_deficit={deficit}, "
                    f"attempted={sampling_state['attempted'].get(source, 0)}, "
                    f"{selection_description}. The actor update was not performed. "
                    "Check rollout validity, group-size/reward filtering, or the task data."
                )

    def _summarize_task_sampling_metrics(
        self,
        sampling_state: dict,
        *,
        eligible_counts_by_source: dict[str, int],
    ) -> dict[str, float]:
        """Summarize target, sampling effort, and post-filter retention by source."""

        metrics: dict[str, float] = {}
        for source in self._task_sampling_plan.sources:
            metric_source = self._metric_source_name(source)
            sampling_prefix = f"sampling/by_data_source/{metric_source}"
            filter_prefix = f"filter/by_data_source/{metric_source}"
            original_target = int(sampling_state["original_target"].get(source, 0))
            attempted = int(sampling_state["attempted"].get(source, 0))
            topup_attempted = int(sampling_state["topup_attempted"].get(source, 0))
            kept = int(sampling_state["kept"].get(source, 0))
            eligible = int(eligible_counts_by_source.get(source, 0))
            progress = sampling_state["source_progress_snapshot"][source]
            metrics[f"{sampling_prefix}/target_prompt_groups"] = float(
                original_target
            )
            metrics[f"{sampling_prefix}/attempted_prompt_groups"] = float(attempted)
            metrics[f"{sampling_prefix}/topup_prompt_groups"] = float(
                topup_attempted
            )
            metrics[f"{sampling_prefix}/selected_prompt_groups"] = float(kept)
            metrics[f"{sampling_prefix}/post_filter_quota_shortfall"] = float(
                max(original_target - kept, 0)
            )
            metrics[f"{sampling_prefix}/topup_attempt_rate"] = (
                topup_attempted / attempted if attempted else 0.0
            )
            metrics[f"{sampling_prefix}/eligible_per_attempt_rate"] = (
                eligible / attempted if attempted else 0.0
            )
            metrics[f"{sampling_prefix}/selected_per_attempt_rate"] = (
                kept / attempted if attempted else 0.0
            )
            metrics[f"{sampling_prefix}/first_pass_coverage"] = float(
                progress["first_pass_coverage"]
            )
            metrics[f"{sampling_prefix}/cycles_completed"] = float(
                progress["cycles_completed"]
            )
            metrics[f"{sampling_prefix}/draws_in_epoch"] = float(
                progress["draws_in_epoch"]
            )
            metrics[f"{sampling_prefix}/cycles_crossed_this_update"] = float(
                sampling_state["completed_cycles_during_update"].get(source, 0)
            )
            metrics[f"{sampling_prefix}/no_progress_cycles"] = float(
                sampling_state["no_progress_cycles"].get(source, 0)
            )
            metrics[f"{filter_prefix}/reward_group_keep_rate"] = (
                kept / eligible if eligible else 0.0
            )
            metrics[f"{filter_prefix}/eligible_prompt_groups"] = float(eligible)
            metrics[f"{filter_prefix}/selected_prompt_groups"] = float(kept)
            metrics[f"{filter_prefix}/filtered_out_prompt_groups"] = float(
                max(eligible - kept, 0)
            )
        coverage_values = [
            float(progress["first_pass_coverage"])
            for progress in sampling_state["source_progress_snapshot"].values()
        ]
        metrics["sampling/task_epoch"] = float(sampling_state["task_epoch"])
        metrics["sampling/task_epoch_completed_after_update"] = float(
            bool(sampling_state["epoch_completed_after_update"])
        )
        metrics["sampling/min_source_first_pass_coverage"] = (
            min(coverage_values) if coverage_values else 0.0
        )
        metrics["sampling/mean_source_first_pass_coverage"] = (
            float(np.mean(coverage_values)) if coverage_values else 0.0
        )
        total_attempted = sum(int(value) for value in sampling_state["attempted"].values())
        total_topup = sum(int(value) for value in sampling_state["topup_attempted"].values())
        metrics["sampling/topup_prompt_rate"] = (
            total_topup / total_attempted if total_attempted else 0.0
        )
        return metrics

    def _build_per_turn_grpo_training_batch(self, batch: DataProto) -> tuple[DataProto, dict[str, float]]:
        trajectory_reward_settings = self._get_trajectory_reward_settings()
        advantage_settings = self._get_per_turn_advantage_settings()
        per_turn_batch, metrics = build_per_turn_grpo_batch(
            batch,
            tokenizer=self.tokenizer,
            processor=self.processor,
            max_prompt_length=self._get_per_turn_max_prompt_length(),
            trajectory_step_cost=trajectory_reward_settings["step_cost"],
            trajectory_equal_weight=self._trajectory_equal_weight_enabled(),
            local_cost_mode=self._get_per_turn_local_cost_mode(),
            **advantage_settings,
            norm_adv_by_std=bool(self.config.algorithm.get("norm_adv_by_std_in_grpo", True)),
        )
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")
        num_mini_batches = self._get_per_turn_num_mini_batches()
        per_turn_batch, pad_size = pad_per_turn_batch_to_divisor(
            per_turn_batch,
            dp_size * num_mini_batches,
        )
        metrics["visharness/per_turn/pre_balance_padding"] = float(pad_size)
        metrics["visharness/per_turn/padded_trainable_records"] = float(len(per_turn_batch))
        return per_turn_batch, metrics

    def _get_per_turn_max_prompt_length(self) -> int:
        visharness_config = self.config.get("visharness", {})
        per_turn_config = visharness_config.get("per_turn", {})
        return int(per_turn_config["max_prompt_length"])

    def _trajectory_equal_weight_enabled(self) -> bool:
        per_turn_config = self.config.get("visharness", {}).get("per_turn", {})
        return bool(per_turn_config.get("trajectory_equal_weight", False))

    def _per_turn_loss_weight_correction_enabled(self) -> bool:
        """Whether q/r policy weights are needed for ordinary per-turn GRPO."""

        return (
            self._trajectory_equal_weight_enabled()
            or self._get_per_turn_local_cost_mode() == "per_event_floor"
        )

    def _get_per_turn_num_mini_batches(self) -> int:
        """Return the fixed actor-update count for ordinary per-turn GRPO.

        When no explicit value is configured, preserve the number of PPO
        mini-batches that the unsplit trajectory batch would have used. The
        rollout group factor cancels from numerator and denominator, leaving
        ``train_batch_size // ppo_mini_batch_size``.
        """

        per_turn_config = self.config.get("visharness", {}).get("per_turn", {})
        configured_value = per_turn_config.get("num_mini_batches", None)
        if configured_value is not None:
            num_mini_batches = int(configured_value)
            if num_mini_batches <= 0:
                raise ValueError(
                    "visharness.per_turn.num_mini_batches must be positive, got "
                    f"{configured_value!r}"
                )
            return num_mini_batches

        train_batch_size = int(self.config.data.train_batch_size)
        ppo_mini_batch_size = int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size)
        if train_batch_size <= 0 or ppo_mini_batch_size <= 0:
            raise ValueError(
                "data.train_batch_size and actor.ppo_mini_batch_size must be positive to derive "
                "the ordinary GRPO per-turn mini-batch count, got "
                f"{train_batch_size=} {ppo_mini_batch_size=}"
            )
        if train_batch_size % ppo_mini_batch_size != 0:
            raise ValueError(
                "Cannot derive a fixed ordinary GRPO per-turn mini-batch count because "
                f"data.train_batch_size ({train_batch_size}) is not divisible by "
                f"actor.ppo_mini_batch_size ({ppo_mini_batch_size}). Set "
                "visharness.per_turn.num_mini_batches explicitly."
            )
        return train_batch_size // ppo_mini_batch_size

    def _get_trajectory_reward_settings(self) -> dict[str, float | bool]:
        visharness_config = self.config.get("visharness", {})
        reward_config = visharness_config.get("trajectory_reward", {})
        step_cost_enabled = bool(reward_config.get("enable_step_cost", True))
        configured_step_cost = abs(float(reward_config.get("step_cost", 0.2)))
        return {
            "enable_step_cost": step_cost_enabled,
            "configured_step_cost": configured_step_cost,
            "step_cost": configured_step_cost if step_cost_enabled else 0.0,
        }

    def _get_per_turn_advantage_settings(self) -> dict[str, float | int | bool]:
        visharness_config = self.config.get("visharness", {})
        per_turn_config = visharness_config.get("per_turn", {})
        advantage_config = visharness_config.get("per_turn_advantage", {})
        settings: dict[str, float | int | bool] = {
            "max_response_length": int(per_turn_config["max_response_length"]),
            "output_format_error_cost": abs(float(advantage_config.get("output_format_error_cost", 0.2))),
            "tool_args_error_cost": abs(float(advantage_config.get("tool_args_error_cost", 0.2))),
            "truncation_error_cost": abs(float(advantage_config.get("truncation_error_cost", 0.5))),
            "overlong_buffer_length": int(advantage_config.get("overlong_buffer_length", 512)),
            "overlong_cost_coef": abs(float(advantage_config.get("overlong_cost_coef", 0.5))),
            "soft_overlong_enabled": bool(advantage_config.get("soft_overlong_enabled", False)),
        }
        for config_name in (
            "output_format_error_cost",
            "tool_args_error_cost",
            "truncation_error_cost",
        ):
            value = float(settings[config_name])
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"visharness.per_turn_advantage.{config_name} must be finite and > 0 "
                    "so hard-error turns always have strictly negative advantage, "
                    f"got {value!r}"
                )
        return settings

    def _get_per_turn_local_cost_mode(self) -> str:
        advantage_config = self.config.get("visharness", {}).get("per_turn_advantage", {})
        mode = str(advantage_config.get("local_cost_mode", "per_event_floor")).strip().lower()
        if mode not in SUPPORTED_LOCAL_COST_MODES:
            raise ValueError(
                "visharness.per_turn_advantage.local_cost_mode must be one of "
                f"{sorted(SUPPORTED_LOCAL_COST_MODES)}, got {mode!r}"
            )
        if self._trajectory_equal_weight_enabled() and mode != "per_event_floor":
            raise ValueError(
                "trajectory_equal_weight=true requires local_cost_mode=per_event_floor so the "
                "trajectory weight applies only to task/base advantage and never scales local costs"
            )
        return mode

    @staticmethod
    def _trajectory_task_rewards(batch: DataProto) -> np.ndarray:
        if "task_reward" in batch.non_tensor_batch:
            return np.asarray(batch.non_tensor_batch["task_reward"], dtype=np.float64)
        if "token_level_scores" in batch.batch:
            return batch.batch["token_level_scores"].sum(dim=-1).detach().cpu().numpy().astype(np.float64)
        raise KeyError("Cannot find trajectory task rewards in task_reward or token_level_scores")

    def _compute_trajectory_reward_breakdown(self, batch: DataProto) -> dict[str, np.ndarray]:
        """Compute trajectory reward and local-error diagnostics.

        When the turn penalty is enabled, the trajectory reward is
        ``task_reward - step_cost * num_assistant_turns``; otherwise it is
        exactly ``task_reward``. Format, argument, truncation, and soft-length
        costs are intentionally excluded here and are applied only after
        trajectory-level GRPO normalization.
        """

        task_rewards = self._trajectory_task_rewards(batch)
        if "turn_records" not in batch.non_tensor_batch:
            raise KeyError("Cannot compute trajectory_reward because rollout output has no turn_records")

        trajectory_reward_settings = self._get_trajectory_reward_settings()
        advantage_settings = self._get_per_turn_advantage_settings()

        totals = []
        trajectory_step_costs = []
        local_advantage_costs = []
        format_error_costs = []
        tool_args_error_costs = []
        truncation_error_costs = []
        soft_overlong_costs = []
        scored_turn_counts = []
        output_format_success_counts = []
        output_format_error_counts = []
        raw_output_format_failure_counts = []
        tool_args_success_counts = []
        tool_args_error_counts = []
        raw_tool_args_failure_counts = []
        truncation_error_counts = []
        hard_error_counts = []
        soft_overlong_counts = []
        for trajectory_index, turn_records in enumerate(batch.non_tensor_batch["turn_records"]):
            turn_records = list(turn_records or [])
            trajectory_step_cost = float(trajectory_reward_settings["step_cost"]) * len(turn_records)
            local_advantage_cost = 0.0
            format_error_cost = 0.0
            tool_args_error_cost = 0.0
            truncation_error_cost = 0.0
            soft_overlong_cost = 0.0
            scored_turn_count = 0
            output_format_success_count = 0
            output_format_error_count = 0
            raw_output_format_failure_count = 0
            tool_args_success_count = 0
            tool_args_error_count = 0
            raw_tool_args_failure_count = 0
            truncation_error_count = 0
            hard_error_count = 0
            soft_overlong_count = 0

            for turn_record in turn_records:
                response_ids = list(turn_record.get("response_ids") or [])
                if not response_ids:
                    continue

                scored_turn_count += 1
                shaping = compute_turn_advantage_shaping(
                    turn_record,
                    **advantage_settings,
                )
                if not shaping["raw_output_format_failure"]:
                    output_format_success_count += 1
                if shaping["output_format_error"]:
                    output_format_error_count += 1
                raw_output_format_failure_count += int(
                    shaping["raw_output_format_failure"]
                )
                if not shaping["raw_output_format_failure"] and not shaping["raw_tool_args_failure"]:
                    tool_args_success_count += 1
                if shaping["tool_args_error"]:
                    tool_args_error_count += 1
                raw_tool_args_failure_count += int(shaping["raw_tool_args_failure"])
                truncation_error_count += int(shaping["truncated_by_length"])
                hard_error_count += int(shaping["hard_error"])
                soft_overlong_count += int(shaping["soft_overlong"])
                local_advantage_cost += float(shaping["local_cost"])
                format_error_cost += float(shaping["format_cost"])
                tool_args_error_cost += float(shaping["tool_args_cost"])
                truncation_error_cost += float(shaping["truncation_cost"])
                soft_overlong_cost += float(shaping["soft_overlong_cost"])

            total = float(task_rewards[trajectory_index]) - trajectory_step_cost
            totals.append(total)
            trajectory_step_costs.append(trajectory_step_cost)
            local_advantage_costs.append(local_advantage_cost)
            format_error_costs.append(format_error_cost)
            tool_args_error_costs.append(tool_args_error_cost)
            truncation_error_costs.append(truncation_error_cost)
            soft_overlong_costs.append(soft_overlong_cost)
            scored_turn_counts.append(scored_turn_count)
            output_format_success_counts.append(output_format_success_count)
            output_format_error_counts.append(output_format_error_count)
            raw_output_format_failure_counts.append(raw_output_format_failure_count)
            tool_args_success_counts.append(tool_args_success_count)
            tool_args_error_counts.append(tool_args_error_count)
            raw_tool_args_failure_counts.append(raw_tool_args_failure_count)
            truncation_error_counts.append(truncation_error_count)
            hard_error_counts.append(hard_error_count)
            soft_overlong_counts.append(soft_overlong_count)

            if output_format_error_count + tool_args_error_count + truncation_error_count != hard_error_count:
                raise RuntimeError(
                    "Trajectory hard-error category counts must be mutually exclusive and sum to "
                    f"hard_error_count for trajectory {trajectory_index}"
                )

        return {
            "trajectory_reward": np.asarray(totals, dtype=np.float64),
            "trajectory_step_cost": np.asarray(trajectory_step_costs, dtype=np.float64),
            "local_advantage_cost": np.asarray(local_advantage_costs, dtype=np.float64),
            "format_error_cost": np.asarray(format_error_costs, dtype=np.float64),
            "tool_args_error_cost": np.asarray(tool_args_error_costs, dtype=np.float64),
            "truncation_error_cost": np.asarray(truncation_error_costs, dtype=np.float64),
            "soft_overlong_cost": np.asarray(soft_overlong_costs, dtype=np.float64),
            "scored_turn_count": np.asarray(scored_turn_counts, dtype=np.float64),
            "output_format_success_count": np.asarray(output_format_success_counts, dtype=np.float64),
            "output_format_error_count": np.asarray(output_format_error_counts, dtype=np.float64),
            "raw_output_format_failure_count": np.asarray(
                raw_output_format_failure_counts,
                dtype=np.float64,
            ),
            "tool_args_success_count": np.asarray(tool_args_success_counts, dtype=np.float64),
            "tool_args_error_count": np.asarray(tool_args_error_counts, dtype=np.float64),
            "raw_tool_args_failure_count": np.asarray(
                raw_tool_args_failure_counts,
                dtype=np.float64,
            ),
            "truncation_error_count": np.asarray(truncation_error_counts, dtype=np.float64),
            "hard_error_count": np.asarray(hard_error_counts, dtype=np.float64),
            "soft_overlong_count": np.asarray(soft_overlong_counts, dtype=np.float64),
        }

    def _compute_trajectory_reward(self, batch: DataProto) -> np.ndarray:
        return self._compute_trajectory_reward_breakdown(batch)["trajectory_reward"]

    @staticmethod
    def _new_trajectory_metrics_accumulator() -> dict:
        return {
            "batch_count": 0,
            "trajectory_rewards": [],
            "task_rewards": [],
            "group_rewards": defaultdict(list),
            "group_sources": {},
            "interaction_turns": [],
            "assistant_response_lengths": [],
            "trajectory_step_costs": [],
            "local_advantage_costs": [],
            "reward_scored_turn_count": 0.0,
            "output_format_error_count": 0.0,
            "raw_output_format_failure_count": 0.0,
            "tool_args_error_count": 0.0,
            "raw_tool_args_failure_count": 0.0,
            "truncation_error_count": 0.0,
            "hard_error_count": 0.0,
            "soft_overlong_count": 0.0,
            "by_data_source": defaultdict(
                lambda: {
                    "trajectory_rewards": [],
                    "task_rewards": [],
                    "interaction_turns": [],
                    "scored_turn_count": 0.0,
                    "output_format_error_count": 0.0,
                    "raw_output_format_failure_count": 0.0,
                    "tool_args_error_count": 0.0,
                    "raw_tool_args_failure_count": 0.0,
                    "truncation_error_count": 0.0,
                    "hard_error_count": 0.0,
                    "soft_overlong_count": 0.0,
                }
            ),
        }

    def _accumulate_trajectory_metrics(self, accumulator: dict, batch: DataProto) -> None:
        """Collect lightweight trajectory statistics without retaining rollout payloads."""

        reward_breakdown = self._compute_trajectory_reward_breakdown(batch)
        trajectory_rewards = reward_breakdown["trajectory_reward"]
        batch.non_tensor_batch["trajectory_reward"] = trajectory_rewards
        task_rewards = self._trajectory_task_rewards(batch)

        batch_index = int(accumulator["batch_count"])
        accumulator["batch_count"] = batch_index + 1
        accumulator["trajectory_rewards"].extend(trajectory_rewards.tolist())
        accumulator["task_rewards"].extend(task_rewards.tolist())
        accumulator["trajectory_step_costs"].extend(reward_breakdown["trajectory_step_cost"].tolist())
        accumulator["local_advantage_costs"].extend(reward_breakdown["local_advantage_cost"].tolist())
        accumulator["reward_scored_turn_count"] += float(reward_breakdown["scored_turn_count"].sum())
        accumulator["output_format_error_count"] += float(reward_breakdown["output_format_error_count"].sum())
        accumulator["raw_output_format_failure_count"] += float(
            reward_breakdown["raw_output_format_failure_count"].sum()
        )
        accumulator["tool_args_error_count"] += float(reward_breakdown["tool_args_error_count"].sum())
        accumulator["raw_tool_args_failure_count"] += float(
            reward_breakdown["raw_tool_args_failure_count"].sum()
        )
        accumulator["truncation_error_count"] += float(reward_breakdown["truncation_error_count"].sum())
        accumulator["hard_error_count"] += float(reward_breakdown["hard_error_count"].sum())
        accumulator["soft_overlong_count"] += float(reward_breakdown["soft_overlong_count"].sum())

        uids = batch.non_tensor_batch.get("uid", np.arange(len(batch)))
        data_sources = batch.non_tensor_batch.get(
            "data_source",
            np.asarray(["unknown"] * len(batch), dtype=object),
        )
        for trajectory_index, (uid, reward, data_source) in enumerate(zip(
            uids,
            trajectory_rewards,
            data_sources,
            strict=True,
        )):
            group_key = (batch_index, self._prompt_uid_key(uid))
            accumulator["group_rewards"][group_key].append(float(reward))
            source = str(data_source)
            previous_source = accumulator["group_sources"].setdefault(group_key, source)
            if previous_source != source:
                raise ValueError(
                    f"Prompt group {group_key!r} contains multiple data sources: "
                    f"{previous_source!r} and {source!r}"
                )
            source_metrics = accumulator["by_data_source"][source]
            source_metrics["trajectory_rewards"].append(float(reward))
            source_metrics["task_rewards"].append(float(task_rewards[trajectory_index]))
            source_metrics["scored_turn_count"] += float(
                reward_breakdown["scored_turn_count"][trajectory_index]
            )
            source_metrics["output_format_error_count"] += float(
                reward_breakdown["output_format_error_count"][trajectory_index]
            )
            source_metrics["raw_output_format_failure_count"] += float(
                reward_breakdown["raw_output_format_failure_count"][trajectory_index]
            )
            source_metrics["tool_args_error_count"] += float(
                reward_breakdown["tool_args_error_count"][trajectory_index]
            )
            source_metrics["raw_tool_args_failure_count"] += float(
                reward_breakdown["raw_tool_args_failure_count"][trajectory_index]
            )
            source_metrics["truncation_error_count"] += float(
                reward_breakdown["truncation_error_count"][trajectory_index]
            )
            source_metrics["hard_error_count"] += float(
                reward_breakdown["hard_error_count"][trajectory_index]
            )
            source_metrics["soft_overlong_count"] += float(
                reward_breakdown["soft_overlong_count"][trajectory_index]
            )
        turn_records_batch = batch.non_tensor_batch.get("turn_records", [[] for _ in range(len(batch))])
        for data_source, turn_records in zip(data_sources, turn_records_batch, strict=True):
            turn_records = list(turn_records or [])
            accumulator["interaction_turns"].append(float(len(turn_records)))
            accumulator["by_data_source"][str(data_source)]["interaction_turns"].append(
                float(len(turn_records))
            )
            for turn_record in turn_records:
                response_ids = list(turn_record.get("response_ids") or [])
                response_length = int(turn_record.get("turn_response_length", len(response_ids)))
                accumulator["assistant_response_lengths"].append(float(response_length))

    @staticmethod
    def _add_distribution_metrics(metrics: dict, prefix: str, values: list[float]) -> None:
        values = np.asarray(values, dtype=np.float64)
        metrics[f"{prefix}_mean"] = float(values.mean()) if values.size else 0.0

    def _summarize_trajectory_metrics(self, accumulator: dict, prefix: str) -> dict[str, float]:
        metrics: dict[str, float] = {
            f"{prefix}/prompt_groups": float(len(accumulator["group_rewards"])),
            f"{prefix}/trajectories": float(len(accumulator["trajectory_rewards"])),
        }
        self._add_distribution_metrics(
            metrics,
            f"{prefix}/trajectory_reward",
            accumulator["trajectory_rewards"],
        )
        task_rewards = np.asarray(accumulator["task_rewards"], dtype=np.float64)
        metrics[f"{prefix}/task_reward_mean"] = float(task_rewards.mean()) if task_rewards.size else 0.0
        for metric_name, accumulator_key in (
            ("trajectory_step_cost", "trajectory_step_costs"),
            ("local_advantage_cost", "local_advantage_costs"),
        ):
            values = np.asarray(accumulator[accumulator_key], dtype=np.float64)
            metrics[f"{prefix}/{metric_name}_mean"] = float(values.mean()) if values.size else 0.0
        scored_turns = float(accumulator["reward_scored_turn_count"])
        metrics[f"{prefix}/output_format_error_rate"] = (
            float(accumulator["output_format_error_count"] / scored_turns) if scored_turns else 0.0
        )
        metrics[f"{prefix}/raw_output_format_failure_rate"] = (
            float(accumulator["raw_output_format_failure_count"] / scored_turns)
            if scored_turns
            else 0.0
        )
        metrics[f"{prefix}/tool_args_error_rate"] = (
            float(accumulator["tool_args_error_count"] / scored_turns) if scored_turns else 0.0
        )
        metrics[f"{prefix}/raw_tool_args_failure_rate"] = (
            float(accumulator["raw_tool_args_failure_count"] / scored_turns)
            if scored_turns
            else 0.0
        )
        metrics[f"{prefix}/truncation_error_rate"] = (
            float(accumulator["truncation_error_count"] / scored_turns) if scored_turns else 0.0
        )
        metrics[f"{prefix}/hard_error_turn_rate"] = (
            float(accumulator["hard_error_count"] / scored_turns) if scored_turns else 0.0
        )
        metrics[f"{prefix}/soft_overlong_rate"] = (
            float(accumulator["soft_overlong_count"] / scored_turns) if scored_turns else 0.0
        )
        group_reward_stds = [
            float(np.std(group_rewards, ddof=1)) if len(group_rewards) > 1 else 0.0
            for group_rewards in accumulator["group_rewards"].values()
        ]
        group_reward_stds = np.asarray(group_reward_stds, dtype=np.float64)
        metrics[f"{prefix}/group_trajectory_reward_std_mean"] = (
            float(group_reward_stds.mean()) if group_reward_stds.size else 0.0
        )
        self._add_distribution_metrics(metrics, f"{prefix}/interaction_turns", accumulator["interaction_turns"])
        self._add_distribution_metrics(
            metrics,
            f"{prefix}/assistant_response_length",
            accumulator["assistant_response_lengths"],
        )

        for source, source_metrics in accumulator["by_data_source"].items():
            metric_source = self._metric_source_name(source)
            source_prefix = f"{prefix}/by_data_source/{metric_source}"
            trajectory_rewards = np.asarray(
                source_metrics["trajectory_rewards"],
                dtype=np.float64,
            )
            task_rewards = np.asarray(source_metrics["task_rewards"], dtype=np.float64)
            interaction_turns = np.asarray(
                source_metrics["interaction_turns"],
                dtype=np.float64,
            )
            source_group_stds = np.asarray(
                [
                    float(np.std(group_rewards, ddof=1)) if len(group_rewards) > 1 else 0.0
                    for group_key, group_rewards in accumulator["group_rewards"].items()
                    if accumulator["group_sources"].get(group_key) == source
                ],
                dtype=np.float64,
            )
            metrics[f"{source_prefix}/prompt_groups"] = float(source_group_stds.size)
            metrics[f"{source_prefix}/task_reward_mean"] = (
                float(task_rewards.mean()) if task_rewards.size else 0.0
            )
            metrics[f"{source_prefix}/trajectory_reward_mean"] = (
                float(trajectory_rewards.mean()) if trajectory_rewards.size else 0.0
            )
            metrics[f"{source_prefix}/group_trajectory_reward_std_mean"] = (
                float(source_group_stds.mean()) if source_group_stds.size else 0.0
            )
            metrics[f"{source_prefix}/interaction_turns_mean"] = (
                float(interaction_turns.mean()) if interaction_turns.size else 0.0
            )
            scored_turn_count = float(source_metrics["scored_turn_count"])
            for metric_name, count_key in (
                ("output_format_error_rate", "output_format_error_count"),
                ("raw_output_format_failure_rate", "raw_output_format_failure_count"),
                ("tool_args_error_rate", "tool_args_error_count"),
                ("raw_tool_args_failure_rate", "raw_tool_args_failure_count"),
                ("truncation_error_rate", "truncation_error_count"),
                ("hard_error_turn_rate", "hard_error_count"),
                ("soft_overlong_rate", "soft_overlong_count"),
            ):
                metrics[f"{source_prefix}/{metric_name}"] = (
                    float(source_metrics[count_key]) / scored_turn_count
                    if scored_turn_count
                    else 0.0
                )
        return metrics

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        """Aggregate task metrics exactly as the standalone test evaluators do."""

        del sample_turns
        metrics, records = summarize_validation(
            data_sources,
            sample_uids,
            reward_extra_infos_dict,
        )
        self._visharness_last_validation_records = records
        return metrics

    def _dynamic_validation_enabled(self) -> bool:
        validation_config = self.config.get("visharness", {}).get("validation", {})
        return bool(validation_config.get("dynamic_scheduling", False))

    def _run_dynamic_validation(self, merged: bool = False):
        """Validate with a continuously refilled pool of single trajectories."""

        if self.use_rm:
            raise RuntimeError(
                "VisHarness dynamic validation does not support a colocated reward model; "
                "disable visharness.validation.dynamic_scheduling or use the standard validator"
            )
        if not hasattr(self.async_rollout_manager, "generate_validation_sequences"):
            raise RuntimeError(
                "Dynamic validation requires "
                "visharness.agent_loop.dynamic_validation_manager.VisHarnessAgentLoopManager"
            )

        validation_config = self.config.get("visharness", {}).get("validation", {})
        max_in_flight = int(validation_config.get("max_in_flight", 16))
        progress_interval = int(validation_config.get("progress_interval", 10))
        if max_in_flight <= 0:
            raise ValueError(
                f"visharness.validation.max_in_flight must be positive, got {max_in_flight}"
            )
        if progress_interval <= 0:
            raise ValueError(
                "visharness.validation.progress_interval must be positive, "
                f"got {progress_interval}"
            )

        repeat_times = int(self.config.actor_rollout_ref.rollout.val_kwargs.n)
        total_jobs = len(self.val_dataset) * repeat_times
        completed_results: dict[int, dict[str, Any]] = {}

        def validation_jobs():
            sequence_index = 0
            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)
                if "uid" not in test_batch.non_tensor_batch:
                    test_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(test_batch))],
                        dtype=object,
                    )

                test_batch = test_batch.repeat(
                    repeat_times=repeat_times,
                    interleave=True,
                )
                for item_index in range(len(test_batch)):
                    item_batch = test_batch.slice(item_index, item_index + 1)
                    # DataProto slices share meta_info by default. Validation
                    # jobs are independent, so detach it before any worker can
                    # add per-generation metadata.
                    item_batch.meta_info = {}
                    ground_truth = item_batch[0].non_tensor_batch.get(
                        "reward_model",
                        {},
                    ).get(
                        "ground_truth",
                        None,
                    )
                    generation_batch = self._get_gen_batch(item_batch)
                    generation_batch.meta_info = {
                        "eos_token_id": self.tokenizer.eos_token_id,
                        "pad_token_id": self.tokenizer.pad_token_id,
                        "recompute_log_prob": False,
                        "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                        "validate": True,
                        "global_steps": self.global_steps,
                    }
                    context = {
                        "test_batch": item_batch,
                        "ground_truth": ground_truth,
                    }
                    yield sequence_index, generation_batch, context
                    sequence_index += 1

            if sequence_index != total_jobs:
                raise RuntimeError(
                    f"Validation dataloader produced {sequence_index} trajectories, "
                    f"expected {total_jobs}"
                )

        def collect_result(
            sequence_index: int,
            context: dict[str, Any],
            generation_output: DataProto,
        ) -> None:
            if len(generation_output) != 1:
                raise RuntimeError(
                    "Dynamic validation jobs must return exactly one trajectory, "
                    f"got {len(generation_output)}"
                )

            output_ids = generation_output.batch["responses"][0]
            output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            test_batch = context["test_batch"].union(generation_output)
            test_batch.meta_info["validate"] = True
            input_ids = test_batch.batch["prompts"][0]
            input_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)

            reward_tensor, reward_extra_info = extract_reward(test_batch)
            scores = reward_tensor.sum(-1).cpu().tolist()
            if len(scores) != 1:
                raise RuntimeError(
                    "Dynamic validation reward must contain exactly one score, "
                    f"got {len(scores)}"
                )

            normalized_extra_info = {}
            for key, values in reward_extra_info.items():
                if isinstance(values, np.ndarray):
                    normalized_values = values.tolist()
                elif isinstance(values, list):
                    normalized_values = values
                else:
                    normalized_values = [values]
                normalized_extra_info[key] = normalized_values

            completed_results[sequence_index] = {
                "input": input_text,
                "output": output_text,
                "ground_truth": context["ground_truth"],
                "score": scores[0],
                "uid": test_batch.non_tensor_batch["uid"][0],
                "turns": test_batch.non_tensor_batch.get("__num_turns__"),
                "data_source": test_batch.non_tensor_batch.get(
                    "data_source",
                    np.array(["unknown"], dtype=object),
                ),
                "reward_extra_info": normalized_extra_info,
            }

        scheduler_stats = self.async_rollout_manager.generate_validation_sequences(
            validation_jobs(),
            max_in_flight=max_in_flight,
            on_complete=collect_result,
            total_jobs=total_jobs,
            progress_interval=progress_interval,
        )
        self._visharness_last_validation_scheduler_stats = scheduler_stats

        if len(completed_results) != total_jobs:
            raise RuntimeError(
                f"Collected {len(completed_results)} validation results, expected {total_jobs}"
            )

        data_source_lst = []
        reward_extra_infos_dict = _align_dynamic_validation_reward_info(
            completed_results,
            total_jobs=total_jobs,
        )
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for sequence_index in range(total_jobs):
            result = completed_results[sequence_index]
            sample_inputs.append(result["input"])
            sample_outputs.append(result["output"])
            sample_gts.append(result["ground_truth"])
            sample_scores.append(result["score"])
            sample_uids.append(result["uid"])
            if result["turns"] is not None:
                sample_turns.append(result["turns"])
            data_source_lst.append(result["data_source"])

        self._maybe_log_val_generations(
            inputs=sample_inputs,
            outputs=sample_outputs,
            scores=sample_scores,
        )

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, values in reward_extra_infos_dict.items():
            assert len(values) == 0 or len(values) == len(sample_scores), (
                f"{key_info}: len(values)={len(values)}, "
                f"len(sample_scores)={len(sample_scores)}"
            )

        if merged:
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        if not data_source_lst:
            return {}
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(
            data_sources,
            sample_uids,
            reward_extra_infos_dict,
            sample_turns,
        )

    def _validate(self, merged: bool = False):
        """Run validation, add wall-clock statistics, and persist compact results."""

        started = time.perf_counter()
        self._visharness_last_validation_records = []
        self._visharness_last_validation_scheduler_stats = {}
        if self._dynamic_validation_enabled():
            result = self._run_dynamic_validation(merged=merged)
        else:
            result = super()._validate(merged=merged)
        if merged:
            return result

        metrics = dict(result or {})
        elapsed = time.perf_counter() - started
        records = list(getattr(self, "_visharness_last_validation_records", []) or [])
        metrics["val-aux/runtime/wall_seconds"] = float(elapsed)
        metrics["val-aux/runtime/trajectories_per_second"] = (
            float(len(records) / elapsed) if elapsed > 0 else 0.0
        )
        scheduler_stats = getattr(
            self,
            "_visharness_last_validation_scheduler_stats",
            {},
        )
        if scheduler_stats:
            metrics["val-aux/runtime/scheduler_peak_in_flight"] = float(
                scheduler_stats["peak_in_flight"]
            )
            metrics["val-aux/runtime/scheduler_workers"] = float(
                scheduler_stats["worker_count"]
            )

        validation_config = self.config.get("visharness", {}).get("validation", {})
        if bool(validation_config.get("save_detailed_results", True)):
            output_root = validation_config.get("results_dir")
            if output_root:
                val_kwargs = self.config.actor_rollout_ref.rollout.val_kwargs
                generation_args = {
                    "max_tokens": int(os.getenv("VISHARNESS_VALIDATION_MAX_TOKENS", "2048")),
                    "temperature": float(val_kwargs.temperature),
                    "top_p": float(val_kwargs.top_p),
                    "presence_penalty": float(
                        os.getenv("VISHARNESS_VALIDATION_PRESENCE_PENALTY", "0.0")
                    ),
                    "extra_body": {
                        "top_k": int(val_kwargs.top_k),
                        "min_p": float(os.getenv("VISHARNESS_VALIDATION_MIN_P", "0.0")),
                        "repetition_penalty": float(
                            os.getenv("VISHARNESS_VALIDATION_REPETITION_PENALTY", "1.0")
                        ),
                    },
                    "n": int(val_kwargs.n),
                    "do_sample": bool(val_kwargs.do_sample),
                }
                artifact_dir = write_validation_artifacts(
                    output_root,
                    global_step=self.global_steps,
                    metrics=metrics,
                    records=records,
                    generation_args=generation_args,
                    metadata={
                        "checkpoint_path": self._checkpoint_dir_for_current_step(),
                        "archived_hf_checkpoint_path": os.path.join(
                            self._get_archived_checkpoint_dir(),
                            f"global_step_{self.global_steps}",
                        ),
                        "validation_data": str(self.config.data.val_files),
                        "experiment_name": str(self.config.trainer.experiment_name),
                    },
                )
                print(f"VisHarness validation details written to {artifact_dir}")
        return metrics

    def _add_actor_update_sample_metrics(self, metrics: dict, batch: DataProto) -> None:
        """Describe the exact non-padding per-turn samples sent to actor update."""

        response_lengths = batch.batch["response_mask"].sum(dim=-1).detach().cpu()
        valid_mask = response_lengths > 0
        response_lengths = response_lengths[valid_mask].to(torch.float32).tolist()
        prompt_width = batch.batch["prompts"].shape[-1]
        prompt_lengths = batch.batch["attention_mask"][:, :prompt_width].sum(dim=-1).detach().cpu()
        prompt_lengths = prompt_lengths[valid_mask].to(torch.float32).tolist()
        self._add_distribution_metrics(metrics, "train/update/assistant_response_length", response_lengths)
        self._add_distribution_metrics(metrics, "train/update/prompt_length", prompt_lengths)
        metrics["train/update/per_turn_samples"] = float(len(response_lengths))
        data_sources = batch.non_tensor_batch.get("data_source")
        turn_advantages = batch.non_tensor_batch.get("turn_advantage")
        effective_policy_advantages = batch.non_tensor_batch.get("effective_policy_advantage")
        turn_error_types = batch.non_tensor_batch.get("turn_error_type")
        raw_output_format_failures = batch.non_tensor_batch.get(
            "turn_raw_output_format_failure"
        )
        raw_tool_args_failures = batch.non_tensor_batch.get("turn_raw_tool_args_failure")
        valid_indices = valid_mask.nonzero(as_tuple=False).flatten().tolist()
        if turn_error_types is not None:
            valid_error_types = [str(turn_error_types[index]) for index in valid_indices]
            for metric_name, error_type in (
                ("output_format_error_rate", "output_format"),
                ("tool_args_error_rate", "tool_args"),
                ("truncation_error_rate", "truncation"),
            ):
                metrics[f"train/update/{metric_name}"] = (
                    sum(value == error_type for value in valid_error_types)
                    / max(len(valid_error_types), 1)
                )
            metrics["train/update/hard_error_turn_rate"] = (
                sum(
                    value in {"output_format", "tool_args", "truncation"}
                    for value in valid_error_types
                )
                / max(len(valid_error_types), 1)
            )
        if raw_output_format_failures is not None:
            metrics["train/update/raw_output_format_failure_rate"] = (
                sum(bool(raw_output_format_failures[index]) for index in valid_indices)
                / max(len(valid_indices), 1)
            )
        if raw_tool_args_failures is not None:
            metrics["train/update/raw_tool_args_failure_rate"] = (
                sum(bool(raw_tool_args_failures[index]) for index in valid_indices)
                / max(len(valid_indices), 1)
            )
        if data_sources is not None:
            source_counts = Counter(str(data_sources[index]) for index in valid_indices)
            for source, count in sorted(source_counts.items()):
                metric_source = self._metric_source_name(source)
                prefix = f"train/update/by_data_source/{metric_source}"
                metrics[f"{prefix}/turn_samples"] = float(count)
                metrics[f"{prefix}/turn_sample_share"] = count / max(len(valid_indices), 1)
                source_indices = [
                    index
                    for index in valid_indices
                    if str(data_sources[index]) == source
                ]
                if turn_error_types is not None:
                    source_error_types = [str(turn_error_types[index]) for index in source_indices]
                    for metric_name, error_type in (
                        ("output_format_error_rate", "output_format"),
                        ("tool_args_error_rate", "tool_args"),
                        ("truncation_error_rate", "truncation"),
                    ):
                        metrics[f"{prefix}/{metric_name}"] = (
                            sum(value == error_type for value in source_error_types)
                            / max(len(source_error_types), 1)
                        )
                    metrics[f"{prefix}/hard_error_turn_rate"] = (
                        sum(
                            value in {"output_format", "tool_args", "truncation"}
                            for value in source_error_types
                        )
                        / max(len(source_error_types), 1)
                    )
                if raw_output_format_failures is not None:
                    metrics[f"{prefix}/raw_output_format_failure_rate"] = (
                        sum(bool(raw_output_format_failures[index]) for index in source_indices)
                        / max(len(source_indices), 1)
                    )
                if raw_tool_args_failures is not None:
                    metrics[f"{prefix}/raw_tool_args_failure_rate"] = (
                        sum(bool(raw_tool_args_failures[index]) for index in source_indices)
                        / max(len(source_indices), 1)
                    )
                if turn_advantages is not None:
                    source_advantages = np.asarray(
                        [
                            float(turn_advantages[index])
                            for index in source_indices
                        ],
                        dtype=np.float64,
                    )
                    metrics[f"{prefix}/turn_advantage_abs_mean"] = (
                        float(np.abs(source_advantages).mean())
                        if source_advantages.size
                        else 0.0
                    )
                if effective_policy_advantages is not None:
                    source_effective_advantages = np.asarray(
                        [
                            float(effective_policy_advantages[index])
                            for index in source_indices
                        ],
                        dtype=np.float64,
                    )
                    metrics[f"{prefix}/effective_policy_advantage_abs_mean"] = (
                        float(np.abs(source_effective_advantages).mean())
                        if source_effective_advantages.size
                        else 0.0
                    )

    def _reference_kl_metrics_by_source(self, batch: DataProto) -> dict[str, float]:
        """Measure pre-update policy/reference KL on each task's real turns."""

        required_tensors = {"old_log_probs", "ref_log_prob", "response_mask"}
        if not required_tensors.issubset(batch.batch.keys()):
            return {}
        data_sources = batch.non_tensor_batch.get("data_source")
        if data_sources is None:
            return {}

        response_mask = batch.batch["response_mask"].to(torch.float32)
        kld = kl_penalty(
            logprob=batch.batch["old_log_probs"],
            ref_logprob=batch.batch["ref_log_prob"],
            kl_penalty=str(self.config.actor_rollout_ref.actor.kl_loss_type),
        ).to(torch.float32)
        loss_agg_mode = str(self.config.actor_rollout_ref.actor.loss_agg_mode)
        metrics: dict[str, float] = {}
        for source in sorted(set(str(value) for value in data_sources)):
            row_mask = torch.tensor(
                [str(value) == source for value in data_sources],
                dtype=torch.bool,
                device=response_mask.device,
            )
            row_mask &= response_mask.sum(dim=-1) > 0
            if not bool(row_mask.any()):
                continue
            source_kld = kld[row_mask]
            source_mask = response_mask[row_mask]
            if loss_agg_mode == "seq-mean-token-mean":
                per_sequence = (source_kld * source_mask).sum(dim=-1) / source_mask.sum(
                    dim=-1
                ).clamp_min(1.0)
                value = per_sequence.mean()
            elif loss_agg_mode == "token-mean":
                value = (source_kld * source_mask).sum() / source_mask.sum().clamp_min(1.0)
            else:
                raise ValueError(
                    f"Unsupported VisHarness loss aggregation mode {loss_agg_mode!r}"
                )
            metric_source = self._metric_source_name(source)
            metrics[
                f"train/update/by_data_source/{metric_source}/pre_update_reference_kl_mean"
            ] = float(value.detach().cpu())
        return metrics

    def _decode_train_generation_samples(self, batch: DataProto, num_samples: int) -> list[dict]:
        """Decode representative exact per-turn actor-update samples for tracking."""

        if num_samples <= 0 or "prompts" not in batch.batch or "responses" not in batch.batch:
            return []

        prompts = batch.batch["prompts"].detach().cpu()
        responses = batch.batch["responses"].detach().cpu()
        attention_mask = batch.batch["attention_mask"].detach().cpu().bool()
        prompt_width = prompts.shape[-1]
        response_width = responses.shape[-1]
        prompt_mask = attention_mask[:, :prompt_width]
        response_mask = batch.batch.get("response_mask")
        if response_mask is None:
            response_mask = attention_mask[:, -response_width:]
        else:
            response_mask = response_mask.detach().cpu().bool()

        turn_advantages = None
        if "token_level_scores" in batch.batch:
            turn_advantages = batch.batch["token_level_scores"].detach().cpu().sum(dim=-1)

        valid_indices = response_mask.sum(dim=-1).nonzero(as_tuple=False).flatten().tolist()
        limit = min(num_samples, len(valid_indices))
        if limit <= 0:
            return []

        selected_indices = []
        for candidate in valid_indices:
            if candidate not in selected_indices:
                selected_indices.append(candidate)
            if len(selected_indices) >= limit:
                break

        samples = []
        for idx in selected_indices:
            prompt_ids = prompts[idx][prompt_mask[idx]].tolist()
            response_ids = responses[idx][response_mask[idx]].tolist()
            prompt = self.tokenizer.decode(
                prompt_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            response = self.tokenizer.decode(
                response_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            samples.append(
                {
                    "uid": str(batch.non_tensor_batch.get("uid", [""] * len(batch))[idx]),
                    "trajectory_uid": str(batch.non_tensor_batch.get("trajectory_uid", [""] * len(batch))[idx]),
                    "turn_index": int(batch.non_tensor_batch.get("turn_index", [-1] * len(batch))[idx]),
                    "task_type": str(batch.non_tensor_batch.get("task_type", [""] * len(batch))[idx]),
                    "tool_name": str(batch.non_tensor_batch.get("turn_tool_name", [""] * len(batch))[idx]),
                    "trajectory_reward": float(
                        batch.non_tensor_batch.get("trajectory_reward", [float("nan")] * len(batch))[idx]
                    ),
                    "trajectory_advantage": float(
                        batch.non_tensor_batch.get("trajectory_advantage", [float("nan")] * len(batch))[idx]
                    ),
                    "trajectory_advantage_suppressed": bool(
                        batch.non_tensor_batch.get(
                            "trajectory_advantage_suppressed",
                            [False] * len(batch),
                        )[idx]
                    ),
                    "local_advantage_cost": float(
                        batch.non_tensor_batch.get("turn_local_advantage_cost", [float("nan")] * len(batch))[idx]
                    ),
                    "turn_advantage": (
                        float(turn_advantages[idx].item()) if turn_advantages is not None else float("nan")
                    ),
                    "ppo_turn_advantage": float(
                        batch.non_tensor_batch.get(
                            "ppo_turn_advantage",
                            [float("nan")] * len(batch),
                        )[idx]
                    ),
                    "trajectory_loss_weight": float(
                        batch.non_tensor_batch.get(
                            "trajectory_loss_weight",
                            [1.0] * len(batch),
                        )[idx]
                    ),
                    "effective_policy_advantage": float(
                        batch.non_tensor_batch.get(
                            "effective_policy_advantage",
                            [float("nan")] * len(batch),
                        )[idx]
                    ),
                    "effective_local_cost": float(
                        batch.non_tensor_batch.get(
                            "effective_local_cost",
                            [float("nan")] * len(batch),
                        )[idx]
                    ),
                    "local_floor_applied": bool(
                        batch.non_tensor_batch.get(
                            "local_floor_applied",
                            [False] * len(batch),
                        )[idx]
                    ),
                    "turn_error_type": str(
                        batch.non_tensor_batch.get("turn_error_type", [""] * len(batch))[idx]
                    ),
                    "prompt": prompt,
                    "response": response,
                }
            )
        return samples

    def _maybe_log_train_generations(self, batch: DataProto, logger) -> None:
        """Periodically upload decoded train prompt/response samples to cloud loggers."""

        log_config = self.config.get("visharness", {}).get("train_generations", {})
        log_freq = int(log_config.get("log_freq", 0))
        if log_freq <= 0 or self.global_steps % log_freq != 0:
            return

        num_samples = int(log_config.get("num_samples", 0))
        samples = self._decode_train_generation_samples(batch, num_samples=num_samples)
        if not samples:
            return

        backends = getattr(logger, "logger", {})
        rows = [
            [
                self.global_steps,
                sample["uid"],
                sample["trajectory_uid"],
                sample["turn_index"],
                sample["task_type"],
                sample["tool_name"],
                sample["trajectory_reward"],
                sample["trajectory_advantage"],
                sample["trajectory_advantage_suppressed"],
                sample["local_advantage_cost"],
                sample["turn_advantage"],
                sample["ppo_turn_advantage"],
                sample["trajectory_loss_weight"],
                sample["effective_policy_advantage"],
                sample["effective_local_cost"],
                sample["local_floor_applied"],
                sample["turn_error_type"],
                sample["prompt"],
                sample["response"],
            ]
            for sample in samples
        ]
        headers = [
            "step",
            "uid",
            "trajectory_uid",
            "turn_index",
            "task_type",
            "tool_name",
            "trajectory_reward",
            "trajectory_advantage",
            "trajectory_advantage_suppressed",
            "local_advantage_cost",
            "legacy_additive_turn_advantage",
            "ppo_turn_advantage_before_q",
            "trajectory_loss_weight_q",
            "effective_policy_advantage_after_q",
            "effective_local_cost_after_q",
            "local_floor_applied",
            "turn_error_type",
            "prompt",
            "response",
        ]

        if "swanlab" in backends:
            import swanlab

            table = swanlab.echarts.Table()
            table.add(headers=headers, rows=rows)
            swanlab.log({"train/generations": table}, step=self.global_steps)

        if "wandb" in backends:
            import wandb

            wandb.log({"train/generations": wandb.Table(columns=headers, data=rows)}, step=self.global_steps)

    def _filter_logged_metrics(self, metrics: dict) -> dict:
        """Keep a concise experiment dashboard while retaining full internal metrics."""

        log_config = self.config.get("visharness", {}).get("logged_metrics", {})
        mode = str(log_config.get("mode", "concise")).lower()
        drop_prefixes = tuple(log_config.get("drop_prefixes", ["critic/", "num_turns/"]))
        filtered = {key: value for key, value in metrics.items() if not key.startswith(drop_prefixes)}
        if mode == "all":
            return filtered
        if mode != "concise":
            raise ValueError(f"visharness.logged_metrics.mode must be 'concise' or 'all', got {mode!r}")

        include_exact = set(
            log_config.get(
                "include_exact",
                [
                    "actor/loss",
                    "actor/pg_loss",
                    "actor/lr",
                    "actor/grad_norm",
                    "actor/ppo_kl",
                    "actor/pg_clipfrac",
                    "actor/pg_clipfrac_lower",
                    "actor/entropy_loss",
                    "actor/kl_loss",
                    "actor/kl_coef",
                    "actor/perf/max_memory_allocated_gb",
                    "actor/perf/max_memory_reserved_gb",
                    "perf/time_per_step",
                    "perf/throughput",
                    "timing_s/gen",
                    "timing_s/update_actor",
                    "timing_s/update_weights",
                ],
            )
        )
        include_prefixes = tuple(
            log_config.get(
                "include_prefixes",
                [
                    "sampling/",
                    "rollout/eligible/",
                    "train/selected/",
                    "train/update/",
                    "filter/",
                    "streaming_rollout/",
                    "timing_s/agent_loop/",
                    "val-core/",
                    "val-aux/",
                ],
            )
        )
        return {
            key: value
            for key, value in filtered.items()
            if key in include_exact or key.startswith(include_prefixes)
        }

    @staticmethod
    def _drop_invalid_trajectories(
        batch: DataProto,
    ) -> tuple[DataProto | None, int, dict[str, int]]:
        """Drop all fatally aborted trajectories and count their invalid reasons."""

        flags = batch.non_tensor_batch.get("trajectory_invalid")
        if flags is None:
            # Backward compatibility for rollout outputs produced before the
            # unified trajectory-invalid fields were introduced.
            flags = batch.non_tensor_batch.get("rollout_prompt_overlong")
        if flags is None:
            return batch, 0, {}

        reasons = batch.non_tensor_batch.get("invalid_reason")
        legacy_prompt_overlong = batch.non_tensor_batch.get("rollout_prompt_overlong")

        drop_mask = np.array([bool(flag) for flag in flags], dtype=bool)
        dropped = int(drop_mask.sum())
        reason_counts = defaultdict(int)
        for idx, should_drop in enumerate(drop_mask):
            if not should_drop:
                continue
            reason = reasons[idx] if reasons is not None else None
            if reason is None and legacy_prompt_overlong is not None and bool(legacy_prompt_overlong[idx]):
                reason = "rollout_prompt_overlong"
            reason_counts[str(reason or "unknown")] += 1

        if dropped == 0:
            return batch, 0, dict(reason_counts)

        keep_idxs = np.nonzero(~drop_mask)[0].tolist()
        if not keep_idxs:
            return None, dropped, dict(reason_counts)
        return batch[keep_idxs], dropped, dict(reason_counts)

    @staticmethod
    def _drop_undersized_prompt_groups(
        batch: DataProto,
        *,
        min_trajectories: int,
    ) -> tuple[DataProto | None, int, int]:
        """Drop complete prompt groups with too few surviving trajectories."""

        if min_trajectories <= 0:
            raise ValueError(f"min_trajectories must be positive, got {min_trajectories}")
        if "uid" not in batch.non_tensor_batch:
            raise KeyError("Cannot filter undersized prompt groups because batch has no 'uid'")

        uid_keys = [VisHarnessTrainer._prompt_uid_key(uid) for uid in batch.non_tensor_batch["uid"]]
        trajectory_counts = defaultdict(int)
        for uid_key in uid_keys:
            trajectory_counts[uid_key] += 1

        undersized_uids = {
            uid_key for uid_key, count in trajectory_counts.items() if count < min_trajectories
        }
        if not undersized_uids:
            return batch, 0, 0

        keep_idxs = [idx for idx, uid_key in enumerate(uid_keys) if uid_key not in undersized_uids]
        dropped_trajectories = len(batch) - len(keep_idxs)
        dropped_prompt_groups = len(undersized_uids)
        if not keep_idxs:
            return None, dropped_trajectories, dropped_prompt_groups
        return batch[keep_idxs], dropped_trajectories, dropped_prompt_groups

    def _get_min_trajectories_after_rollout_filter(self) -> int:
        filter_config = self.config.get("visharness", {}).get("filter_groups", {})
        min_trajectories = int(filter_config.get("min_trajectories_after_rollout_filter", 4))
        if min_trajectories <= 0:
            raise ValueError(
                "visharness.filter_groups.min_trajectories_after_rollout_filter must be positive, "
                f"got {min_trajectories}"
            )
        return min_trajectories

    def _get_min_reward_std(self) -> float:
        """Return the minimum within-prompt sample std used by group filtering."""

        filter_config = self.config.get("visharness", {}).get("filter_groups", {})
        min_reward_std = float(filter_config.get("min_reward_std", 0.0))
        if not np.isfinite(min_reward_std) or min_reward_std < 0:
            raise ValueError(f"visharness.filter_groups.min_reward_std must be non-negative, got {min_reward_std}")
        return min_reward_std

    def _group_filter_enabled(self) -> bool:
        return bool(self.config.algorithm.filter_groups.enable)

    def _minimum_prompt_group_size_filter_enabled(self) -> bool:
        """Whether undersized surviving rollout groups must be discarded.

        Reward-variance filtering is optional, but task-balanced sampling still
        promises a final batch of complete-enough GRPO prompt groups. Therefore
        it keeps the minimum surviving-trajectory guard active independently of
        ``algorithm.filter_groups.enable``.
        """

        return self._group_filter_enabled() or self._task_sampling_enabled()

    def _maybe_drop_undersized_prompt_groups(
        self,
        batch: DataProto,
    ) -> tuple[DataProto | None, int, int]:
        """Apply the minimum group size required by filtering or task sampling."""

        if not self._minimum_prompt_group_size_filter_enabled():
            return batch, 0, 0
        return self._drop_undersized_prompt_groups(
            batch,
            min_trajectories=self._get_min_trajectories_after_rollout_filter(),
        )

    @staticmethod
    def _as_scalar_metric(metric_val) -> float:
        metric_arr = np.asarray(metric_val, dtype=np.float64)
        if metric_arr.size != 1:
            raise ValueError(f"filter_groups metric must be scalar per trajectory, got shape {metric_arr.shape}")
        return float(metric_arr.reshape(-1)[0])

    def _prepare_filter_group_metric(self, batch: DataProto, metric_name: str) -> None:
        if metric_name in batch.non_tensor_batch:
            return
        if metric_name == "seq_final_reward":
            batch.non_tensor_batch["seq_final_reward"] = (
                batch.batch["token_level_rewards"].sum(dim=-1).detach().cpu().numpy()
            )
        elif metric_name == "seq_reward":
            batch.non_tensor_batch["seq_reward"] = batch.batch["token_level_scores"].sum(dim=-1).detach().cpu().numpy()
        elif metric_name == "trajectory_reward":
            batch.non_tensor_batch["trajectory_reward"] = self._compute_trajectory_reward(batch)

    def _select_filter_group_uids(self, batch: DataProto, metric_name: str):
        """Keep prompt groups whose trajectory rewards have sufficient variation.

        The selector uses only the within-prompt sample standard deviation. It
        intentionally ignores reward range and local turn errors: a prompt with
        no trajectory-level reward contrast cannot provide a task-reward GRPO
        direction and is filtered even if one of its turns has a local error.
        """

        if metric_name not in batch.non_tensor_batch:
            raise KeyError(f"filter_groups metric {metric_name!r} was not found in non_tensor_batch")

        min_reward_std = self._get_min_reward_std()
        prompt_uid2metric_vals = defaultdict(list)
        for uid, metric_val in zip(batch.non_tensor_batch["uid"], batch.non_tensor_batch[metric_name], strict=True):
            prompt_uid2metric_vals[uid].append(self._as_scalar_metric(metric_val))

        kept_prompt_uids = []
        for uid, metric_vals in prompt_uid2metric_vals.items():
            metric_vals = np.asarray(metric_vals, dtype=np.float64)
            reward_std = float(np.std(metric_vals, ddof=1)) if metric_vals.size > 1 else 0.0

            # Preserve upstream DAPO semantics at a zero threshold: constant
            # groups must still be filtered instead of passing via std >= 0.
            keep_for_std = reward_std > 0 if min_reward_std == 0 else reward_std >= min_reward_std
            if keep_for_std:
                kept_prompt_uids.append(uid)

        # Keep this compatibility field because the per-turn batch builder and
        # historical metrics understand it. The std-only filter no longer has
        # a production path that suppresses task advantages for selected groups.
        batch.non_tensor_batch["trajectory_advantage_suppressed"] = np.zeros(
            len(batch),
            dtype=bool,
        )

        return kept_prompt_uids, {}

    def _apply_filter_groups(
        self,
        *,
        new_batch: DataProto,
        batch: DataProto | None,
        metrics: dict,
        num_prompt_in_batch: int,
    ) -> tuple[DataProto, int, dict[str, int]]:
        metric_name = str(self.config.algorithm.filter_groups.metric or "")
        if metric_name != "trajectory_reward":
            raise ValueError(
                "VisHarness GRPO group filtering must use "
                "algorithm.filter_groups.metric=trajectory_reward; "
                f"got {metric_name!r}"
            )
        self._prepare_filter_group_metric(new_batch, metric_name)
        kept_prompt_uids, filter_group_metrics = self._select_filter_group_uids(new_batch, metric_name)
        metrics.update(filter_group_metrics)
        num_prompt_in_batch += len(kept_prompt_uids)

        kept_traj_idxs = []
        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
            if traj_from_prompt_uid in kept_prompt_uids:
                kept_traj_idxs.append(idx)

        new_batch = new_batch[kept_traj_idxs]
        kept_counts_by_source = (
            self._prompt_group_counts_by_source(new_batch)
            if len(new_batch) > 0
            else {}
        )
        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])
        return batch, num_prompt_in_batch, kept_counts_by_source

    @staticmethod
    def _prompt_uid_key(uid):
        if hasattr(uid, "item"):
            try:
                uid = uid.item()
            except ValueError:
                pass
        try:
            hash(uid)
        except TypeError:
            return repr(uid)
        return uid

    def _truncate_to_prompt_groups(self, batch: DataProto, prompt_bsz: int) -> tuple[DataProto, int]:
        """Keep the first prompt_bsz prompt groups without splitting a uid group."""
        if "uid" not in batch.non_tensor_batch:
            raise KeyError("Cannot truncate by prompt group because batch.non_tensor_batch has no 'uid'")

        selected_uids = []
        selected_uid_set = set()
        for uid in batch.non_tensor_batch["uid"]:
            uid_key = self._prompt_uid_key(uid)
            if uid_key in selected_uid_set:
                continue
            selected_uids.append(uid_key)
            selected_uid_set.add(uid_key)
            if len(selected_uids) >= prompt_bsz:
                break

        keep_indices = [
            idx
            for idx, uid in enumerate(batch.non_tensor_batch["uid"])
            if self._prompt_uid_key(uid) in selected_uid_set
        ]
        return batch[keep_indices], len(selected_uids)

    def _prepare_raw_prompt_batch(self, batch_dict: dict) -> DataProto:
        """Create a prompt-level DataProto with stable uids before any top-up slicing."""

        batch = DataProto.from_single_dict(batch_dict)
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        if "uid" not in batch.non_tensor_batch:
            batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)
        return batch

    def _take_cycling_raw_prompts_by_source(
        self,
        requested_counts: dict[str, int],
    ) -> tuple[DataProto, dict[str, int], dict[str, int]]:
        """Fetch an exact quota directly from independent per-source pools."""

        draws, completed_cycles = self._task_source_pool.take(requested_counts)
        if len(draws) != sum(int(count) for count in requested_counts.values()):
            raise RuntimeError(
                "Cycling task-source pool returned a partial batch: "
                f"requested={requested_counts}, returned={len(draws)}"
            )
        samples = [self.train_dataset[draw.dataset_index] for draw in draws]
        batch_dict = self._visharness_collate_fn(samples)
        batch = self._prepare_raw_prompt_batch(batch_dict)
        if len(batch) != len(draws):
            raise RuntimeError(
                "Task prompt collation changed the batch length: "
                f"draws={len(draws)}, collated={len(batch)}"
            )
        data_sources = batch.non_tensor_batch.get("data_source")
        if data_sources is None:
            raise KeyError("Task-balanced sampling requires non_tensor_batch['data_source']")
        actual_sources = [str(source) for source in data_sources]
        expected_sources = [draw.source for draw in draws]
        if actual_sources != expected_sources:
            raise RuntimeError(
                "Dataset rows do not match the task-source sampling plan: "
                f"expected={expected_sources}, actual={actual_sources}"
            )

        original_uids = batch.non_tensor_batch.get("uid")
        if original_uids is not None:
            batch.non_tensor_batch["dataset_uid"] = np.asarray(
                list(original_uids),
                dtype=object,
            )
        batch.non_tensor_batch["uid"] = np.asarray(
            [draw.attempt_uid for draw in draws],
            dtype=object,
        )
        batch.non_tensor_batch["task_dataset_index"] = np.asarray(
            [draw.dataset_index for draw in draws],
            dtype=np.int64,
        )
        batch.non_tensor_batch["task_source_cycle"] = np.asarray(
            [draw.source_cycle for draw in draws],
            dtype=np.int64,
        )
        selected_counts = {
            source: int(count)
            for source, count in Counter(expected_sources).items()
        }
        expected_counts = {
            str(source): int(count)
            for source, count in requested_counts.items()
            if int(count) > 0
        }
        if selected_counts != expected_counts:
            raise RuntimeError(
                "Task prompt selection did not preserve the requested quota: "
                f"requested={expected_counts}, selected={selected_counts}"
            )
        return batch, selected_counts, completed_cycles

    @staticmethod
    def _take_prompt_prefix(batch: DataProto, take_count: int) -> tuple[DataProto, DataProto | None]:
        """Take the first take_count prompts and return the unused suffix as a buffer."""

        if take_count <= 0:
            raise ValueError(f"take_count must be positive, got {take_count}")
        if len(batch) <= take_count:
            return batch, None
        return batch[:take_count], batch[take_count:]

    def _take_raw_prompts(self, target_prompt_count: int, next_raw_prompt_batch) -> DataProto | None:
        """Take prompts from the persisted suffix before advancing the dataloader."""

        if target_prompt_count <= 0:
            raise ValueError(f"target_prompt_count must be positive, got {target_prompt_count}")

        prompt_chunks = []
        remaining = int(target_prompt_count)
        while remaining > 0:
            raw_prompt_buffer = getattr(self, "_visharness_raw_prompt_buffer", None)
            if raw_prompt_buffer is None:
                raw_prompt_buffer = next_raw_prompt_batch()
                if raw_prompt_buffer is None:
                    break
                if not isinstance(raw_prompt_buffer, DataProto):
                    raise TypeError(
                        "next_raw_prompt_batch must return DataProto or None, "
                        f"got {type(raw_prompt_buffer)}"
                    )
                if len(raw_prompt_buffer) == 0:
                    raise ValueError("next_raw_prompt_batch returned an empty DataProto")
                self._visharness_raw_prompt_buffer = raw_prompt_buffer

            take_count = min(remaining, len(raw_prompt_buffer))
            prompt_chunk, raw_prompt_buffer = self._take_prompt_prefix(raw_prompt_buffer, take_count)
            self._visharness_raw_prompt_buffer = raw_prompt_buffer
            prompt_chunks.append(prompt_chunk)
            remaining -= len(prompt_chunk)

        if not prompt_chunks:
            return None
        if len(prompt_chunks) == 1:
            return prompt_chunks[0]
        return DataProto.concat(prompt_chunks)

    @classmethod
    def _prompt_group_counts_by_source(cls, batch: DataProto) -> dict[str, int]:
        if "uid" not in batch.non_tensor_batch:
            raise KeyError("Cannot count prompt groups by source because batch has no 'uid'")
        data_sources = batch.non_tensor_batch.get("data_source")
        if data_sources is None:
            raise KeyError("Cannot count prompt groups by source because batch has no 'data_source'")

        source_by_uid: dict[Any, str] = {}
        for uid, source_value in zip(batch.non_tensor_batch["uid"], data_sources, strict=True):
            uid_key = cls._prompt_uid_key(uid)
            source = str(source_value)
            previous_source = source_by_uid.setdefault(uid_key, source)
            if previous_source != source:
                raise ValueError(
                    f"Prompt group {uid_key!r} contains multiple data sources: "
                    f"{previous_source!r} and {source!r}"
                )
        return {
            source: int(count)
            for source, count in Counter(source_by_uid.values()).items()
        }

    @staticmethod
    def _raw_prompt_buffer_size(raw_prompt_buffers: dict[str, DataProto] | None) -> int:
        if not raw_prompt_buffers:
            return 0
        return sum(len(buffer) for buffer in raw_prompt_buffers.values())

    def _has_buffered_raw_prompts(self) -> bool:
        raw_prompt_buffer = getattr(self, "_visharness_raw_prompt_buffer", None)
        return bool(raw_prompt_buffer is not None and len(raw_prompt_buffer) > 0)

    @staticmethod
    def _raw_prompt_buffer_from_resume_state(resume_state: dict) -> DataProto | None:
        """Validate and return an unconsumed raw-prompt suffix from a checkpoint."""

        raw_prompt_buffer = resume_state.get("raw_prompt_buffer")
        if raw_prompt_buffer is None:
            return None
        if not isinstance(raw_prompt_buffer, DataProto):
            raise ValueError(
                "Invalid VisHarness resume state: raw_prompt_buffer must be a DataProto or None, "
                f"got {type(raw_prompt_buffer)}"
            )
        raw_prompt_buffer.check_consistency()
        if len(raw_prompt_buffer) == 0:
            return None
        return raw_prompt_buffer

    def _task_sampling_resume_signature(self) -> dict[str, Any]:
        if not self._task_sampling_enabled():
            return {"enabled": False}
        plan = self._task_sampling_plan
        sampling_config = self._task_sampling_config()
        return {
            "enabled": True,
            "alpha": float(plan.alpha),
            "batch_size": int(plan.batch_size),
            "epoch_semantics": "all_sources_first_pass_with_source_cycling",
            "source_counts": dict(plan.source_counts),
            "seed": int(self.config.data.get("seed", 0)),
            "max_no_progress_cycles": int(
                sampling_config.get("max_no_progress_cycles", 3)
            ),
        }

    @staticmethod
    def _semantic_sha256(value: Any) -> str:
        """Hash nested checkpoint/dataset values without relying on pickle bytes."""

        digest = hashlib.sha256()

        def emit(tag: bytes, payload: bytes = b"") -> None:
            digest.update(tag)
            digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
            digest.update(payload)

        def visit(item: Any) -> None:
            if isinstance(item, np.generic):
                visit(item.item())
            elif item is None:
                emit(b"none")
            elif isinstance(item, bool):
                emit(b"bool", b"1" if item else b"0")
            elif isinstance(item, int):
                emit(b"int", str(item).encode("ascii"))
            elif isinstance(item, float):
                emit(b"float", item.hex().encode("ascii"))
            elif isinstance(item, str):
                emit(b"str", item.encode("utf-8"))
            elif isinstance(item, os.PathLike):
                emit(b"path", os.fspath(item).encode("utf-8"))
            elif isinstance(item, bytes | bytearray | memoryview):
                emit(b"bytes", bytes(item))
            elif isinstance(item, torch.Tensor):
                tensor = item.detach().cpu().contiguous()
                emit(b"torch-dtype", str(tensor.dtype).encode("ascii"))
                visit(tuple(int(size) for size in tensor.shape))
                raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
                emit(b"torch-data", raw)
            elif isinstance(item, np.ndarray):
                emit(b"numpy-dtype", str(item.dtype).encode("ascii"))
                visit(tuple(int(size) for size in item.shape))
                if item.dtype.hasobject:
                    for nested in item.reshape(-1).tolist():
                        visit(nested)
                else:
                    emit(b"numpy-data", np.ascontiguousarray(item).tobytes())
            elif isinstance(item, Mapping):
                emit(b"mapping-start", str(len(item)).encode("ascii"))
                ordered_items = sorted(
                    item.items(),
                    key=lambda pair: VisHarnessTrainer._semantic_sha256(pair[0]),
                )
                for key, nested in ordered_items:
                    visit(key)
                    visit(nested)
                emit(b"mapping-end")
            elif isinstance(item, Sequence):
                emit(b"sequence-start", str(len(item)).encode("ascii"))
                for nested in item:
                    visit(nested)
                emit(b"sequence-end")
            elif isinstance(item, set | frozenset):
                emit(b"set-start", str(len(item)).encode("ascii"))
                for nested_digest in sorted(
                    VisHarnessTrainer._semantic_sha256(nested) for nested in item
                ):
                    emit(b"set-item", nested_digest.encode("ascii"))
                emit(b"set-end")
            else:
                raise TypeError(
                    "Cannot build a stable resume fingerprint for value of type "
                    f"{type(item).__module__}.{type(item).__qualname__}"
                )

        visit(value)
        return digest.hexdigest()

    def _train_dataset_resume_signature(self) -> dict[str, Any] | None:
        """Fingerprint the ordered, post-filter training dataset used by the sampler."""

        cached = getattr(self, "_visharness_train_dataset_signature", None)
        if cached is not None:
            return dict(cached)

        train_dataset = getattr(self, "train_dataset", None)
        dataframe = getattr(train_dataset, "dataframe", None)
        if dataframe is None:
            return None

        row_count = int(len(dataframe))
        columns = [str(column) for column in getattr(dataframe, "column_names", [])]
        content_digest = hashlib.sha256()
        for row_index in range(row_count):
            row = dataframe[row_index]
            row_digest = self._semantic_sha256(row)
            content_digest.update(bytes.fromhex(row_digest))

        source_counts: dict[str, int] = {}
        if "data_source" in columns:
            source_counts = {
                str(source): int(count)
                for source, count in sorted(Counter(str(value) for value in dataframe["data_source"]).items())
            }
        signature = {
            "version": 1,
            "row_count": row_count,
            "columns": columns,
            "source_counts": source_counts,
            "ordered_content_sha256": content_digest.hexdigest(),
        }
        self._visharness_train_dataset_signature = dict(signature)
        return signature

    def _validate_train_dataset_resume_state(self, resume_state: dict) -> bool:
        """Reject a checkpoint whose sampler indices refer to different rows."""

        if not resume_state:
            return True
        saved_signature = resume_state.get("train_dataset_signature")
        if saved_signature is None:
            print(
                "Warning: this legacy VisHarness checkpoint predates strict training-data "
                "fingerprints. Source counts and sampling settings were checked, but exact "
                "row contents/order cannot be proven. Newly saved checkpoints will use "
                "strict dataset validation.",
                flush=True,
            )
            return False
        current_signature = self._train_dataset_resume_signature()
        if current_signature is None:
            raise RuntimeError(
                "Cannot validate the checkpoint training-data fingerprint because the current "
                "train dataset has no dataframe."
            )
        if saved_signature != current_signature:
            raise ValueError(
                "Training data do not match the checkpoint. The ordered, post-filter dataset "
                "has changed, so restoring its saved sampler position would skip or repeat "
                "prompts. Start a new run or restore the original training data. "
                f"saved={saved_signature}, current={current_signature}"
            )
        return True

    def _dataloader_state_signature(self, dataloader_state: dict | None = None) -> str | None:
        if dataloader_state is None:
            train_dataloader = getattr(self, "train_dataloader", None)
            if train_dataloader is None:
                return None
            dataloader_state = train_dataloader.state_dict()
        # Runtime consistency depends on the delivered sampler position and
        # terminal bit. Worker prefetch snapshots may change internally while
        # model shards are being saved even though no prompt was delivered, so
        # hashing the entire torchdata implementation detail would create false
        # resume mismatches.
        samples_yielded = self._find_samples_yielded_in_state(dataloader_state)
        logical_state: dict[str, Any] = {
            "samples_yielded": samples_yielded,
            "iterator_finished": bool(dataloader_state.get("_iterator_finished", False)),
        }
        if samples_yielded is None:
            logical_state["fallback_full_state_sha256"] = self._semantic_sha256(
                dataloader_state
            )
        return self._semantic_sha256(logical_state)

    def _validate_dataloader_resume_state(self, resume_state: dict) -> bool:
        if not resume_state or not self._resumed_from_checkpoint():
            return True
        if self._task_sampling_enabled():
            # The task source pool, not StatefulDataLoader, is authoritative in
            # coverage-based task-balanced training.
            return True
        checkpoint_dir = self._resume_checkpoint_dir_for_current_step()
        dataloader_path = os.path.join(checkpoint_dir, "data.pt")
        if not os.path.exists(dataloader_path):
            raise FileNotFoundError(
                "The selected checkpoint is missing its dataloader state: "
                f"{dataloader_path}. Exact resume is not possible."
            )
        saved_signature = resume_state.get("dataloader_state_signature")
        if saved_signature is None:
            return False
        try:
            dataloader_state = torch.load(
                dataloader_path,
                map_location="cpu",
                weights_only=False,
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to load dataloader state from {dataloader_path}; exact resume is not possible."
            ) from error
        current_signature = self._dataloader_state_signature(dataloader_state)
        if str(saved_signature) != str(current_signature):
            raise ValueError(
                "VisHarness resume state and data.pt are from different runtime snapshots. "
                f"checkpoint={checkpoint_dir}"
            )
        return True

    def _validate_task_sampling_resume_state(self, resume_state: dict) -> None:
        if not resume_state:
            return
        saved_signature = resume_state.get("task_sampling")
        if saved_signature is None:
            # Checkpoints written before task-balanced sampling did not contain
            # a signature. Their original sampler state may be incompatible,
            # so only permit the unchanged legacy (disabled) path.
            if self._task_sampling_enabled():
                raise ValueError(
                    "Cannot enable task-balanced sampling while resuming a "
                    "checkpoint that predates its sampling-state signature. "
                    "Start a new run, or resume with "
                    "visharness.task_sampling.enable=false."
                )
            return
        if self._task_sampling_enabled() and int(resume_state.get("version", 1)) < 5:
            raise ValueError(
                "This checkpoint predates the independent cycling task-source pool. "
                "Its shared DataLoader/buffer state cannot be converted into exact "
                "per-source permutations and coverage. Start a new run, or use the "
                "checkpoint weights as an explicit warm start with reset optimizer "
                "and sampling state."
            )
        current_signature = self._task_sampling_resume_signature()
        if saved_signature != current_signature:
            raise ValueError(
                "Task-sampling configuration does not match the checkpoint. "
                f"saved={saved_signature}, current={current_signature}. "
                "Resume with the original sampling settings or start a new run."
            )

    @staticmethod
    def _effective_restored_prompt_examples(
        dataloader_samples_yielded: int | None,
        raw_prompt_buffer: DataProto | dict[str, DataProto] | None,
    ) -> int | None:
        """Exclude dataloader-prefetched prompts that have not entered rollout yet."""

        if dataloader_samples_yielded is None:
            return None
        if isinstance(raw_prompt_buffer, dict):
            buffered_prompt_count = VisHarnessTrainer._raw_prompt_buffer_size(raw_prompt_buffer)
        else:
            buffered_prompt_count = len(raw_prompt_buffer) if raw_prompt_buffer is not None else 0
        return max(int(dataloader_samples_yielded) - int(buffered_prompt_count), 0)

    @staticmethod
    def _find_samples_yielded_in_state(state) -> int | None:
        """Best-effort extraction of torchdata StatefulDataLoader sampler progress."""

        if isinstance(state, dict):
            value = state.get("samples_yielded")
            if isinstance(value, (int, np.integer)):
                return int(value)
            for nested in state.values():
                found = VisHarnessTrainer._find_samples_yielded_in_state(nested)
                if found is not None:
                    return found
        elif isinstance(state, (list, tuple)):
            for nested in state:
                found = VisHarnessTrainer._find_samples_yielded_in_state(nested)
                if found is not None:
                    return found
        return None

    def _train_dataloader_samples_yielded(self) -> int | None:
        """Return restored dataloader sample progress when available.

        The VisHarness top-up loop can consume more raw prompts than
        ``global_steps * train_batch_size``. StatefulDataLoader persists that
        true sampler progress in checkpoints, so use it for progress display
        after resume. This is intentionally best-effort because torchdata's
        internal state-dict layout is not part of VisHarness' public API.
        """

        try:
            state = self.train_dataloader.state_dict()
        except Exception:
            return None
        samples_yielded = self._find_samples_yielded_in_state(state)
        if samples_yielded is None or samples_yielded < 0:
            return None
        return int(samples_yielded)

    def _checkpoint_dir_for_current_step(self) -> str:
        checkpoint_root = os.path.abspath(os.path.expanduser(str(self.config.trainer.default_local_dir)))
        return os.path.join(checkpoint_root, f"global_step_{self.global_steps}")

    def _resume_checkpoint_dir_for_current_step(self) -> str:
        if self.config.trainer.resume_mode == "resume_path":
            resume_from_path = self.config.trainer.get("resume_from_path", None)
            if resume_from_path:
                return os.path.abspath(os.path.expanduser(str(resume_from_path)))
        return self._checkpoint_dir_for_current_step()

    def _visharness_resume_state_path(self, checkpoint_dir: str | None = None) -> str:
        if checkpoint_dir is None:
            checkpoint_dir = self._checkpoint_dir_for_current_step()
        return os.path.join(checkpoint_dir, "visharness_resume_state.pt")

    def _resumed_from_checkpoint(self) -> bool:
        """Distinguish a real global_step_0 resume from a fresh run."""

        loaded = getattr(self, "_visharness_checkpoint_loaded", None)
        if loaded is not None:
            return bool(loaded)
        # Keep helper-level tests and older direct callers backward compatible.
        return int(getattr(self, "global_steps", 0)) > 0

    def _load_visharness_resume_state(self) -> dict:
        if not self._resumed_from_checkpoint():
            return {}
        state_path = self._visharness_resume_state_path(self._resume_checkpoint_dir_for_current_step())
        if not os.path.exists(state_path):
            raise FileNotFoundError(
                "The selected checkpoint is missing its VisHarness resume state: "
                f"{state_path}. The checkpoint may be incomplete and cannot be resumed safely."
            )
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=False)
        except Exception as error:
            raise RuntimeError(
                f"Failed to load VisHarness resume state from {state_path}; "
                "the checkpoint cannot be resumed safely."
            ) from error
        if not isinstance(state, dict):
            raise TypeError(
                f"Invalid VisHarness resume state at {state_path}: expected dict, got {type(state)}"
            )
        state_version = int(state.get("version", 1))
        if state_version > 5:
            raise ValueError(
                f"VisHarness resume state at {state_path} uses unsupported version "
                f"{state_version}; this code supports up to version 5."
            )
        saved_global_steps = state.get("global_steps", None)
        if saved_global_steps is None:
            raise ValueError(f"VisHarness resume state at {state_path} has no global_steps field")
        if int(saved_global_steps) != int(self.global_steps):
            raise ValueError(
                "VisHarness resume state global_steps mismatch: "
                f"state={saved_global_steps}, checkpoint={self.global_steps}."
            )
        return state

    @staticmethod
    def _atomic_torch_save(value: Any, path: str) -> None:
        """Durably replace one torch-serialized file without exposing a partial file."""

        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        temporary_path = f"{path}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        try:
            with open(temporary_path, "wb") as file:
                torch.save(value, file)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    def _save_visharness_resume_state(
        self,
        *,
        dataloader_state: dict | None = None,
    ) -> None:
        checkpoint_dir = self._checkpoint_dir_for_current_step()
        os.makedirs(checkpoint_dir, exist_ok=True)
        train_epoch = getattr(self, "_visharness_train_epoch", None)
        if train_epoch is None:
            train_epoch = int(self.global_steps) // max(int(len(self.train_dataloader)), 1)
        task_source_pool_state = None
        raw_prompt_buffer = None
        if self._task_sampling_enabled():
            task_source_pool_state = self._task_source_pool.state_dict()
        else:
            raw_prompt_buffer = getattr(self, "_visharness_raw_prompt_buffer", None)
            if raw_prompt_buffer is not None:
                if not isinstance(raw_prompt_buffer, DataProto):
                    raise TypeError(
                        "_visharness_raw_prompt_buffer must be a DataProto or None, "
                        f"got {type(raw_prompt_buffer)}"
                    )
                raw_prompt_buffer.check_consistency()
                if len(raw_prompt_buffer) == 0:
                    raw_prompt_buffer = None
        state = {
            "version": 5,
            "global_steps": int(self.global_steps),
            "train_epoch": int(train_epoch),
            "epoch_start_global_step": int(
                getattr(self, "_visharness_epoch_start_global_step", 0)
            ),
            "task_sampling": self._task_sampling_resume_signature(),
            "task_source_pool_state": task_source_pool_state,
            "train_dataset_signature": self._train_dataset_resume_signature(),
            "dataloader_state_signature": (
                None
                if self._task_sampling_enabled()
                else self._dataloader_state_signature(dataloader_state)
            ),
            "training_completed": bool(
                getattr(self, "_visharness_training_completed", False)
            ),
            "completion_reason": getattr(
                self,
                "_visharness_completion_reason",
                None,
            ),
            "last_validation_step": getattr(
                self,
                "_visharness_last_validation_step",
                None,
            ),
            "raw_prompt_buffer": raw_prompt_buffer,
        }
        self._atomic_torch_save(
            state,
            self._visharness_resume_state_path(checkpoint_dir),
        )

    def _refresh_checkpoint_runtime_state(self) -> None:
        """Refresh data.pt and VisHarness state without re-saving model shards."""

        checkpoint_dir = self._checkpoint_dir_for_current_step()
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(
                f"Cannot refresh runtime state because checkpoint directory does not exist: {checkpoint_dir}"
            )
        dataloader_state = self.train_dataloader.state_dict()
        self._atomic_torch_save(
            dataloader_state,
            os.path.join(checkpoint_dir, "data.pt"),
        )
        self._save_visharness_resume_state(dataloader_state=dataloader_state)

    @staticmethod
    def _resume_state_buffer_size(resume_state: dict) -> int:
        raw_prompt_buffers = resume_state.get("raw_prompt_buffers")
        if isinstance(raw_prompt_buffers, dict):
            return sum(
                len(buffer)
                for buffer in raw_prompt_buffers.values()
                if isinstance(buffer, DataProto)
            )
        raw_prompt_buffer = resume_state.get("raw_prompt_buffer")
        return len(raw_prompt_buffer) if isinstance(raw_prompt_buffer, DataProto) else 0

    def _resume_training_completion(self, resume_state: dict) -> tuple[bool, str | None]:
        """Return explicit completion state, with a safe legacy inference."""

        if not resume_state:
            return False, None
        if "training_completed" in resume_state:
            completed = bool(resume_state["training_completed"])
            reason = resume_state.get("completion_reason") if completed else None
            return completed, str(reason) if reason is not None else None

        # Version <=3 checkpoints did not store an explicit completion bit.
        # Infer completion only from terminal states that cannot contain another
        # complete update; source buffers take precedence over exhausted epochs.
        if int(self.global_steps) >= int(self.total_training_steps):
            return True, "legacy_static_step_limit"
        train_epoch = resume_state.get("train_epoch")
        if (
            train_epoch is not None
            and int(train_epoch) >= int(self.config.trainer.total_epochs)
            and self._resume_state_buffer_size(resume_state) == 0
        ):
            return True, "legacy_data_exhausted"
        return False, None

    def _validation_artifact_matches_checkpoint(self, global_step: int) -> bool:
        """Recognize a complete saved validation for legacy resume states."""

        validation_config = self.config.get("visharness", {}).get("validation", {})
        output_root = validation_config.get("results_dir")
        if not output_root:
            return False
        metrics_path = os.path.join(
            os.path.abspath(os.path.expanduser(str(output_root))),
            f"global_step_{int(global_step)}",
            "metrics.json",
        )
        if not os.path.isfile(metrics_path):
            return False
        try:
            with open(metrics_path, encoding="utf-8") as file:
                summary = json.load(file)
        except (OSError, ValueError, TypeError):
            return False
        if int(summary.get("global_step", -1)) != int(global_step):
            return False
        val_kwargs = self.config.actor_rollout_ref.rollout.get("val_kwargs", {})
        expected_samples = len(self.val_dataset) * int(val_kwargs.get("n", 1))
        if int(summary.get("sample_count", -1)) != int(expected_samples):
            return False
        metadata = summary.get("metadata") or {}
        saved_checkpoint = metadata.get("checkpoint_path")
        if not saved_checkpoint:
            return False
        saved_validation_data = metadata.get("validation_data")
        if saved_validation_data is not None and str(saved_validation_data) != str(
            self.config.data.val_files
        ):
            return False
        expected_checkpoint = self._resume_checkpoint_dir_for_current_step()
        return os.path.abspath(os.path.expanduser(str(saved_checkpoint))) == os.path.abspath(
            os.path.expanduser(str(expected_checkpoint))
        )

    def _restored_last_validation_step(self, resume_state: dict) -> int | None:
        if resume_state and "last_validation_step" in resume_state:
            saved_step = resume_state.get("last_validation_step")
            if saved_step is None:
                return None
            saved_step = int(saved_step)
            if saved_step > int(self.global_steps):
                raise ValueError(
                    "VisHarness resume state has a validation step newer than its checkpoint: "
                    f"validation={saved_step}, checkpoint={self.global_steps}."
                )
            return saved_step
        if self.global_steps > 0 and self._validation_artifact_matches_checkpoint(self.global_steps):
            print(
                f"VisHarness resume: found complete validation artifacts for global_step_{self.global_steps}.",
                flush=True,
            )
            return int(self.global_steps)
        return None

    def _maybe_restore_dataloader_state_after_base_load(self) -> None:
        """Restore dataloader state using the saved iterator state, not static step math.

        verl's base loader skips dataloader-state restore whenever
        ``global_steps % len(train_dataloader) == 0``. That heuristic is valid
        for the vanilla one-dataloader-batch-per-update loop, but VisHarness can
        consume extra prompt batches during top-up sampling. In that case,
        global_steps is not a reliable epoch-boundary signal. We therefore
        re-check the actual saved StatefulDataLoader state and restore it unless
        that state explicitly says the iterator had finished.
        """

        if (
            self.config.trainer.resume_mode == "disable"
            or not self._resumed_from_checkpoint()
            or self._task_sampling_enabled()
        ):
            return
        dataloader_path = os.path.join(self._resume_checkpoint_dir_for_current_step(), "data.pt")
        if not os.path.exists(dataloader_path):
            return
        try:
            dataloader_state = torch.load(dataloader_path, map_location="cpu", weights_only=False)
        except Exception as error:
            print(f"Warning: failed to inspect dataloader state at {dataloader_path}: {error}")
            return
        if not isinstance(dataloader_state, dict):
            return
        if bool(dataloader_state.get("_iterator_finished", False)):
            return
        target_samples_yielded = self._find_samples_yielded_in_state(dataloader_state)
        current_samples_yielded = self._train_dataloader_samples_yielded()
        if target_samples_yielded is not None and current_samples_yielded == target_samples_yielded:
            return
        self.train_dataloader.load_state_dict(dataloader_state)

    def _infer_current_train_epoch(
        self,
        resume_state: dict,
        restored_prompt_examples_consumed: int | None,
        prompt_bsz_for_progress: int,
    ) -> int:
        total_epochs = max(int(self.config.trainer.total_epochs), 1)
        # ``total_epochs`` is a valid saved state: it means the dataloader
        # stream is exhausted. Source-aware top-up buffers may still need to be
        # consumed without reopening an already-finished epoch.
        max_epoch_index = total_epochs

        state_epoch = resume_state.get("train_epoch", None)
        if state_epoch is not None:
            train_epoch = min(max(int(state_epoch), 0), max_epoch_index)
            if int(self.global_steps) > 0:
                print(f"VisHarness resume: using saved train_epoch={train_epoch}.")
            return train_epoch

        static_epoch = int(self.global_steps) // max(int(len(self.train_dataloader)), 1)
        train_epoch = static_epoch
        if restored_prompt_examples_consumed is not None:
            samples_per_epoch = int(len(self.train_dataloader)) * max(int(prompt_bsz_for_progress), 1)
            if samples_per_epoch > 0:
                sample_epoch = int(restored_prompt_examples_consumed) // samples_per_epoch
                train_epoch = max(train_epoch, sample_epoch)
        train_epoch = min(max(int(train_epoch), 0), max_epoch_index)
        if int(self.global_steps) > 0:
            print(
                "VisHarness resume: no saved train_epoch found; inferred train_epoch="
                f"{train_epoch} from dataloader/global-step state."
            )
        return train_epoch

    def compute_kl_related_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict):
        """Delay old/reference logprob computation until trajectories are split."""

        return batch

    def _collect_streaming_filtered_rollouts(
        self,
        *,
        task_sampling_state: dict | None,
        next_raw_prompt_batch,
        metrics: dict,
        timing_raw: dict,
        raw_trajectory_metrics: dict,
    ) -> dict[str, Any]:
        """Fill one actor batch with continuously refilled prompt-group jobs."""

        if not self._group_filter_enabled():
            raise RuntimeError(
                "_collect_streaming_filtered_rollouts requires filter_groups.enable=true"
            )
        if not hasattr(
            self.async_rollout_manager,
            "generate_training_prompt_groups",
        ):
            raise RuntimeError(
                "Streaming training requires "
                "visharness.agent_loop.dynamic_validation_manager."
                "VisHarnessAgentLoopManager"
            )
        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            raise RuntimeError(
                "Streaming VisHarness group filtering does not support REMAX baselines"
            )

        prompt_bsz = int(self.config.data.train_batch_size)
        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        if prompt_bsz <= 0 or rollout_n <= 0:
            raise ValueError(
                f"Invalid streaming rollout sizes: prompts={prompt_bsz}, n={rollout_n}"
            )

        accepted_batches: dict[int, DataProto] = {}
        slot_sources: dict[int, str | None] = {}
        slot_attempts: defaultdict[int, int] = defaultdict(int)
        generated_prompt_attempts = 0
        generated_trajectory_attempts = 0
        rollout_invalid_trajectory_attempts = 0
        rollout_prompt_overlong_attempts = 0
        rollout_tool_oom_invalid_attempts = 0
        rollout_undersized_prompt_group_attempts = 0
        rollout_undersized_prompt_group_trajectory_attempts = 0
        max_num_gen_batches = int(
            self.config.algorithm.filter_groups.max_num_gen_batches
        )

        def register_prompt_attempt(
            *,
            source: str | None,
            is_topup: bool,
            completed_cycles_by_source: dict[str, int],
        ) -> None:
            nonlocal generated_prompt_attempts
            generated_prompt_attempts += 1
            if task_sampling_state is None:
                return
            if source is None:
                raise RuntimeError(
                    "Task-balanced streaming rollout received a prompt without a source"
                )
            task_sampling_state["attempted"][source] += 1
            if is_topup:
                task_sampling_state["topup_attempted"][source] += 1
            for completed_source, count in completed_cycles_by_source.items():
                task_sampling_state["completed_cycles_during_update"][
                    completed_source
                ] += int(count)

        def prompt_to_job(
            *,
            slot_index: int,
            prompt_batch: DataProto,
            source: str | None,
            is_topup: bool,
            completed_cycles_by_source: dict[str, int],
        ):
            if len(prompt_batch) != 1:
                raise RuntimeError(
                    "A streaming prompt-group job must start from one prompt, got "
                    f"{len(prompt_batch)}"
                )
            register_prompt_attempt(
                source=source,
                is_topup=is_topup,
                completed_cycles_by_source=completed_cycles_by_source,
            )
            generation_batch = self._get_gen_batch(prompt_batch)
            generation_batch = generation_batch.repeat(
                repeat_times=rollout_n,
                interleave=True,
            )
            context = {
                "raw_prompt": prompt_batch,
                "source": source,
                "attempt_index": int(slot_attempts[slot_index]),
            }
            slot_attempts[slot_index] += 1
            return slot_index, generation_batch, context

        def take_initial_jobs():
            if task_sampling_state is not None:
                request_counts = dict(task_sampling_state["original_target"])
                (
                    initial_prompts,
                    selected_counts_by_source,
                    completed_cycles_by_source,
                ) = self._take_cycling_raw_prompts_by_source(request_counts)
                if sum(selected_counts_by_source.values()) != prompt_bsz:
                    raise RuntimeError(
                        "Task-balanced streaming rollout did not draw its complete "
                        f"initial quota: selected={selected_counts_by_source}, "
                        f"expected={request_counts}"
                    )
            else:
                initial_prompts = self._take_raw_prompts(
                    prompt_bsz,
                    next_raw_prompt_batch,
                )
                completed_cycles_by_source = {}
                if initial_prompts is None or len(initial_prompts) != prompt_bsz:
                    available = 0 if initial_prompts is None else len(initial_prompts)
                    raise RuntimeError(
                        "Streaming group filtering requires a complete initial actor "
                        f"batch of {prompt_bsz} prompts, got {available}"
                    )

            jobs = []
            for slot_index in range(prompt_bsz):
                prompt_batch = initial_prompts.slice(slot_index, slot_index + 1)
                source_values = prompt_batch.non_tensor_batch.get("data_source")
                source = (
                    str(source_values[0])
                    if source_values is not None
                    else None
                )
                slot_sources[slot_index] = source
                # A multi-source draw reports all crossed cycles once. Commit
                # that delta with the first row rather than once per prompt.
                cycle_delta = completed_cycles_by_source if slot_index == 0 else {}
                jobs.append(
                    prompt_to_job(
                        slot_index=slot_index,
                        prompt_batch=prompt_batch,
                        source=source,
                        is_topup=False,
                        completed_cycles_by_source=cycle_delta,
                    )
                )
            return jobs

        def take_replacement_job(slot_index: int):
            source = slot_sources[slot_index]
            next_attempt_index = int(slot_attempts[slot_index])
            if (
                max_num_gen_batches > 0
                and next_attempt_index >= max_num_gen_batches
            ):
                raise ValueError(
                    "Streaming prompt-group slot exhausted its configured generation "
                    f"attempts: slot={slot_index}, attempts={next_attempt_index}, "
                    f"max_num_gen_batches={max_num_gen_batches}"
                )

            if task_sampling_state is not None:
                if source is None:
                    raise RuntimeError(
                        f"Task-balanced quota slot {slot_index} has no data source"
                    )
                (
                    replacement_prompt,
                    selected_counts_by_source,
                    completed_cycles_by_source,
                ) = self._take_cycling_raw_prompts_by_source({source: 1})
                if selected_counts_by_source != {source: 1}:
                    raise RuntimeError(
                        "Task-balanced streaming replacement changed source: "
                        f"slot={slot_index}, expected={source!r}, "
                        f"selected={selected_counts_by_source}"
                    )
            else:
                replacement_prompt = self._take_raw_prompts(
                    1,
                    next_raw_prompt_batch,
                )
                completed_cycles_by_source = {}
                if replacement_prompt is None:
                    raise RuntimeError(
                        "Training data were exhausted before streaming group filtering "
                        f"could fill quota slot {slot_index}"
                    )

            return prompt_to_job(
                slot_index=slot_index,
                prompt_batch=replacement_prompt,
                source=source,
                is_topup=True,
                completed_cycles_by_source=completed_cycles_by_source,
            )

        def process_completed_group(
            slot_index: int,
            context: dict[str, Any],
            generation_output: DataProto,
        ):
            nonlocal generated_trajectory_attempts
            nonlocal rollout_invalid_trajectory_attempts
            nonlocal rollout_prompt_overlong_attempts
            nonlocal rollout_tool_oom_invalid_attempts
            nonlocal rollout_undersized_prompt_group_attempts
            nonlocal rollout_undersized_prompt_group_trajectory_attempts

            if len(generation_output) != rollout_n:
                raise RuntimeError(
                    "Streaming prompt-group rollout returned an incomplete group: "
                    f"slot={slot_index}, expected={rollout_n}, "
                    f"returned={len(generation_output)}"
                )
            generated_trajectory_attempts += len(generation_output)
            if task_sampling_state is not None:
                completed_source = context["source"]
                if completed_source is None:
                    raise RuntimeError(
                        f"Completed task-balanced slot {slot_index} has no source"
                    )
                # Count only completed attempts for the no-progress guard.
                # Initial jobs for other slots may still be in flight, so
                # treating their submission as a failed cycle can abort the
                # update before their rewards are even available.
                task_sampling_state["attempted_since_progress"][
                    completed_source
                ] += 1

            raw_prompt = context["raw_prompt"]
            completed_batch = raw_prompt.repeat(
                repeat_times=rollout_n,
                interleave=True,
            )
            completed_batch = completed_batch.union(generation_output)
            (
                completed_batch,
                dropped_invalid_trajectories,
                invalid_reason_counts,
            ) = self._drop_invalid_trajectories(completed_batch)
            rollout_invalid_trajectory_attempts += dropped_invalid_trajectories
            rollout_prompt_overlong_attempts += invalid_reason_counts.get(
                "rollout_prompt_overlong",
                0,
            )
            rollout_tool_oom_invalid_attempts += invalid_reason_counts.get(
                "tool_oom_retry_exhausted",
                0,
            )
            if dropped_invalid_trajectories:
                reason_summary = ", ".join(
                    f"{reason}={count}"
                    for reason, count in sorted(invalid_reason_counts.items())
                )
                print(
                    "Dropped "
                    f"{dropped_invalid_trajectories} invalid trajectory/trajectories "
                    f"from streaming slot {slot_index} before reward/filter "
                    f"({reason_summary}).",
                    flush=True,
                )

            if completed_batch is not None:
                (
                    completed_batch,
                    dropped_undersized_trajectories,
                    dropped_undersized_prompt_groups,
                ) = self._maybe_drop_undersized_prompt_groups(completed_batch)
                rollout_undersized_prompt_group_attempts += (
                    dropped_undersized_prompt_groups
                )
                rollout_undersized_prompt_group_trajectory_attempts += (
                    dropped_undersized_trajectories
                )

            if completed_batch is not None:
                with marked_timer("reward", timing_raw, "yellow"):
                    if self.use_rm and "rm_scores" not in completed_batch.batch.keys():
                        batch_reward = self._compute_reward_colocate(completed_batch)
                        completed_batch = completed_batch.union(batch_reward)

                    reward_tensor, reward_extra_infos_dict = extract_reward(
                        completed_batch
                    )
                    completed_batch.batch["token_level_scores"] = reward_tensor
                    if reward_extra_infos_dict:
                        completed_batch.non_tensor_batch.update(
                            {
                                key: np.asarray(values)
                                for key, values in reward_extra_infos_dict.items()
                            }
                        )
                    completed_batch.batch["token_level_rewards"] = (
                        completed_batch.batch["token_level_scores"]
                    )

                self._accumulate_trajectory_metrics(
                    raw_trajectory_metrics,
                    completed_batch,
                )
                (
                    filtered_batch,
                    kept_prompt_count,
                    kept_counts_by_source,
                ) = self._apply_filter_groups(
                    new_batch=completed_batch,
                    batch=None,
                    metrics=metrics,
                    num_prompt_in_batch=0,
                )
            else:
                filtered_batch = None
                kept_prompt_count = 0
                kept_counts_by_source = {}

            if kept_prompt_count == 1:
                if slot_index in accepted_batches:
                    raise RuntimeError(
                        f"Streaming quota slot {slot_index} was accepted more than once"
                    )
                if filtered_batch is None or len(filtered_batch) == 0:
                    raise RuntimeError(
                        f"Accepted streaming slot {slot_index} has no trajectories"
                    )
                accepted_batches[slot_index] = filtered_batch
                if task_sampling_state is not None:
                    source = slot_sources[slot_index]
                    if kept_counts_by_source != {source: 1}:
                        raise RuntimeError(
                            "Accepted streaming group does not match its quota source: "
                            f"slot={slot_index}, expected={source!r}, "
                            f"kept={kept_counts_by_source}"
                        )
                    task_sampling_state["kept"][source] += 1
            elif kept_prompt_count != 0:
                raise RuntimeError(
                    "One streaming job must select zero or one prompt group, got "
                    f"{kept_prompt_count} for slot {slot_index}"
                )

            if task_sampling_state is not None:
                self._record_task_sampling_filter_progress(task_sampling_state)

            if kept_prompt_count == 1:
                return None
            return take_replacement_job(slot_index)

        initial_jobs = take_initial_jobs()
        scheduler_stats = self.async_rollout_manager.generate_training_prompt_groups(
            initial_jobs,
            on_complete=process_completed_group,
            total_slots=prompt_bsz,
            trajectories_per_group=rollout_n,
            progress_interval=1,
        )

        if len(accepted_batches) != prompt_bsz:
            raise RuntimeError(
                "Streaming rollout scheduler returned without a complete actor batch: "
                f"accepted={sorted(accepted_batches)}, expected={prompt_bsz}"
            )
        if int(scheduler_stats["submitted_groups"]) != generated_prompt_attempts:
            raise RuntimeError(
                "Streaming scheduler submission count disagrees with sampler state: "
                f"scheduler={scheduler_stats['submitted_groups']}, "
                f"sampler={generated_prompt_attempts}"
            )

        ordered_batches = [
            accepted_batches[slot_index]
            for slot_index in range(prompt_bsz)
        ]
        _normalize_worker_output_fields(ordered_batches)
        batch = DataProto.concat(ordered_batches)
        num_prompt_in_batch = len(accepted_batches)

        if task_sampling_state is not None:
            final_kept = {
                source: int(task_sampling_state["kept"].get(source, 0))
                for source in self._task_sampling_plan.sources
            }
            if final_kept != task_sampling_state["original_target"]:
                raise RuntimeError(
                    "Task-balanced streaming actor update does not satisfy its exact "
                    f"post-filter quota: target={task_sampling_state['original_target']}, "
                    f"kept={final_kept}"
                )
            task_sampling_state["source_progress_snapshot"] = (
                self._task_source_pool.progress_by_source()
            )
            task_sampling_state["epoch_completed_after_update"] = (
                self._task_source_pool.all_sources_first_pass_completed
            )

        max_attempt_depth = max(slot_attempts.values(), default=1)
        self.gen_steps += max(max_attempt_depth - 1, 0)
        timing_raw.update(scheduler_stats.pop("timing"))
        metrics.update(
            {
                "streaming_rollout/completed_slots": float(
                    scheduler_stats["completed_slots"]
                ),
                "streaming_rollout/submitted_prompt_groups": float(
                    scheduler_stats["submitted_groups"]
                ),
                "streaming_rollout/peak_in_flight_groups": float(
                    scheduler_stats["peak_in_flight_groups"]
                ),
                "streaming_rollout/worker_count": float(
                    scheduler_stats["worker_count"]
                ),
                "streaming_rollout/wall_seconds": float(
                    scheduler_stats["wall_seconds"]
                ),
                "streaming_rollout/max_attempt_depth": float(max_attempt_depth),
            }
        )

        _, reward_extra_infos_dict = extract_reward(batch)
        return {
            "batch": batch,
            "num_prompt_in_batch": num_prompt_in_batch,
            "num_gen_batches": max_attempt_depth,
            "generated_prompt_attempts": generated_prompt_attempts,
            "generated_trajectory_attempts": generated_trajectory_attempts,
            "rollout_invalid_trajectory_attempts": (
                rollout_invalid_trajectory_attempts
            ),
            "rollout_prompt_overlong_attempts": rollout_prompt_overlong_attempts,
            "rollout_tool_oom_invalid_attempts": (
                rollout_tool_oom_invalid_attempts
            ),
            "rollout_undersized_prompt_group_attempts": (
                rollout_undersized_prompt_group_attempts
            ),
            "rollout_undersized_prompt_group_trajectory_attempts": (
                rollout_undersized_prompt_group_trajectory_attempts
            ),
            "reward_extra_infos_dict": reward_extra_infos_dict,
        }

    def _collect_synchronous_rollouts(
        self,
        *,
        task_sampling_state: dict | None,
        next_raw_prompt_batch,
        metrics: dict,
        timing_raw: dict,
        raw_trajectory_metrics: dict,
    ) -> dict[str, Any] | None:
        """Collect the non-streaming path used when reward filtering is disabled."""

        prompt_bsz = int(self.config.data.train_batch_size)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        generated_prompt_attempts = 0
        generated_trajectory_attempts = 0
        rollout_invalid_trajectory_attempts = 0
        rollout_prompt_overlong_attempts = 0
        rollout_tool_oom_invalid_attempts = 0
        rollout_undersized_prompt_group_attempts = 0
        rollout_undersized_prompt_group_trajectory_attempts = 0
        reward_extra_infos_dict = {}

        while True:
            selected_counts_by_source: dict[str, int] = {}
            if task_sampling_state is not None:
                deficits = self._task_quota_deficits(
                    task_sampling_state["original_target"],
                    task_sampling_state["kept"],
                )
                if not deficits:
                    raise RuntimeError(
                        "Task-balanced sampling entered generation without a quota "
                        f"deficit: state={task_sampling_state}"
                    )
                (
                    new_batch,
                    selected_counts_by_source,
                    completed_cycles_by_source,
                ) = self._take_cycling_raw_prompts_by_source(dict(deficits))
                for source, count in completed_cycles_by_source.items():
                    task_sampling_state["completed_cycles_during_update"][source] += int(
                        count
                    )
            else:
                target_prompt_count = prompt_bsz
                if self._group_filter_enabled():
                    target_prompt_count = max(
                        prompt_bsz - num_prompt_in_batch,
                        1,
                    )
                new_batch = self._take_raw_prompts(
                    target_prompt_count,
                    next_raw_prompt_batch,
                )
            if new_batch is None:
                return None

            prompt_attempt_count = len(new_batch)
            generated_prompt_attempts += prompt_attempt_count
            if task_sampling_state is not None:
                if sum(selected_counts_by_source.values()) != prompt_attempt_count:
                    raise RuntimeError(
                        "Selected task counts do not match prompt batch length: "
                        f"{selected_counts_by_source} vs {prompt_attempt_count}"
                    )
                for source, count in selected_counts_by_source.items():
                    task_sampling_state["attempted"][source] += int(count)
                    task_sampling_state["attempted_since_progress"][source] += int(
                        count
                    )
                    if num_gen_batches > 0:
                        task_sampling_state["topup_attempted"][source] += int(count)
            num_gen_batches += 1

            gen_batch = self._get_gen_batch(new_batch)
            generation_input = gen_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.n,
                interleave=True,
            )

            with marked_timer("gen", timing_raw, "red"):
                generation_output = self.async_rollout_manager.generate_sequences(
                    generation_input
                )
                generated_trajectory_attempts += len(generation_output)
                timing_raw.update(generation_output.meta_info["timing"])
                generation_output.meta_info.pop("timing", None)

            if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                with marked_timer("gen_max", timing_raw, "red"):
                    gen_baseline_batch = deepcopy(gen_batch)
                    gen_baseline_batch.meta_info["do_sample"] = False
                    gen_baseline_output = self.async_rollout_manager.generate_sequences(
                        gen_baseline_batch
                    )
                    new_batch = new_batch.union(gen_baseline_output)
                    rm_scores = None
                    if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                        rm_scores = self._compute_reward_colocate(new_batch)
                        new_batch = new_batch.union(rm_scores)
                    reward_baseline_tensor, _ = extract_reward(new_batch)
                    reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
                    keys_to_pop = set(gen_baseline_output.batch.keys())
                    if rm_scores is not None:
                        keys_to_pop.update(rm_scores.batch.keys())
                    new_batch.pop(batch_keys=list(keys_to_pop))
                    new_batch.batch["reward_baselines"] = reward_baseline_tensor

            new_batch = new_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.n,
                interleave=True,
            )
            new_batch = new_batch.union(generation_output)

            (
                new_batch,
                dropped_invalid_trajectories,
                invalid_reason_counts,
            ) = self._drop_invalid_trajectories(new_batch)
            rollout_invalid_trajectory_attempts += dropped_invalid_trajectories
            rollout_prompt_overlong_attempts += invalid_reason_counts.get(
                "rollout_prompt_overlong",
                0,
            )
            rollout_tool_oom_invalid_attempts += invalid_reason_counts.get(
                "tool_oom_retry_exhausted",
                0,
            )
            if new_batch is not None:
                (
                    new_batch,
                    dropped_undersized_trajectories,
                    dropped_undersized_prompt_groups,
                ) = self._maybe_drop_undersized_prompt_groups(new_batch)
                rollout_undersized_prompt_group_attempts += (
                    dropped_undersized_prompt_groups
                )
                rollout_undersized_prompt_group_trajectory_attempts += (
                    dropped_undersized_trajectories
                )

            if new_batch is None:
                if task_sampling_state is not None:
                    self._record_task_sampling_filter_progress(task_sampling_state)
                max_batches = int(
                    self.config.algorithm.filter_groups.max_num_gen_batches
                )
                if max_batches <= 0 or num_gen_batches < max_batches:
                    self.gen_steps += 1
                    continue
                raise ValueError(
                    "No usable trajectories remain after rollout filtering and "
                    f"num_gen_batches={num_gen_batches} reached {max_batches}"
                )

            with marked_timer("reward", timing_raw, "yellow"):
                if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                    batch_reward = self._compute_reward_colocate(new_batch)
                    new_batch = new_batch.union(batch_reward)
                reward_tensor, reward_extra_infos_dict = extract_reward(new_batch)
                new_batch.batch["token_level_scores"] = reward_tensor
                if reward_extra_infos_dict:
                    new_batch.non_tensor_batch.update(
                        {
                            key: np.asarray(values)
                            for key, values in reward_extra_infos_dict.items()
                        }
                    )
                new_batch.batch["token_level_rewards"] = new_batch.batch[
                    "token_level_scores"
                ]

            self._accumulate_trajectory_metrics(raw_trajectory_metrics, new_batch)
            if not self._group_filter_enabled():
                batch = (
                    new_batch
                    if batch is None
                    else DataProto.concat([batch, new_batch])
                )
                if task_sampling_state is not None:
                    kept_counts_by_source = self._prompt_group_counts_by_source(
                        new_batch
                    )
                    for source, count in kept_counts_by_source.items():
                        task_sampling_state["kept"][source] += int(count)
                    num_prompt_in_batch = sum(task_sampling_state["kept"].values())
                else:
                    num_prompt_in_batch += prompt_attempt_count
            else:
                batch, num_prompt_in_batch, kept_counts_by_source = (
                    self._apply_filter_groups(
                        new_batch=new_batch,
                        batch=batch,
                        metrics=metrics,
                        num_prompt_in_batch=num_prompt_in_batch,
                    )
                )
                if task_sampling_state is not None:
                    for source, count in kept_counts_by_source.items():
                        task_sampling_state["kept"][source] += int(count)

            if task_sampling_state is not None:
                self._record_task_sampling_filter_progress(task_sampling_state)
                remaining_deficits = self._task_quota_deficits(
                    task_sampling_state["original_target"],
                    task_sampling_state["kept"],
                )
                needs_topup = bool(remaining_deficits)
            elif self._group_filter_enabled():
                needs_topup = num_prompt_in_batch < prompt_bsz
            else:
                needs_topup = False

            if needs_topup:
                max_batches = int(
                    self.config.algorithm.filter_groups.max_num_gen_batches
                )
                if max_batches <= 0 or num_gen_batches < max_batches:
                    self.gen_steps += 1
                    continue
                raise ValueError(
                    f"num_gen_batches={num_gen_batches} reached "
                    f"max_num_gen_batches={max_batches} before filling the actor batch"
                )
            break

        if task_sampling_state is not None:
            final_kept = {
                source: int(task_sampling_state["kept"].get(source, 0))
                for source in self._task_sampling_plan.sources
            }
            if final_kept != task_sampling_state["original_target"]:
                raise RuntimeError(
                    "Task-balanced actor update does not satisfy its exact post-filter "
                    f"quota: target={task_sampling_state['original_target']}, "
                    f"kept={final_kept}"
                )
            task_sampling_state["source_progress_snapshot"] = (
                self._task_source_pool.progress_by_source()
            )
            task_sampling_state["epoch_completed_after_update"] = (
                self._task_source_pool.all_sources_first_pass_completed
            )

        if self._group_filter_enabled():
            batch, num_prompt_in_batch = self._truncate_to_prompt_groups(
                batch,
                prompt_bsz,
            )
        _, reward_extra_infos_dict = extract_reward(batch)
        return {
            "batch": batch,
            "num_prompt_in_batch": num_prompt_in_batch,
            "num_gen_batches": num_gen_batches,
            "generated_prompt_attempts": generated_prompt_attempts,
            "generated_trajectory_attempts": generated_trajectory_attempts,
            "rollout_invalid_trajectory_attempts": (
                rollout_invalid_trajectory_attempts
            ),
            "rollout_prompt_overlong_attempts": rollout_prompt_overlong_attempts,
            "rollout_tool_oom_invalid_attempts": (
                rollout_tool_oom_invalid_attempts
            ),
            "rollout_undersized_prompt_group_attempts": (
                rollout_undersized_prompt_group_attempts
            ),
            "rollout_undersized_prompt_group_trajectory_attempts": (
                rollout_undersized_prompt_group_trajectory_attempts
            ),
            "reward_extra_infos_dict": reward_extra_infos_dict,
        }

    def _archive_hf_checkpoint_enabled(self) -> bool:
        archive_config = self.config.get("visharness", {}).get("archive_checkpoints", {})
        return bool(archive_config.get("enable", False))

    def _get_archived_checkpoint_dir(self) -> str:
        archive_config = self.config.get("visharness", {}).get("archive_checkpoints", {})
        archive_dir = archive_config.get("dir", None)
        if archive_dir:
            return os.path.abspath(os.path.expanduser(str(archive_dir)))
        return os.path.join(os.path.abspath(self.config.trainer.default_local_dir), "archived")

    @staticmethod
    def _hf_checkpoint_has_weights(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        for filename in os.listdir(path):
            if filename.endswith((".safetensors", ".bin")):
                return True
        return False

    def _archive_current_hf_checkpoint(self) -> None:
        source_dir = os.path.join(
            self.config.trainer.default_local_dir,
            f"global_step_{self.global_steps}",
            "actor",
            "huggingface",
        )
        target_dir = os.path.join(self._get_archived_checkpoint_dir(), f"global_step_{self.global_steps}")
        if not os.path.isdir(source_dir):
            raise FileNotFoundError(
                f"Cannot archive HF checkpoint because {source_dir} does not exist. "
                "Make sure actor_rollout_ref.actor.checkpoint.save_contents contains 'hf_model'."
            )
        if not self._hf_checkpoint_has_weights(source_dir):
            raise FileNotFoundError(
                f"Cannot archive HF checkpoint because {source_dir} does not contain model weight files. "
                "Set actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model']."
            )

        parent_dir = os.path.dirname(target_dir)
        os.makedirs(parent_dir, exist_ok=True)
        tmp_dir = f"{target_dir}.tmp"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        shutil.copytree(source_dir, tmp_dir, symlinks=True)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.rename(tmp_dir, target_dir)
        print(f"Archived HF checkpoint for inference: {target_dir}")

    def _save_checkpoint(self):
        # The base implementation publishes latest_checkpointed_iteration.txt
        # at the end of its save. Persist the VisHarness-only sampler buffers
        # first so a published checkpoint is never missing state required for
        # an exact resume. A failed base save leaves only an unpublished orphan
        # directory, which auto-resume will ignore.
        self._save_visharness_resume_state()
        super()._save_checkpoint()
        if self._archive_hf_checkpoint_enabled():
            self._archive_current_hf_checkpoint()
        self._prune_resume_checkpoint_roles()

    def _prune_resume_checkpoint_roles(self) -> None:
        """Enforce role checkpoint retention after a fully published save.

        Upstream verl keeps role paths in process memory, so that list is empty
        after resume. VisHarness instead rediscovers exact-resume checkpoints
        from disk after each successful save. Keeping this in the custom
        trainer avoids modifying verl's FSDP checkpoint implementation.
        """

        checkpoint_config = self.config.actor_rollout_ref.actor.checkpoint
        if bool(checkpoint_config.get("async_save", False)):
            print(
                "Skipping VisHarness checkpoint pruning because async_save=true; "
                "the checkpoint may not be complete when the trainer returns.",
                flush=True,
            )
            return

        checkpoint_root = os.path.abspath(
            os.path.expanduser(str(self.config.trainer.default_local_dir))
        )
        role_limits = [
            (
                "actor",
                self.config.trainer.get("max_actor_ckpt_to_keep", None),
            )
        ]
        if self.use_critic:
            role_limits.append(
                (
                    "critic",
                    self.config.trainer.get("max_critic_ckpt_to_keep", None),
                )
            )

        for role, raw_limit in role_limits:
            limit = None if raw_limit is None else int(raw_limit)
            removed_paths = prune_resume_checkpoints(
                checkpoint_root,
                role=role,
                max_to_keep=limit,
            )
            for removed_path in removed_paths:
                print(
                    "VisHarness checkpoint retention removed old "
                    f"{role} checkpoint: {removed_path}",
                    flush=True,
                )

    def _load_checkpoint(self):
        result = super()._load_checkpoint()
        # Upstream returns 0 when no checkpoint was selected and None after a
        # successful load. Keep this distinction because global_step_0 can be a
        # legitimate terminal checkpoint when no complete update can be formed.
        self._visharness_checkpoint_loaded = result is None
        self._maybe_restore_dataloader_state_after_base_load()
        return result

    def _finalize_naturally_exhausted_training(
        self,
        *,
        last_completed_step: int,
        last_validation_step: int | None,
        last_saved_step: int | None,
        logger,
        progress_bar,
        refresh_update_progress,
        consumed_prompt_examples: int,
    ) -> dict[str, Any] | None:
        """Finalize the last completed update after the prompt stream is exhausted."""

        attempted_step = int(self.global_steps)
        self.global_steps = int(last_completed_step)
        self._visharness_training_completed = True
        self._visharness_completion_reason = "data_exhausted"
        if attempted_step != self.global_steps:
            print(
                "Training data exhausted before update "
                f"{attempted_step} could be formed; finalizing the last completed "
                f"update {self.global_steps}.",
                flush=True,
            )

        metrics: dict[str, Any] = {}
        timing_raw = defaultdict(float)
        if last_saved_step != self.global_steps:
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
        else:
            # A periodic checkpoint may already contain the correct weights
            # for this step but predate the failed attempt to form the next
            # update. Refresh both data.pt and the VisHarness state so they
            # describe the same final exhausted iterator/buffer state without
            # re-saving model shards under the same checkpoint path.
            with marked_timer("save_runtime_state", timing_raw, "green"):
                self._refresh_checkpoint_runtime_state()
        metrics.update({f"timing/{key}": value for key, value in timing_raw.items()})

        final_val_metrics = None
        if (
            self.config.trainer.test_freq > 0
            and last_validation_step != self.global_steps
        ):
            validation_timing = defaultdict(float)
            with marked_timer("testing", validation_timing, "green"):
                final_val_metrics = self._validate()
            last_validation_step = int(self.global_steps)
            self._visharness_last_validation_step = int(self.global_steps)
            metrics.update(final_val_metrics)
            metrics.update(
                {f"timing/{key}": value for key, value in validation_timing.items()}
            )

        if metrics:
            logger.log(data=self._filter_logged_metrics(metrics), step=self.global_steps)
        if final_val_metrics is not None:
            # Persist the marker after both validation and metric logging. If
            # either fails, resume will rerun validation rather than silently
            # treating an unlogged result as complete.
            self._save_visharness_resume_state()
        refresh_update_progress(self.global_steps, consumed_prompt_examples)
        if final_val_metrics is not None:
            pprint(f"Final validation metrics: {final_val_metrics}")
        progress_bar.close()
        return final_val_metrics

    def fit(self):
        """DAPO training loop with optional VisHarness reward-group filtering.

        This intentionally keeps verl's upstream DAPO source untouched. In
        ordinary task-balanced GRPO, prompt sources use independent cycling
        pools, post-filter deficits are refilled from the same source, and an
        epoch ends only after every source completes its first pass.
        """

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        self.global_steps = 0
        self.gen_steps = 0
        self.max_steps_duration = 0
        self._visharness_training_completed = False
        self._visharness_completion_reason = None
        self._visharness_last_validation_step = None

        self._load_checkpoint()
        if len(self.train_dataloader) == 0:
            raise ValueError("Train dataloader is empty; check train data path and prompt length filtering")
        initial_global_steps = int(self.global_steps)
        prompt_bsz_for_progress = max(int(self.config.data.train_batch_size), 1)
        if self._task_sampling_enabled():
            total_raw_prompt_examples = (
                sum(self._task_sampling_plan.source_counts.values())
                * int(self.config.trainer.total_epochs)
            )
        else:
            try:
                total_raw_prompt_examples = int(len(self.train_dataset)) * int(self.config.trainer.total_epochs)
            except Exception:
                total_raw_prompt_examples = (
                    int(len(self.train_dataloader))
                    * prompt_bsz_for_progress
                    * int(self.config.trainer.total_epochs)
                )
        estimated_prompt_examples_consumed = initial_global_steps * prompt_bsz_for_progress
        restored_dataloader_samples_yielded = (
            None
            if self._task_sampling_enabled()
            else self._train_dataloader_samples_yielded()
        )
        resume_state = self._load_visharness_resume_state()
        self._validate_task_sampling_resume_state(resume_state)
        self._validate_train_dataset_resume_state(resume_state)
        self._validate_dataloader_resume_state(resume_state)
        if self._task_sampling_enabled():
            saved_pool_state = resume_state.get("task_source_pool_state")
            if resume_state and saved_pool_state is None:
                raise ValueError(
                    "Task-balanced checkpoint has no task_source_pool_state; "
                    "exact resume is not possible."
                )
            if saved_pool_state is not None:
                self._task_source_pool.load_state_dict(saved_pool_state)
            current_epoch = int(self._task_source_pool.epoch_index)
            if current_epoch > int(self.config.trainer.total_epochs):
                raise ValueError(
                    "Checkpoint task epoch exceeds trainer.total_epochs: "
                    f"task_epoch={current_epoch}, "
                    f"total_epochs={self.config.trainer.total_epochs}"
                )
            saved_train_epoch = resume_state.get("train_epoch")
            if saved_train_epoch is not None and int(saved_train_epoch) != current_epoch:
                raise ValueError(
                    "Checkpoint task epoch disagrees with the restored source pool: "
                    f"train_epoch={saved_train_epoch}, pool_epoch={current_epoch}"
                )
            self._visharness_train_epoch = current_epoch
            self._visharness_epoch_start_global_step = int(
                resume_state.get("epoch_start_global_step", 0)
            )
            if self._visharness_epoch_start_global_step > initial_global_steps:
                raise ValueError(
                    "Checkpoint epoch_start_global_step is newer than global_steps: "
                    f"epoch_start={self._visharness_epoch_start_global_step}, "
                    f"global_steps={initial_global_steps}"
                )
            restored_raw_prompt_buffer = None
            restored_buffer_size = 0
            prompt_examples_consumed_before_fit = sum(
                int(progress["total_draws"])
                for progress in self._task_source_pool.progress_by_source().values()
            )
        else:
            restored_raw_prompt_buffer = self._raw_prompt_buffer_from_resume_state(resume_state)
            restored_buffer_size = (
                len(restored_raw_prompt_buffer)
                if restored_raw_prompt_buffer is not None
                else 0
            )
        if not self._task_sampling_enabled():
            self._visharness_raw_prompt_buffer = restored_raw_prompt_buffer
            self._visharness_raw_prompt_buffers = {}
            restored_prompt_examples_consumed = self._effective_restored_prompt_examples(
                restored_dataloader_samples_yielded,
                restored_raw_prompt_buffer,
            )
            current_epoch = self._infer_current_train_epoch(
                resume_state=resume_state,
                restored_prompt_examples_consumed=restored_dataloader_samples_yielded,
                prompt_bsz_for_progress=prompt_bsz_for_progress,
            )
            self._visharness_train_epoch = int(current_epoch)
            self._visharness_epoch_start_global_step = int(
                resume_state.get("epoch_start_global_step", 0)
            )
        resume_completed, resume_completion_reason = self._resume_training_completion(
            resume_state
        )
        self._visharness_training_completed = bool(resume_completed)
        self._visharness_completion_reason = resume_completion_reason
        if self._task_sampling_enabled():
            reached_epoch_limit = current_epoch >= int(
                self.config.trainer.total_epochs
            )
            if bool(resume_completed) != bool(reached_epoch_limit):
                raise ValueError(
                    "Checkpoint task epoch and training completion flag disagree: "
                    f"task_epoch={current_epoch}, "
                    f"total_epochs={self.config.trainer.total_epochs}, "
                    f"training_completed={resume_completed}"
                )
        last_validation_step = self._restored_last_validation_step(resume_state)
        self._visharness_last_validation_step = last_validation_step
        if not self._task_sampling_enabled():
            if restored_prompt_examples_consumed is not None:
                prompt_examples_consumed_before_fit = max(
                    int(restored_prompt_examples_consumed),
                    int(estimated_prompt_examples_consumed),
                )
                if (
                    initial_global_steps > 0
                    and restored_prompt_examples_consumed
                    != estimated_prompt_examples_consumed
                ):
                    print(
                        "Progress resume: using effective consumed prompts="
                        f"{restored_prompt_examples_consumed} "
                        f"(dataloader samples_yielded={restored_dataloader_samples_yielded}, "
                        f"restored buffer={restored_buffer_size}) "
                        "instead of global_steps*train_batch_size="
                        f"{estimated_prompt_examples_consumed}."
                    )
            else:
                prompt_examples_consumed_before_fit = (
                    estimated_prompt_examples_consumed
                )
            prompt_examples_consumed_before_fit = min(
                int(prompt_examples_consumed_before_fit),
                total_raw_prompt_examples,
            )
        prompt_examples_consumed_this_fit = 0

        # All lightweight VisHarness/runtime compatibility checks above must
        # pass before synchronizing rollout weights, opening the tracker, or
        # spending time on validation.
        self.checkpoint_manager.update_weights()
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        last_completed_step = int(self.global_steps)
        last_saved_step: int | None = (
            int(self.global_steps) if int(self.global_steps) > 0 else None
        )

        val_before_train = bool(self.config.trainer.get("val_before_train", True))
        val_only = bool(self.config.trainer.get("val_only", False))
        should_run_initial_validation = val_only or (
            val_before_train and last_validation_step != int(self.global_steps)
        )
        if should_run_initial_validation:
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=self._filter_logged_metrics(val_metrics), step=self.global_steps)
            last_validation_step = int(self.global_steps)
            self._visharness_last_validation_step = int(self.global_steps)
            if self.global_steps > 0 and os.path.isdir(
                self._resume_checkpoint_dir_for_current_step()
            ):
                self._save_visharness_resume_state()
        elif val_before_train:
            print(
                f"Skipping initial validation: global_step_{self.global_steps} already has "
                "a complete validation result.",
                flush=True,
            )
        if val_only:
            return
        if resume_completed:
            print(
                "The selected checkpoint already represents completed training "
                f"({resume_completion_reason or 'completed'}); no additional actor update is required.",
                flush=True,
            )
            return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training Progress",
            unit="update",
        )

        def refresh_update_progress(completed_updates: int, consumed_prompt_examples: int) -> None:
            completed_updates = max(0, int(completed_updates))
            postfix: dict[str, Any]
            if self._task_sampling_enabled():
                task_epoch = int(self._visharness_train_epoch)
                total_epochs = int(self.config.trainer.total_epochs)
                progress_by_source = self._task_source_pool.progress_by_source()
                epoch_updates = max(
                    completed_updates
                    - int(self._visharness_epoch_start_global_step),
                    0,
                )
                remaining_current_epoch = 0
                estimated_full_epoch_updates = 0
                for source in self._task_sampling_plan.sources:
                    progress = progress_by_source[source]
                    if epoch_updates > 0 and int(progress["draws_in_epoch"]) > 0:
                        attempted_per_update = (
                            float(progress["draws_in_epoch"])
                            / float(epoch_updates)
                        )
                    else:
                        attempted_per_update = max(
                            float(self._task_sampling_plan.probabilities[source])
                            * float(self._task_sampling_plan.batch_size),
                            1e-12,
                        )
                    remaining_first_pass = max(
                        int(progress["source_size"])
                        - int(progress["first_pass_draws"]),
                        0,
                    )
                    remaining_current_epoch = max(
                        remaining_current_epoch,
                        int(math.ceil(remaining_first_pass / attempted_per_update)),
                    )
                    estimated_full_epoch_updates = max(
                        estimated_full_epoch_updates,
                        int(
                            math.ceil(
                                int(progress["source_size"])
                                / attempted_per_update
                            )
                        ),
                    )
                future_epochs = max(total_epochs - task_epoch - 1, 0)
                if task_epoch >= total_epochs:
                    estimated_remaining_updates = 0
                else:
                    estimated_remaining_updates = (
                        remaining_current_epoch
                        + future_epochs * estimated_full_epoch_updates
                    )
                coverage_postfix = {
                    self._metric_source_name(source): (
                        f"{float(progress_by_source[source]['first_pass_coverage']):.1%}"
                        f"/c{int(progress_by_source[source]['cycles_completed'])}"
                    )
                    for source in self._task_sampling_plan.sources
                }
                postfix = {
                    "updated": f"{completed_updates}/{completed_updates + estimated_remaining_updates}",
                    "remaining_updates_est": estimated_remaining_updates,
                    "epoch": f"{min(task_epoch + 1, total_epochs)}/{total_epochs}",
                    **coverage_postfix,
                }
            else:
                total_consumed_prompt_examples = min(
                    prompt_examples_consumed_before_fit
                    + max(int(consumed_prompt_examples), 0),
                    total_raw_prompt_examples,
                )
                remaining_prompt_examples = max(
                    total_raw_prompt_examples - total_consumed_prompt_examples,
                    0,
                )
                if completed_updates > 0 and total_consumed_prompt_examples > 0:
                    avg_prompt_examples_per_update = max(
                        float(total_consumed_prompt_examples) / float(completed_updates),
                        1.0,
                    )
                else:
                    avg_prompt_examples_per_update = float(prompt_bsz_for_progress)
                estimated_remaining_updates = int(
                    math.ceil(
                        float(remaining_prompt_examples)
                        / avg_prompt_examples_per_update
                    )
                )
                postfix = {
                    "updated": f"{completed_updates}/{completed_updates + estimated_remaining_updates}",
                    "remaining_updates_est": estimated_remaining_updates,
                }
            estimated_total_updates = max(completed_updates + estimated_remaining_updates, completed_updates, 1)
            self._visharness_estimated_remaining_updates = int(
                estimated_remaining_updates
            )
            self._visharness_estimated_total_updates = int(
                estimated_total_updates
            )
            progress_bar.total = estimated_total_updates
            delta = completed_updates - progress_bar.n
            if delta > 0:
                progress_bar.update(delta)
            elif delta < 0:
                progress_bar.n = completed_updates
            progress_bar.set_postfix(postfix, refresh=True)

        refresh_update_progress(self.global_steps, prompt_examples_consumed_this_fit)

        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_gen_batches = 0
        generated_prompt_attempts = 0
        generated_trajectory_attempts = 0
        rollout_invalid_trajectory_attempts = 0
        rollout_prompt_overlong_attempts = 0
        rollout_tool_oom_invalid_attempts = 0
        rollout_undersized_prompt_group_attempts = 0
        rollout_undersized_prompt_group_trajectory_attempts = 0
        raw_trajectory_metrics = self._new_trajectory_metrics_accumulator()

        def new_task_sampling_state() -> dict | None:
            if not self._task_sampling_enabled():
                return None
            original_target = self._task_quota_for_update(max(int(self.global_steps) - 1, 0))
            if sum(original_target.values()) != int(self.config.data.train_batch_size):
                raise RuntimeError(
                    "Task-sampling quota does not match train batch size: "
                    f"quota={original_target}, train_batch_size={self.config.data.train_batch_size}"
                )
            return {
                "original_target": dict(original_target),
                "kept": defaultdict(int),
                "attempted": defaultdict(int),
                "topup_attempted": defaultdict(int),
                "attempted_since_progress": defaultdict(int),
                "no_progress_cycles": defaultdict(int),
                "kept_at_progress_check": defaultdict(int),
                "completed_cycles_during_update": defaultdict(int),
                "source_progress_snapshot": self._task_source_pool.progress_by_source(),
                "task_epoch": int(self._visharness_train_epoch),
                "epoch_completed_after_update": False,
            }

        task_sampling_state = new_task_sampling_state()
        train_epoch = current_epoch
        self._visharness_train_epoch = int(train_epoch)
        train_iter = None if self._task_sampling_enabled() else iter(self.train_dataloader)

        def next_raw_prompt_batch():
            nonlocal train_epoch, train_iter
            if self._task_sampling_enabled():
                raise RuntimeError(
                    "The shared train DataLoader must not be consumed during "
                    "task-balanced source-pool sampling"
                )
            while train_epoch < self.config.trainer.total_epochs:
                try:
                    return self._prepare_raw_prompt_batch(next(train_iter))
                except StopIteration:
                    train_epoch += 1
                    self._visharness_train_epoch = int(train_epoch)
                    if train_epoch >= self.config.trainer.total_epochs:
                        return None
                    train_iter = iter(self.train_dataloader)
            return None

        while (
            train_epoch < self.config.trainer.total_epochs
            or (
                not self._task_sampling_enabled()
                and self._has_buffered_raw_prompts()
            )
        ):
                self._visharness_train_epoch = int(train_epoch)
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                with marked_timer("step", timing_raw):
                    if self._group_filter_enabled():
                        with marked_timer("gen", timing_raw, "red"):
                            rollout_result = (
                                self._collect_streaming_filtered_rollouts(
                                    task_sampling_state=task_sampling_state,
                                    next_raw_prompt_batch=next_raw_prompt_batch,
                                    metrics=metrics,
                                    timing_raw=timing_raw,
                                    raw_trajectory_metrics=raw_trajectory_metrics,
                                )
                            )
                    else:
                        rollout_result = self._collect_synchronous_rollouts(
                            task_sampling_state=task_sampling_state,
                            next_raw_prompt_batch=next_raw_prompt_batch,
                            metrics=metrics,
                            timing_raw=timing_raw,
                            raw_trajectory_metrics=raw_trajectory_metrics,
                        )
                    if rollout_result is None:
                        break

                    batch = rollout_result["batch"]
                    num_gen_batches += int(rollout_result["num_gen_batches"])
                    generated_prompt_attempts += int(
                        rollout_result["generated_prompt_attempts"]
                    )
                    generated_trajectory_attempts += int(
                        rollout_result["generated_trajectory_attempts"]
                    )
                    rollout_invalid_trajectory_attempts += int(
                        rollout_result["rollout_invalid_trajectory_attempts"]
                    )
                    rollout_prompt_overlong_attempts += int(
                        rollout_result["rollout_prompt_overlong_attempts"]
                    )
                    rollout_tool_oom_invalid_attempts += int(
                        rollout_result["rollout_tool_oom_invalid_attempts"]
                    )
                    rollout_undersized_prompt_group_attempts += int(
                        rollout_result[
                            "rollout_undersized_prompt_group_attempts"
                        ]
                    )
                    rollout_undersized_prompt_group_trajectory_attempts += int(
                        rollout_result[
                            "rollout_undersized_prompt_group_trajectory_attempts"
                        ]
                    )
                    reward_extra_infos_dict = rollout_result[
                        "reward_extra_infos_dict"
                    ]

                    if task_sampling_state is not None:
                        is_last_step = bool(
                            task_sampling_state["epoch_completed_after_update"]
                            and train_epoch + 1
                            >= int(self.config.trainer.total_epochs)
                        )
                    else:
                        is_last_step = (
                            self.global_steps >= self.total_training_steps
                        )
                    kept_trajectory_metrics = self._new_trajectory_metrics_accumulator()
                    self._accumulate_trajectory_metrics(kept_trajectory_metrics, batch)
                    metrics.update(
                        self._summarize_trajectory_metrics(
                            raw_trajectory_metrics,
                            "rollout/eligible",
                        )
                    )
                    metrics.update(
                        self._summarize_trajectory_metrics(
                            kept_trajectory_metrics,
                            "train/selected",
                        )
                    )
                    metrics["sampling/invalid_trajectory_rate"] = rollout_invalid_trajectory_attempts / max(
                        generated_trajectory_attempts, 1
                    )
                    metrics["sampling/prompt_overlong_trajectory_rate"] = rollout_prompt_overlong_attempts / max(
                        generated_trajectory_attempts, 1
                    )
                    metrics["sampling/tool_oom_trajectory_rate"] = rollout_tool_oom_invalid_attempts / max(
                        generated_trajectory_attempts, 1
                    )
                    metrics["sampling/undersized_prompt_group_rate"] = (
                        rollout_undersized_prompt_group_attempts / max(generated_prompt_attempts, 1)
                    )
                    metrics["sampling/undersized_prompt_group_trajectory_rate"] = (
                        rollout_undersized_prompt_group_trajectory_attempts
                        / max(generated_trajectory_attempts, 1)
                    )
                    raw_prompt_groups = len(raw_trajectory_metrics["group_rewards"])
                    kept_prompt_groups = len(kept_trajectory_metrics["group_rewards"])
                    metrics["filter/reward_group_keep_rate"] = kept_prompt_groups / max(raw_prompt_groups, 1)

                    with marked_timer("per_turn", timing_raw, "brown"):
                        try:
                            batch, per_turn_metrics = self._build_per_turn_grpo_training_batch(batch)
                        except NoTrainablePerTurnSamplesError as exc:
                            print(f"No trainable VisHarness per-turn samples: {exc}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                batch = None
                                if task_sampling_state is not None:
                                    previous_attempted = dict(task_sampling_state["attempted"])
                                    previous_topup_attempted = dict(
                                        task_sampling_state["topup_attempted"]
                                    )
                                    task_sampling_state = new_task_sampling_state()
                                    task_sampling_state["attempted"].update(previous_attempted)
                                    task_sampling_state["topup_attempted"].update(
                                        previous_topup_attempted
                                    )
                                self.gen_steps += 1
                                is_last_step = (
                                    False
                                    if task_sampling_state is not None
                                    else self.global_steps >= self.total_training_steps
                                )
                                continue
                            raise
                        metrics.update(per_turn_metrics)
                        metrics["train/update/trajectory_reward_mean"] = per_turn_metrics[
                            "visharness/per_turn/trajectory_reward_mean"
                        ]
                        metrics["train/update/trajectory_advantage_mean"] = per_turn_metrics[
                            "visharness/per_turn/trajectory_advantage_mean"
                        ]
                        metrics["train/update/trajectory_advantage_std"] = per_turn_metrics[
                            "visharness/per_turn/trajectory_advantage_std"
                        ]
                        metrics["train/update/trajectory_advantage_suppressed_trajectory_ratio"] = (
                            per_turn_metrics[
                                "visharness/per_turn/trajectory_advantage_suppressed_trajectory_ratio"
                            ]
                        )
                        metrics["train/update/group_trajectory_reward_std_mean"] = per_turn_metrics[
                            "visharness/per_turn/mean_group_trajectory_reward_std"
                        ]
                        metrics["train/update/assigned_trajectory_advantage_mean"] = per_turn_metrics[
                            "visharness/per_turn/assigned_trajectory_advantage_mean"
                        ]
                        metrics["train/update/base_advantage_after_clamp_mean"] = per_turn_metrics[
                            "visharness/per_turn/base_advantage_after_clamp_mean"
                        ]
                        metrics["train/update/final_turn_advantage_mean"] = per_turn_metrics[
                            "visharness/per_turn/final_turn_advantage_mean"
                        ]
                        metrics["train/update/legacy_additive_turn_advantage_mean"] = per_turn_metrics[
                            "visharness/per_turn/legacy_additive_turn_advantage_mean"
                        ]
                        metrics["train/update/local_advantage_cost_mean"] = per_turn_metrics[
                            "visharness/per_turn/local_advantage_cost_mean"
                        ]
                        useful_trajectory_weight_metrics = {
                            "trajectory_equal_weight_enabled",
                            "local_cost_per_event_floor_enabled",
                            "trainable_trajectory_count",
                            "trainable_turns_per_trajectory_mean",
                            "trainable_turns_per_trajectory_min",
                            "trainable_turns_per_trajectory_max",
                            "trajectory_loss_weight_mean",
                            "trajectory_loss_weight_min",
                            "trajectory_loss_weight_max",
                            "trajectory_loss_weight_std",
                            "trajectory_loss_mass_mean",
                            "trajectory_loss_mass_std",
                            "ppo_turn_advantage_mean",
                            "ppo_turn_advantage_abs_mean",
                            "effective_policy_advantage_mean",
                            "effective_policy_advantage_abs_mean",
                            "effective_policy_advantage_std",
                            "effective_policy_advantage_min",
                            "effective_policy_advantage_max",
                            "effective_positive_advantage_count",
                            "effective_negative_advantage_count",
                            "effective_zero_advantage_count",
                            "effective_positive_advantage_rate",
                            "effective_negative_advantage_rate",
                            "effective_zero_advantage_rate",
                            "effective_positive_advantage_mass",
                            "effective_negative_advantage_mass",
                            "effective_net_advantage_mass",
                            "effective_positive_advantage_mass_share",
                            "effective_negative_advantage_mass_share",
                            "effective_local_cost_mean",
                            "local_floor_applied_rate",
                            "hard_error_local_floor_applied_rate",
                            "trajectory_weighted_assigned_advantage_mean",
                            "trajectory_weighted_final_advantage_mean",
                            "trajectory_weighted_legacy_final_advantage_mean",
                            "trainable_turn_count_advantage_covariance",
                        }
                        for metric_name, metric_value in per_turn_metrics.items():
                            short_name = metric_name.removeprefix("visharness/per_turn/")
                            if short_name in useful_trajectory_weight_metrics:
                                metrics[f"train/update/{short_name}"] = metric_value
                        metrics["train/update/hard_error_turn_rate"] = per_turn_metrics[
                            "visharness/per_turn/hard_error_turn_rate"
                        ]
                        metrics["train/update/positive_trajectory_advantage_clamped_rate"] = per_turn_metrics[
                            "visharness/per_turn/positive_trajectory_advantage_clamped_rate"
                        ]
                        metrics["train/update/output_format_error_rate"] = per_turn_metrics[
                            "visharness/per_turn/output_format_error_rate"
                        ]
                        metrics["train/update/raw_output_format_failure_rate"] = per_turn_metrics[
                            "visharness/per_turn/raw_output_format_failure_rate"
                        ]
                        metrics["train/update/tool_args_error_rate"] = per_turn_metrics[
                            "visharness/per_turn/tool_args_error_rate"
                        ]
                        metrics["train/update/raw_tool_args_failure_rate"] = per_turn_metrics[
                            "visharness/per_turn/raw_tool_args_failure_rate"
                        ]
                        metrics["train/update/truncation_error_rate"] = per_turn_metrics[
                            "visharness/per_turn/truncation_error_rate"
                        ]
                        metrics["train/update/soft_overlong_rate"] = per_turn_metrics[
                            "visharness/per_turn/soft_overlong_rate"
                        ]
                        metrics["train/update/prompt_overlong_turn_ratio"] = per_turn_metrics[
                            "visharness/per_turn/prompt_overlong_drop_ratio"
                        ]
                        self._add_actor_update_sample_metrics(metrics, batch)

                    self.checkpoint_manager.sleep_replicas()

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    self._maybe_log_train_generations(batch, logger)

                    if not self.config.algorithm.use_kl_in_reward:
                        batch = self.compute_kl_related_metrics(batch, metrics, timing_raw)

                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                        batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                        metrics.update(is_metrics)

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self._update_actor(batch)

                        if (
                            task_sampling_state is not None
                            and task_sampling_state["epoch_completed_after_update"]
                        ):
                            # Commit the data state only after the actor update.
                            # A checkpoint now always points at the exact next
                            # prompt that should be rolled out.
                            self._task_source_pool.start_next_epoch()
                            train_epoch += 1
                            self._visharness_train_epoch = int(train_epoch)
                            self._visharness_epoch_start_global_step = int(
                                self.global_steps
                            )
                            if is_last_step:
                                self._visharness_training_completed = True
                                self._visharness_completion_reason = (
                                    "all_sources_first_pass_completed"
                                )
                        elif is_last_step:
                            self._visharness_training_completed = True
                            self._visharness_completion_reason = "static_step_limit"

                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, "green"):
                                self._save_checkpoint()
                            last_saved_step = int(self.global_steps)

                        with marked_timer("update_weights", timing_raw, "red"):
                            self.checkpoint_manager.update_weights()
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        last_validation_step = int(self.global_steps)
                        self._visharness_last_validation_step = int(self.global_steps)
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw.get("step", 0)
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)

                metrics["sampling/rollout_batches"] = float(num_gen_batches)
                metrics["sampling/attempted_prompt_groups"] = float(generated_prompt_attempts)
                metrics["sampling/attempted_trajectories"] = float(generated_trajectory_attempts)
                if task_sampling_state is not None:
                    eligible_counts_by_source = Counter(
                        raw_trajectory_metrics["group_sources"].values()
                    )
                    metrics.update(
                        self._summarize_task_sampling_metrics(
                            task_sampling_state,
                            eligible_counts_by_source=eligible_counts_by_source,
                        )
                    )
                prompt_examples_consumed_this_fit += generated_prompt_attempts
                batch = None
                num_gen_batches = 0
                generated_prompt_attempts = 0
                generated_trajectory_attempts = 0
                rollout_invalid_trajectory_attempts = 0
                rollout_prompt_overlong_attempts = 0
                rollout_tool_oom_invalid_attempts = 0
                rollout_undersized_prompt_group_attempts = 0
                rollout_undersized_prompt_group_trajectory_attempts = 0
                raw_trajectory_metrics = self._new_trajectory_metrics_accumulator()
                task_sampling_state = None

                refresh_update_progress(
                    self.global_steps,
                    prompt_examples_consumed_this_fit,
                )
                metrics["sampling/estimated_remaining_actor_updates"] = float(
                    self._visharness_estimated_remaining_updates
                )
                metrics["sampling/estimated_total_actor_updates"] = float(
                    self._visharness_estimated_total_updates
                )
                logger.log(data=self._filter_logged_metrics(metrics), step=self.global_steps)
                last_completed_step = int(self.global_steps)
                if (
                    last_saved_step == int(self.global_steps)
                    and last_validation_step == int(self.global_steps)
                ):
                    # Checkpointing precedes validation. Publish the completed
                    # validation marker only after its metrics were logged.
                    self._save_visharness_resume_state()

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                self.global_steps += 1
                self.gen_steps += 1
                task_sampling_state = new_task_sampling_state()

        if self._task_sampling_enabled():
            raise RuntimeError(
                "Coverage-based task-balanced training left its main loop without "
                "finalizing the epoch-ending actor update. This indicates an "
                "internal state transition error."
            )
        self._visharness_train_epoch = int(train_epoch)
        # Prompts generated for the final incomplete update were consumed from
        # the dataset even though no actor update was formed. Once the finite
        # stream is exhausted, any unusable source-buffer suffix is exhausted
        # as well, so final progress must report no remaining input prompts.
        final_consumed_prompt_examples = (
            prompt_examples_consumed_this_fit + generated_prompt_attempts
        )
        final_consumed_prompt_examples = max(
            final_consumed_prompt_examples,
            total_raw_prompt_examples - prompt_examples_consumed_before_fit,
        )
        self._finalize_naturally_exhausted_training(
            last_completed_step=last_completed_step,
            last_validation_step=last_validation_step,
            last_saved_step=last_saved_step,
            logger=logger,
            progress_bar=progress_bar,
            refresh_update_progress=refresh_update_progress,
            consumed_prompt_examples=final_consumed_prompt_examples,
        )

    def _update_actor(self, batch: DataProto) -> DataProto:
        per_turn_batch = batch

        num_real_turns = int(per_turn_batch.meta_info.get("visharness_num_turns", len(per_turn_batch)))
        num_trajectories = int(per_turn_batch.meta_info.get("visharness_num_trajectories", len(batch)))
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")
        num_mini_batches = self._get_per_turn_num_mini_batches()
        if num_real_turns < num_mini_batches:
            raise ValueError(
                "Per-turn GRPO has fewer real turn samples than fixed optimizer "
                f"mini-batches: {num_real_turns=} {num_mini_batches=}"
            )
        planned_batch_size, expected_pad_size, planned_mini_batch_size = (
            plan_fixed_per_turn_mini_batches(
                len(per_turn_batch),
                num_mini_batches=num_mini_batches,
                dp_size=dp_size,
            )
        )
        per_turn_batch, pad_size = pad_per_turn_batch_to_divisor(
            per_turn_batch,
            dp_size * num_mini_batches,
        )
        if len(per_turn_batch) != planned_batch_size or pad_size != expected_pad_size:
            raise RuntimeError(
                "Per-turn padding disagrees with the fixed mini-batch plan: "
                f"planned_size={planned_batch_size}, actual_size={len(per_turn_batch)}, "
                f"planned_padding={expected_pad_size}, actual_padding={pad_size}"
            )

        old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(per_turn_batch)
        old_log_prob.batch.pop("entropys")
        per_turn_batch = per_turn_batch.union(old_log_prob)
        if self.use_reference_policy:
            per_turn_batch = per_turn_batch.union(self._compute_ref_log_prob(per_turn_batch))
        reference_kl_metrics = self._reference_kl_metrics_by_source(per_turn_batch)

        rollout_config = self.config.actor_rollout_ref.rollout
        per_turn_batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        per_turn_batch.meta_info["temperature"] = rollout_config.temperature

        calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
            self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        )
        mini_batch_size = planned_mini_batch_size

        if num_mini_batches is None or len(per_turn_batch) != mini_batch_size * num_mini_batches:
            raise RuntimeError(
                "Invalid per-turn optimizer mini-batch plan: "
                f"batch_size={len(per_turn_batch)}, mini_batch_size={mini_batch_size}, "
                f"num_mini_batches={num_mini_batches}"
            )
        batch_td = left_right_2_no_padding(per_turn_batch.to_tensordict())
        annotate_effective_global_batch_size(
            batch_td,
            global_mini_batch_size=mini_batch_size,
            dp_size=dp_size,
        )
        trajectory_weight_correction_range = None
        if self._per_turn_loss_weight_correction_enabled():
            trajectory_weight_corrections = rescale_trajectory_loss_weights_for_minibatches(
                batch_td,
                num_mini_batches=num_mini_batches,
                num_real_sequences=num_real_turns,
            )
            valid_rows = batch_td["response_mask"].reshape(len(batch_td), -1).to(torch.bool).any(dim=-1)
            real_corrections = trajectory_weight_corrections[valid_rows]
            trajectory_weight_correction_range = (
                float(real_corrections.min().item()),
                float(real_corrections.max().item()),
            )
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=calculate_entropy,
            distillation_use_topk=False,
            global_batch_size=mini_batch_size,
            mini_batch_size=mini_batch_size,
            epochs=self.config.actor_rollout_ref.actor.ppo_epochs,
            seed=self.config.actor_rollout_ref.actor.data_loader_seed,
            dataloader_kwargs={"shuffle": self.config.actor_rollout_ref.actor.shuffle},
            compute_loss=True,
        )
        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        metrics = tu.get(actor_output, "metrics")
        metrics = rename_dict(metrics, "actor/")
        metrics["perf/mfu/actor"] = metrics.pop("actor/mfu")
        total_padding = len(per_turn_batch) - num_real_turns
        actual_optimizer_steps = num_mini_batches * int(self.config.actor_rollout_ref.actor.ppo_epochs)
        metrics["visharness/per_turn_samples"] = [num_real_turns]
        metrics["visharness/per_turn_padding"] = [pad_size]
        metrics["visharness/per_turn_total_padding"] = [total_padding]
        metrics["visharness/turns_per_trajectory"] = [num_real_turns / max(num_trajectories, 1)]
        metrics["visharness/per_turn_mini_batch_size"] = [mini_batch_size]
        metrics["visharness/per_turn_num_mini_batches"] = [num_mini_batches]
        metrics["visharness/per_turn_optimizer_steps"] = [actual_optimizer_steps]
        metrics["visharness/per_turn_real_samples_per_update"] = [num_real_turns / num_mini_batches]
        metrics["train/update/optimizer_minibatches"] = [num_mini_batches]
        metrics["train/update/global_mini_batch_size"] = [mini_batch_size]
        metrics["train/update/actual_optimizer_steps"] = [actual_optimizer_steps]
        metrics["train/update/per_turn_padding_ratio"] = [total_padding / max(len(per_turn_batch), 1)]
        if trajectory_weight_correction_range is not None:
            metrics["train/update/trajectory_minibatch_weight_correction_min"] = [
                trajectory_weight_correction_range[0]
            ]
            metrics["train/update/trajectory_minibatch_weight_correction_max"] = [
                trajectory_weight_correction_range[1]
            ]
        metrics.update({key: [value] for key, value in reference_kl_metrics.items()})
        metrics["perf/mfu/actor_infer_per_turn"] = (
            old_log_prob_mfu if isinstance(old_log_prob_mfu, list) else [old_log_prob_mfu]
        )

        return DataProto.from_single_dict(data={}, meta_info={"metrics": metrics})
