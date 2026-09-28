from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto

from visharness.rl_trainer.visharness_trainer import VisHarnessTrainer


class _FakeStreamingManager:
    def generate_training_prompt_groups(
        self,
        jobs,
        *,
        on_complete,
        total_slots,
        trajectories_per_group,
        progress_interval,
    ):
        del progress_interval
        pending = list(jobs)
        submitted = len(pending)
        completed_slots = 0
        while pending:
            slot_index, generation_batch, context = pending.pop(0)
            assert len(generation_batch) == trajectories_per_group
            prompt_id = int(generation_batch.batch["prompt_id"][0].item())
            rewards = {
                0: [0.0, 0.0],  # filtered, so slot 0 must be refilled
                1: [0.0, 1.0],
                2: [0.0, 2.0],
            }[prompt_id]
            output = DataProto.from_dict(
                tensors={
                    "rm_scores": torch.tensor(rewards, dtype=torch.float32).view(
                        trajectories_per_group,
                        1,
                    )
                }
            )
            replacement = on_complete(slot_index, context, output)
            if replacement is None:
                completed_slots += 1
            else:
                pending.append(replacement)
                submitted += 1

        return {
            "completed_slots": float(completed_slots),
            "submitted_groups": float(submitted),
            "completed_groups": float(submitted),
            "peak_in_flight_groups": float(total_slots),
            "worker_count": float(total_slots),
            "wall_seconds": 1.0,
            "timing": {},
        }


def _prompt_batch(prompt_ids, sources=None):
    prompt_ids = list(prompt_ids)
    if sources is None:
        sources = ["task"] * len(prompt_ids)
    sources = list(sources)
    return DataProto.from_dict(
        tensors={
            "prompt_id": torch.tensor(prompt_ids, dtype=torch.long).view(-1, 1)
        },
        non_tensors={
            "uid": np.asarray([f"prompt-{value}" for value in prompt_ids], dtype=object),
            "data_source": np.asarray(sources, dtype=object),
        },
    )


def test_streaming_trainer_refills_only_the_failed_prompt_slot():
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer._task_sampling_plan = None
    trainer.async_rollout_manager = _FakeStreamingManager()
    trainer.use_rm = False
    trainer.gen_steps = 1
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": 2},
            "actor_rollout_ref": {"rollout": {"n": 2}},
            "algorithm": {
                "adv_estimator": "grpo",
                "filter_groups": {
                    "enable": True,
                    "metric": "trajectory_reward",
                    "max_num_gen_batches": 0,
                },
            },
            "visharness": {
                "filter_groups": {
                    "min_reward_std": 0.1,
                    "min_trajectories_after_rollout_filter": 2,
                },
            },
        }
    )

    trainer._get_gen_batch = lambda batch: batch
    trainer._drop_invalid_trajectories = lambda batch: (batch, 0, {})
    trainer._maybe_drop_undersized_prompt_groups = lambda batch: (batch, 0, 0)
    trainer._accumulate_trajectory_metrics = lambda _accumulator, _batch: None
    trainer._prepare_filter_group_metric = lambda batch, _metric_name: (
        batch.non_tensor_batch.__setitem__(
            "trajectory_reward",
            batch.batch["rm_scores"].sum(dim=-1).cpu().numpy(),
        )
    )

    raw_batches = iter([_prompt_batch([0, 1]), _prompt_batch([2])])
    result = trainer._collect_streaming_filtered_rollouts(
        task_sampling_state=None,
        next_raw_prompt_batch=lambda: next(raw_batches, None),
        metrics={},
        timing_raw=defaultdict(float),
        raw_trajectory_metrics={},
    )

    selected_prompt_ids = result["batch"].batch["prompt_id"].flatten().tolist()
    assert selected_prompt_ids == [2, 2, 1, 1]
    assert result["generated_prompt_attempts"] == 3
    assert result["generated_trajectory_attempts"] == 6
    assert result["num_gen_batches"] == 2
    assert result["num_prompt_in_batch"] == 2


def test_streaming_task_quota_refills_from_the_failed_slots_source():
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer._task_sampling_plan = SimpleNamespace(sources=("source-a", "source-b"))
    trainer._task_source_pool = SimpleNamespace(
        progress_by_source=lambda: {
            "source-a": {"first_pass_coverage": 0.5},
            "source-b": {"first_pass_coverage": 1.0},
        },
        all_sources_first_pass_completed=False,
    )
    trainer.async_rollout_manager = _FakeStreamingManager()
    trainer.use_rm = False
    trainer.gen_steps = 1
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": 2},
            "actor_rollout_ref": {"rollout": {"n": 2}},
            "algorithm": {
                "adv_estimator": "grpo",
                "filter_groups": {
                    "enable": True,
                    "metric": "trajectory_reward",
                    "max_num_gen_batches": 0,
                },
            },
            "visharness": {
                "filter_groups": {
                    "min_reward_std": 0.1,
                    "min_trajectories_after_rollout_filter": 2,
                },
            },
        }
    )

    trainer._get_gen_batch = lambda batch: batch
    trainer._drop_invalid_trajectories = lambda batch: (batch, 0, {})
    trainer._maybe_drop_undersized_prompt_groups = lambda batch: (batch, 0, 0)
    trainer._accumulate_trajectory_metrics = lambda _accumulator, _batch: None
    trainer._record_task_sampling_filter_progress = lambda _state: None
    trainer._prepare_filter_group_metric = lambda batch, _metric_name: (
        batch.non_tensor_batch.__setitem__(
            "trajectory_reward",
            batch.batch["rm_scores"].sum(dim=-1).cpu().numpy(),
        )
    )

    draws = iter(
        [
            (
                _prompt_batch([0, 1], ["source-a", "source-b"]),
                {"source-a": 1, "source-b": 1},
                {},
            ),
            (
                _prompt_batch([2], ["source-a"]),
                {"source-a": 1},
                {"source-a": 1},
            ),
        ]
    )

    def take_by_source(requested_counts):
        batch, selected, cycles = next(draws)
        assert selected == {
            source: count
            for source, count in requested_counts.items()
            if count
        }
        return batch, selected, cycles

    trainer._take_cycling_raw_prompts_by_source = take_by_source
    sampling_state = {
        "original_target": {"source-a": 1, "source-b": 1},
        "kept": defaultdict(int),
        "attempted": defaultdict(int),
        "topup_attempted": defaultdict(int),
        "attempted_since_progress": defaultdict(int),
        "no_progress_cycles": defaultdict(int),
        "kept_at_progress_check": defaultdict(int),
        "completed_cycles_during_update": defaultdict(int),
        "source_progress_snapshot": {},
        "epoch_completed_after_update": False,
    }

    result = trainer._collect_streaming_filtered_rollouts(
        task_sampling_state=sampling_state,
        next_raw_prompt_batch=lambda: None,
        metrics={},
        timing_raw=defaultdict(float),
        raw_trajectory_metrics={},
    )

    assert result["batch"].non_tensor_batch["data_source"].tolist() == [
        "source-a",
        "source-a",
        "source-b",
        "source-b",
    ]
    assert dict(sampling_state["kept"]) == {"source-a": 1, "source-b": 1}
    assert dict(sampling_state["attempted"]) == {"source-a": 2, "source-b": 1}
    assert dict(sampling_state["topup_attempted"]) == {"source-a": 1}
    assert dict(sampling_state["completed_cycles_during_update"]) == {
        "source-a": 1
    }
