from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from visharness.data.task_balanced_sampler import (
    CyclingTaskSourcePool,
    TaskSamplingPlan,
    TemperatureBalancedSampler,
    balanced_source_sequence,
    build_task_sampling_plan,
)
from visharness.rl_trainer.visharness_trainer import VisHarnessTrainer


class _FakeDataFrame:
    def __init__(self, sources):
        self._sources = list(sources)
        self.column_names = ["data_source"]

    def __getitem__(self, key):
        if isinstance(key, int):
            return {"data_source": self._sources[key]}
        if key != "data_source":
            raise KeyError(key)
        return list(self._sources)

    def __len__(self):
        return len(self._sources)


class _FakeDataset:
    def __init__(self, sources):
        self.sources = list(sources)
        self.dataframe = _FakeDataFrame(sources)

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, index):
        return {
            "prompt_id": torch.tensor(index, dtype=torch.long),
            "uid": f"dataset-{index}",
            "data_source": self.sources[index],
        }


def _small_plan() -> TaskSamplingPlan:
    return TaskSamplingPlan(
        indices_by_source={
            "visharness/gres": (0, 1, 2),
            "visharness/reasonseg": (3, 4),
            "visharness/rec8k": (5, 6, 7, 8),
        },
        source_counts={
            "visharness/gres": 3,
            "visharness/reasonseg": 2,
            "visharness/rec8k": 4,
        },
        probabilities={
            "visharness/gres": 3 / 8,
            "visharness/reasonseg": 1 / 8,
            "visharness/rec8k": 4 / 8,
        },
        num_samples=16,
        batch_size=8,
        alpha=0.5,
        epoch_size_mode="all_sources_once",
    )


def _task_sampling_trainer(
    max_no_progress_cycles: int = 3,
    *,
    reward_group_filter_enabled: bool = True,
) -> VisHarnessTrainer:
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer._task_sampling_plan = _small_plan()
    trainer._task_source_pool = CyclingTaskSourcePool(trainer._task_sampling_plan, seed=42)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "filter_groups": {"enable": reward_group_filter_enabled}
            },
            "data": {"seed": 42},
            "visharness": {
                "task_sampling": {
                    "max_no_progress_cycles": max_no_progress_cycles,
                },
                "filter_groups": {"min_reward_std": 0.3},
            },
        }
    )
    return trainer


def test_temperature_plan_matches_expected_training_distribution():
    sources = (
        ["visharness/rec8k"] * 2660
        + ["visharness/gres"] * 1600
        + ["visharness/reasonseg"] * 239
    )
    plan = build_task_sampling_plan(
        _FakeDataset(sources),
        alpha=0.5,
        batch_size=8,
    )

    assert plan.num_samples == 5520
    assert plan.probabilities["visharness/rec8k"] == pytest.approx(0.48185432896)
    assert plan.probabilities["visharness/gres"] == pytest.approx(0.37371018833)
    assert plan.probabilities["visharness/reasonseg"] == pytest.approx(0.14443548270)
    assert plan.quota_for_update(0) == {
        "visharness/gres": 3,
        "visharness/reasonseg": 1,
        "visharness/rec8k": 4,
    }
    assert plan.quota_for_update(3) == {
        "visharness/gres": 3,
        "visharness/reasonseg": 2,
        "visharness/rec8k": 3,
    }
    epoch_counts = Counter(balanced_source_sequence(plan.probabilities, plan.num_samples))
    assert epoch_counts == {
        "visharness/rec8k": 2660,
        "visharness/gres": 2063,
        "visharness/reasonseg": 797,
    }


def test_temperature_sampler_resume_reproduces_exact_remaining_indices():
    sampler = TemperatureBalancedSampler(_small_plan(), seed=42)
    iterator = iter(sampler)
    _ = [next(iterator) for _ in range(7)]
    state = iterator.state_dict()
    expected = [next(iterator) for _ in range(9)]

    restored = iter(sampler)
    restored.load_state_dict(state)
    actual = [next(restored) for _ in range(9)]

    assert actual == expected


def test_temperature_sampler_preserves_each_update_quota_after_batch_shuffle():
    plan = _small_plan()
    sampled_indices = list(TemperatureBalancedSampler(plan, seed=42))
    source_by_index = {
        index: source
        for source, indices in plan.indices_by_source.items()
        for index in indices
    }

    for update_index, start in enumerate(range(0, len(sampled_indices), plan.batch_size)):
        batch_indices = sampled_indices[start : start + plan.batch_size]
        actual = Counter(source_by_index[index] for index in batch_indices)
        assert actual == Counter(plan.quota_for_update(update_index))


def test_cycling_source_pool_returns_exact_quota_and_cycles_independently():
    plan = _small_plan()
    pool = CyclingTaskSourcePool(plan, seed=42)

    first_draws, first_cycles = pool.take(
        {
            "visharness/gres": 3,
            "visharness/reasonseg": 1,
            "visharness/rec8k": 4,
        }
    )
    assert Counter(draw.source for draw in first_draws) == {
        "visharness/gres": 3,
        "visharness/reasonseg": 1,
        "visharness/rec8k": 4,
    }
    assert first_cycles == {
        "visharness/gres": 1,
        "visharness/rec8k": 1,
    }
    assert len({draw.attempt_uid for draw in first_draws}) == len(first_draws)
    assert not pool.all_sources_first_pass_completed

    second_draws, second_cycles = pool.take({"visharness/reasonseg": 2})
    assert len(second_draws) == 2
    assert second_cycles == {"visharness/reasonseg": 1}
    assert pool.all_sources_first_pass_completed
    assert pool.source_progress("visharness/reasonseg")["cycles_completed"] == 1
    assert pool.source_progress("visharness/reasonseg")["draws_in_epoch"] == 3


def test_cycling_source_pool_first_cycle_is_without_replacement():
    plan = _small_plan()
    pool = CyclingTaskSourcePool(plan, seed=7)

    draws, completed_cycles = pool.take({"visharness/rec8k": 5})

    first_cycle_indices = [draw.dataset_index for draw in draws[:4]]
    assert set(first_cycle_indices) == set(plan.indices_by_source["visharness/rec8k"])
    assert len(first_cycle_indices) == len(set(first_cycle_indices))
    assert completed_cycles == {"visharness/rec8k": 1}
    assert draws[-1].source_cycle == 1


def test_cycling_source_pool_resume_reproduces_exact_future_draws():
    plan = _small_plan()
    pool = CyclingTaskSourcePool(plan, seed=42)
    pool.take({"visharness/gres": 4, "visharness/reasonseg": 3})
    state = pool.state_dict()

    expected_draws, expected_cycles = pool.take(
        {"visharness/gres": 5, "visharness/reasonseg": 4, "visharness/rec8k": 6}
    )
    restored = CyclingTaskSourcePool(plan, seed=42)
    restored.load_state_dict(state)
    actual_draws, actual_cycles = restored.take(
        {"visharness/gres": 5, "visharness/reasonseg": 4, "visharness/rec8k": 6}
    )

    assert actual_draws == expected_draws
    assert actual_cycles == expected_cycles
    assert restored.state_dict()["epoch_index"] == pool.state_dict()["epoch_index"]


def test_cycling_source_pool_rng_is_independent_between_sources():
    plan = _small_plan()
    baseline = CyclingTaskSourcePool(plan, seed=42)
    interleaved = CyclingTaskSourcePool(plan, seed=42)

    expected, _ = baseline.take({"visharness/rec8k": 8})
    interleaved.take({"visharness/gres": 20, "visharness/reasonseg": 20})
    actual, _ = interleaved.take({"visharness/rec8k": 8})

    assert [draw.dataset_index for draw in actual] == [
        draw.dataset_index for draw in expected
    ]


def test_cycling_source_pool_next_epoch_resets_coverage_but_keeps_total_draws():
    plan = _small_plan()
    pool = CyclingTaskSourcePool(plan, seed=42)
    pool.take({source: plan.source_counts[source] for source in plan.sources})
    previous_totals = {
        source: pool.source_progress(source)["total_draws"]
        for source in plan.sources
    }

    pool.start_next_epoch()

    assert pool.epoch_index == 1
    assert not pool.all_sources_first_pass_completed
    for source in plan.sources:
        progress = pool.source_progress(source)
        assert progress["first_pass_coverage"] == 0
        assert progress["cycles_completed"] == 0
        assert progress["draws_in_epoch"] == 0
        assert progress["total_draws"] == previous_totals[source]


def test_cycling_source_pool_rejects_corrupt_resume_permutation():
    pool = CyclingTaskSourcePool(_small_plan(), seed=42)
    state = pool.state_dict()
    state["sources"]["visharness/gres"]["permutation"] = [0, 0, 2]

    with pytest.raises(ValueError, match="exact permutation"):
        CyclingTaskSourcePool(_small_plan(), seed=42).load_state_dict(state)


def test_stateful_dataloader_restores_temperature_sampler_exactly():
    loader = StatefulDataLoader(
        list(range(9)),
        batch_size=4,
        sampler=TemperatureBalancedSampler(_small_plan(), seed=42),
        num_workers=0,
        drop_last=True,
    )
    iterator = iter(loader)
    next(iterator)
    state = loader.state_dict()
    expected = next(iterator).tolist()

    restored_loader = StatefulDataLoader(
        list(range(9)),
        batch_size=4,
        sampler=TemperatureBalancedSampler(_small_plan(), seed=42),
        num_workers=0,
        drop_last=True,
    )
    restored_loader.load_state_dict(state)

    assert next(iter(restored_loader)).tolist() == expected


def test_trainer_fetches_exact_source_quota_and_assigns_unique_attempt_uids():
    trainer = _task_sampling_trainer()
    trainer.train_dataset = _FakeDataset(
        ["visharness/gres"] * 3
        + ["visharness/reasonseg"] * 2
        + ["visharness/rec8k"] * 4
    )
    trainer.config.actor_rollout_ref = OmegaConf.create(
        {"rollout": {"temperature": 0.8}}
    )

    def collate(samples):
        return {
            "prompt_id": torch.stack([sample["prompt_id"] for sample in samples]),
            "uid": np.asarray([sample["uid"] for sample in samples], dtype=object),
            "data_source": np.asarray(
                [sample["data_source"] for sample in samples], dtype=object
            ),
        }

    trainer._visharness_collate_fn = collate
    batch, counts, completed_cycles = trainer._take_cycling_raw_prompts_by_source(
        {
            "visharness/gres": 3,
            "visharness/reasonseg": 3,
            "visharness/rec8k": 2,
        }
    )

    assert counts == {
        "visharness/gres": 3,
        "visharness/reasonseg": 3,
        "visharness/rec8k": 2,
    }
    assert completed_cycles == {
        "visharness/gres": 1,
        "visharness/reasonseg": 1,
    }
    assert len(batch) == 8
    assert len(set(batch.non_tensor_batch["uid"].tolist())) == 8
    assert all(
        str(uid).startswith("task-prompt/e0/")
        for uid in batch.non_tensor_batch["uid"]
    )
    assert all(
        str(uid).startswith("dataset-")
        for uid in batch.non_tensor_batch["dataset_uid"]
    )


def test_no_progress_guard_raises_after_configured_complete_cycles():
    trainer = _task_sampling_trainer(max_no_progress_cycles=3)
    state = {
        "original_target": {"visharness/reasonseg": 1},
        "kept": {},
        "attempted": defaultdict(int),
        "attempted_since_progress": defaultdict(int),
        "no_progress_cycles": defaultdict(int),
        "kept_at_progress_check": defaultdict(int),
    }

    for _ in range(2):
        state["attempted"]["visharness/reasonseg"] += 2
        state["attempted_since_progress"]["visharness/reasonseg"] += 2
        trainer._record_task_sampling_filter_progress(state)

    state["attempted"]["visharness/reasonseg"] += 2
    state["attempted_since_progress"]["visharness/reasonseg"] += 2
    with pytest.raises(RuntimeError, match="made no selectable-group progress"):
        trainer._record_task_sampling_filter_progress(state)


def test_no_progress_guard_supports_disabled_reward_group_filter():
    trainer = _task_sampling_trainer(
        max_no_progress_cycles=1,
        reward_group_filter_enabled=False,
    )
    state = {
        "original_target": {"visharness/reasonseg": 1},
        "kept": {},
        "attempted": {"visharness/reasonseg": 2},
        "attempted_since_progress": defaultdict(
            int, {"visharness/reasonseg": 2}
        ),
        "no_progress_cycles": defaultdict(int),
        "kept_at_progress_check": defaultdict(int),
    }

    with pytest.raises(
        RuntimeError,
        match="reward_std_filter=disabled, min_surviving_trajectories=4",
    ):
        trainer._record_task_sampling_filter_progress(state)


def test_no_progress_guard_resets_after_a_prompt_is_kept():
    trainer = _task_sampling_trainer(max_no_progress_cycles=3)
    state = {
        "original_target": {"visharness/reasonseg": 1},
        "kept": {},
        "attempted": {"visharness/reasonseg": 4},
        "attempted_since_progress": defaultdict(
            int, {"visharness/reasonseg": 4}
        ),
        "no_progress_cycles": defaultdict(
            int, {"visharness/reasonseg": 2}
        ),
        "kept_at_progress_check": defaultdict(int),
    }
    state["kept"]["visharness/reasonseg"] = 1

    trainer._record_task_sampling_filter_progress(state)

    assert state["no_progress_cycles"]["visharness/reasonseg"] == 0
    assert state["attempted_since_progress"]["visharness/reasonseg"] == 0


def test_task_sampling_metrics_use_explicit_stages_and_correct_denominators():
    trainer = _task_sampling_trainer()
    state = {
        "original_target": {
            "visharness/rec8k": 4,
            "visharness/gres": 3,
            "visharness/reasonseg": 1,
        },
        "attempted": {
            "visharness/rec8k": 4,
            "visharness/gres": 5,
            "visharness/reasonseg": 2,
        },
        "topup_attempted": {
            "visharness/rec8k": 0,
            "visharness/gres": 2,
            "visharness/reasonseg": 1,
        },
        "kept": {
            "visharness/rec8k": 4,
            "visharness/gres": 2,
            "visharness/reasonseg": 1,
        },
        "source_progress_snapshot": trainer._task_source_pool.progress_by_source(),
        "no_progress_cycles": {},
        "completed_cycles_during_update": {},
        "task_epoch": 0,
        "epoch_completed_after_update": False,
    }

    metrics = trainer._summarize_task_sampling_metrics(
        state,
        eligible_counts_by_source={
            "visharness/rec8k": 4,
            "visharness/gres": 4,
            "visharness/reasonseg": 1,
        },
    )

    assert metrics["sampling/by_data_source/gres/target_prompt_groups"] == 3
    assert metrics["sampling/by_data_source/gres/attempted_prompt_groups"] == 5
    assert metrics["sampling/by_data_source/gres/topup_prompt_groups"] == 2
    assert metrics["sampling/by_data_source/gres/selected_prompt_groups"] == 2
    assert metrics["sampling/by_data_source/gres/post_filter_quota_shortfall"] == 1
    assert metrics["sampling/by_data_source/gres/selected_per_attempt_rate"] == pytest.approx(
        0.4
    )
    assert metrics["filter/by_data_source/gres/reward_group_keep_rate"] == pytest.approx(
        0.5
    )
    assert metrics["sampling/min_source_first_pass_coverage"] == 0


def test_cycling_source_pool_resume_state_round_trip(tmp_path):
    trainer = _task_sampling_trainer()
    trainer.global_steps = 5
    trainer._visharness_train_epoch = 0
    trainer._visharness_epoch_start_global_step = 0
    trainer.config.trainer = OmegaConf.create(
        {"default_local_dir": str(tmp_path / "checkpoints")}
    )
    trainer._task_source_pool.take(
        {"visharness/gres": 2, "visharness/reasonseg": 1}
    )
    expected_pool_state = trainer._task_source_pool.state_dict()

    trainer._save_visharness_resume_state()

    state = torch.load(
        tmp_path / "checkpoints" / "global_step_5" / "visharness_resume_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    restored_pool = CyclingTaskSourcePool(_small_plan(), seed=42)
    restored_pool.load_state_dict(state["task_source_pool_state"])
    assert state["version"] == 5
    assert state["task_sampling"] == {
        "enabled": True,
        "alpha": 0.5,
        "batch_size": 8,
        "epoch_semantics": "all_sources_first_pass_with_source_cycling",
        "source_counts": {
            "visharness/gres": 3,
            "visharness/reasonseg": 2,
            "visharness/rec8k": 4,
        },
        "seed": 42,
        "max_no_progress_cycles": 3,
    }
    assert state["dataloader_state_signature"] is None
    assert restored_pool.state_dict()["epoch_index"] == expected_pool_state["epoch_index"]
    assert restored_pool.progress_by_source() == trainer._task_source_pool.progress_by_source()


def test_saved_exhausted_epoch_is_not_reopened_when_source_buffers_remain():
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.global_steps = 12
    trainer.train_dataloader = [None] * 10
    trainer.config = OmegaConf.create({"trainer": {"total_epochs": 1}})

    train_epoch = trainer._infer_current_train_epoch(
        resume_state={"train_epoch": 1},
        restored_prompt_examples_consumed=16,
        prompt_bsz_for_progress=8,
    )

    assert train_epoch == 1


def test_resume_rejects_changed_task_sampling_plan():
    trainer = _task_sampling_trainer()
    changed_signature = trainer._task_sampling_resume_signature()
    changed_signature["alpha"] = 0.7

    with pytest.raises(ValueError, match="does not match"):
        trainer._validate_task_sampling_resume_state(
            {"version": 5, "task_sampling": changed_signature}
        )


def test_resume_rejects_legacy_shared_dataloader_task_checkpoint():
    trainer = _task_sampling_trainer()

    with pytest.raises(ValueError, match="independent cycling task-source pool"):
        trainer._validate_task_sampling_resume_state(
            {"version": 4, "task_sampling": trainer._task_sampling_resume_signature()}
        )


def test_resume_rejects_enabling_task_sampling_for_legacy_checkpoint():
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer._task_sampling_plan = _small_plan()

    with pytest.raises(ValueError, match="predates"):
        trainer._validate_task_sampling_resume_state(
            {"version": 2, "global_steps": 10}
        )


def test_train_dataset_resume_signature_detects_same_counts_in_different_order():
    original = VisHarnessTrainer.__new__(VisHarnessTrainer)
    original.train_dataset = _FakeDataset(
        ["visharness/gres", "visharness/reasonseg", "visharness/gres"]
    )
    saved_signature = original._train_dataset_resume_signature()

    reordered = VisHarnessTrainer.__new__(VisHarnessTrainer)
    reordered.train_dataset = _FakeDataset(
        ["visharness/gres", "visharness/gres", "visharness/reasonseg"]
    )

    assert saved_signature["source_counts"] == reordered._train_dataset_resume_signature()[
        "source_counts"
    ]
    with pytest.raises(ValueError, match="Training data do not match"):
        reordered._validate_train_dataset_resume_state(
            {"version": 4, "train_dataset_signature": saved_signature}
        )


def test_dataloader_state_signature_detects_mismatched_runtime_snapshot(tmp_path):
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.global_steps = 5
    checkpoint_dir = tmp_path / "checkpoints" / "global_step_5"
    checkpoint_dir.mkdir(parents=True)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(tmp_path / "checkpoints"),
                "resume_mode": "auto",
            }
        }
    )
    saved_data_state = {"samples_yielded": 40, "generator": torch.tensor([1, 2])}
    torch.save(saved_data_state, checkpoint_dir / "data.pt")

    assert trainer._validate_dataloader_resume_state(
        {
            "version": 4,
            "dataloader_state_signature": trainer._dataloader_state_signature(
                saved_data_state
            ),
        }
    )
    with pytest.raises(ValueError, match="different runtime snapshots"):
        trainer._validate_dataloader_resume_state(
            {"version": 4, "dataloader_state_signature": "0" * 64}
        )
