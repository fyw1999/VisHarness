import asyncio
import time
from types import SimpleNamespace

import numpy as np
import torch
from verl import DataProto
from verl.experimental.agent_loop import AgentLoopWorker as VerlAgentLoopWorker

from visharness.agent_loop.dynamic_validation_manager import (
    VisHarnessAgentLoopManager,
    VisHarnessAgentLoopWorker,
    _normalize_worker_output_fields,
)
from visharness.rl_trainer.visharness_trainer import (
    _align_dynamic_validation_reward_info,
)


class _RemoteMethod:
    def __init__(self, function):
        self._function = function

    def remote(self, value):
        return self._function(value)


class _FakeWorker:
    def __init__(self, state):
        self._state = state
        self.generate_sequences = _RemoteMethod(self._generate_sequences)

    async def _generate_sequences(self, job):
        job_index, delay = job
        self._state["active"] += 1
        self._state["peak_active"] = max(
            self._state["peak_active"],
            self._state["active"],
        )
        self._state["started"][job_index] = time.perf_counter()
        await asyncio.sleep(delay)
        self._state["finished"][job_index] = time.perf_counter()
        self._state["active"] -= 1
        return DataProto(meta_info={"metrics": [{"job": job_index}], "kept": job_index})


class _TrainingWorker:
    def __init__(self, index, delay):
        self.index = index
        self.delay = delay
        self.generate_sequences = _RemoteMethod(self._generate_sequences)

    async def _generate_sequences(self, _job):
        await asyncio.sleep(self.delay)
        return DataProto(
            non_tensor_batch={"worker_index": np.array([self.index])},
            meta_info={
                "metrics": [
                    {
                        "generate_sequences": 0.0,
                        "tool_calls": 0.0,
                        "compute_score": 0.0,
                        "num_preempted": 0,
                    }
                ]
            },
        )


class _TrainingPrompts:
    meta_info = {"validate": True}

    def __len__(self):
        return 2

    def chunk(self, count):
        assert count == 2
        return [object(), object()]


def test_training_manager_restores_worker_order_after_async_completion():
    manager = object.__new__(VisHarnessAgentLoopManager)
    manager.agent_loop_workers = [
        _TrainingWorker(0, 0.05),
        _TrainingWorker(1, 0.01),
    ]
    manager.config = {}
    manager._performance_metrics = lambda _metrics, _output: {}

    output = manager.generate_sequences(_TrainingPrompts())

    assert output.non_tensor_batch["worker_index"].tolist() == [0, 1]


def test_worker_normalizes_optional_reward_fields_before_upstream_postprocess(
    monkeypatch,
):
    inputs = [
        SimpleNamespace(extra_fields={"reward_extra_info": {"iou": 0.5}}),
        SimpleNamespace(
            extra_fields={"reward_extra_info": {"task_type": "point"}}
        ),
    ]

    def fake_upstream_postprocess(
        _self,
        normalized_inputs,
        input_non_tensor_batch=None,
        validate=False,
    ):
        del input_non_tensor_batch, validate
        infos = [
            item.extra_fields["reward_extra_info"]
            for item in normalized_inputs
        ]
        assert list(infos[0]) == ["iou", "task_type"]
        assert list(infos[1]) == ["iou", "task_type"]
        return DataProto(
            non_tensor_batch={
                key: np.array([info[key] for info in infos])
                for key in infos[0]
            }
        )

    monkeypatch.setattr(
        VerlAgentLoopWorker,
        "_postprocess",
        fake_upstream_postprocess,
    )
    worker = object.__new__(VisHarnessAgentLoopWorker)

    output = worker._postprocess(inputs)

    assert output.non_tensor_batch["iou"].tolist() == [0.5, None]
    assert output.non_tensor_batch["task_type"].tolist() == [None, "point"]
    assert output.non_tensor_batch["iou"].dtype == object


def test_manager_normalizes_optional_fields_across_workers():
    outputs = [
        DataProto(
            non_tensor_batch={"iou": np.array([0.5])},
            meta_info={"reward_extra_keys": ["iou"]},
        ),
        DataProto(
            non_tensor_batch={"task_type": np.array(["point"])},
            meta_info={"reward_extra_keys": ["task_type"]},
        ),
    ]

    _normalize_worker_output_fields(outputs)

    assert outputs[0].meta_info["reward_extra_keys"] == ["iou", "task_type"]
    assert outputs[1].meta_info["reward_extra_keys"] == ["iou", "task_type"]
    assert outputs[0].non_tensor_batch["task_type"].tolist() == [None]
    assert outputs[1].non_tensor_batch["iou"].tolist() == [None]


def test_dynamic_validation_refills_each_completed_slot_immediately():
    state = {
        "active": 0,
        "peak_active": 0,
        "started": {},
        "finished": {},
    }
    manager = object.__new__(VisHarnessAgentLoopManager)
    manager.agent_loop_workers = [_FakeWorker(state), _FakeWorker(state)]

    jobs = [
        (0, (0, 0.10), None),
        (1, (1, 0.01), None),
        (2, (2, 0.01), None),
        (3, (3, 0.01), None),
        (4, (4, 0.01), None),
        (5, (5, 0.01), None),
    ]
    completed = {}
    stats = manager.generate_validation_sequences(
        jobs,
        max_in_flight=4,
        on_complete=lambda sequence_index, _context, output: completed.update(
            {sequence_index: output.meta_info}
        ),
        total_jobs=len(jobs),
        progress_interval=100,
    )

    assert sorted(completed) == list(range(len(jobs)))
    assert all("metrics" not in meta_info for meta_info in completed.values())
    assert [completed[index]["kept"] for index in range(len(jobs))] == list(range(len(jobs)))
    assert stats["completed"] == len(jobs)
    assert stats["peak_in_flight"] == 4
    assert state["peak_active"] == 4
    assert state["started"][4] < state["finished"][0]


def test_dynamic_validation_reward_fields_stay_aligned_when_one_is_missing():
    completed_results = {
        0: {
            "score": 1.0,
            "reward_extra_info": {
                "assistant_response_token_count": [10],
                "task_type": ["point"],
            },
        },
        1: {
            "score": 0.0,
            "reward_extra_info": {},
        },
        2: {
            "score": 2.0,
            "reward_extra_info": {
                "assistant_response_token_count": [30],
                "task_type": ["gres_rle_mask"],
            },
        },
    }

    aligned = _align_dynamic_validation_reward_info(
        completed_results,
        total_jobs=3,
    )

    assert aligned["reward"] == [1.0, 0.0, 2.0]
    assert aligned["assistant_response_token_count"] == [10, None, 30]
    assert aligned["task_type"] == ["point", None, "gres_rle_mask"]


class _StreamingTrainingWorker:
    def __init__(self, state):
        self._state = state
        self.generate_sequences = _RemoteMethod(self._generate_sequences)

    async def _generate_sequences(self, batch):
        job_name = str(batch.non_tensor_batch["job_name"][0])
        delay = float(batch.non_tensor_batch["delay"][0])
        self._state["started"][job_name] = time.perf_counter()
        await asyncio.sleep(delay)
        self._state["finished"][job_name] = time.perf_counter()
        return DataProto.from_dict(
            tensors={
                "prompts": torch.ones((1, 2), dtype=torch.long),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
            },
            non_tensors={"job_name": np.array([job_name], dtype=object)},
            meta_info={
                "metrics": [
                    {
                        "generate_sequences": delay,
                        "tool_calls": 0.0,
                        "compute_score": 0.0,
                        "num_preempted": 0,
                    }
                ]
            },
        )


def _streaming_training_batch(job_name, delay):
    return DataProto.from_dict(
        tensors={"placeholder": torch.zeros((1, 1), dtype=torch.long)},
        non_tensors={
            "job_name": np.array([job_name], dtype=object),
            "delay": np.array([delay], dtype=np.float64),
        },
    )


def test_streaming_training_refills_failed_group_before_slow_group_finishes():
    state = {"started": {}, "finished": {}}
    manager = object.__new__(VisHarnessAgentLoopManager)
    manager.agent_loop_workers = [
        _StreamingTrainingWorker(state),
        _StreamingTrainingWorker(state),
    ]
    attempts = {0: 0, 1: 0}
    accepted = {}

    def on_complete(slot_index, _context, output):
        job_name = str(output.non_tensor_batch["job_name"][0])
        attempts[slot_index] += 1
        if slot_index == 1 and attempts[slot_index] == 1:
            return (
                slot_index,
                _streaming_training_batch("fast-replacement", 0.01),
                None,
            )
        accepted[slot_index] = job_name
        return None

    stats = manager.generate_training_prompt_groups(
        [
            (0, _streaming_training_batch("slow-initial", 0.10), None),
            (1, _streaming_training_batch("fast-rejected", 0.01), None),
        ],
        on_complete=on_complete,
        total_slots=2,
        trajectories_per_group=1,
        progress_interval=100,
    )

    assert accepted == {0: "slow-initial", 1: "fast-replacement"}
    assert stats["submitted_groups"] == 3
    assert stats["completed_slots"] == 2
    assert stats["peak_in_flight_groups"] == 2
    assert state["started"]["fast-replacement"] < state["finished"]["slow-initial"]
    assert stats["timing"]["agent_loop/generate_sequences/max"] == 0.10


def test_streaming_training_serves_all_slots_when_workers_are_fewer():
    state = {"started": {}, "finished": {}}
    manager = object.__new__(VisHarnessAgentLoopManager)
    manager.agent_loop_workers = [
        _StreamingTrainingWorker(state),
        _StreamingTrainingWorker(state),
    ]
    attempts = {0: 0, 1: 0, 2: 0}
    accepted = {}

    def on_complete(slot_index, _context, output):
        job_name = str(output.non_tensor_batch["job_name"][0])
        attempts[slot_index] += 1
        if slot_index == 0 and attempts[slot_index] == 1:
            return (
                slot_index,
                _streaming_training_batch("slot-0-replacement", 0.01),
                None,
            )
        accepted[slot_index] = job_name
        return None

    stats = manager.generate_training_prompt_groups(
        [
            (0, _streaming_training_batch("slot-0-initial", 0.01), None),
            (1, _streaming_training_batch("slot-1-initial", 0.02), None),
            (2, _streaming_training_batch("slot-2-initial", 0.01), None),
        ],
        on_complete=on_complete,
        total_slots=3,
        trajectories_per_group=1,
        progress_interval=100,
    )

    assert accepted == {
        0: "slot-0-replacement",
        1: "slot-1-initial",
        2: "slot-2-initial",
    }
    assert stats["completed_slots"] == 3
    assert stats["submitted_groups"] == 4
    assert stats["peak_in_flight_groups"] == 2
    assert state["started"]["slot-2-initial"] <= state["started"][
        "slot-0-replacement"
    ]
