"""Deterministic temperature-balanced sampling across training data sources."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Sampler


def _normalized_temperature_weights(
    source_counts: Mapping[str, int],
    *,
    alpha: float,
) -> dict[str, float]:
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError(f"task-sampling alpha must be finite and non-negative, got {alpha}")
    if not source_counts:
        raise ValueError("Cannot build task-balanced sampling plan from an empty dataset")
    invalid_counts = {source: count for source, count in source_counts.items() if int(count) <= 0}
    if invalid_counts:
        raise ValueError(f"Every data source must contain at least one sample, got {invalid_counts}")

    ordered_sources = sorted(str(source) for source in source_counts)
    weights = {source: float(source_counts[source]) ** alpha for source in ordered_sources}
    total_weight = sum(weights.values())
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise ValueError(f"Invalid task-sampling weight sum: {total_weight}")
    return {source: weights[source] / total_weight for source in ordered_sources}


def balanced_source_sequence(
    probabilities: Mapping[str, float],
    num_slots: int,
) -> list[str]:
    """Return a deterministic low-discrepancy source sequence.

    At every prefix, the assigned source counts stay close to the requested
    probabilities. This avoids the large per-update task-composition variance
    of independent weighted sampling.
    """

    if num_slots < 0:
        raise ValueError(f"num_slots must be non-negative, got {num_slots}")
    if num_slots == 0:
        return []
    if not probabilities:
        raise ValueError("probabilities cannot be empty")

    ordered_sources = sorted(str(source) for source in probabilities)
    normalized = {source: float(probabilities[source]) for source in ordered_sources}
    if any(not math.isfinite(value) or value < 0 for value in normalized.values()):
        raise ValueError(f"Probabilities must be finite and non-negative, got {normalized}")
    probability_sum = sum(normalized.values())
    if probability_sum <= 0:
        raise ValueError(f"Probability sum must be positive, got {probability_sum}")
    normalized = {source: value / probability_sum for source, value in normalized.items()}

    assigned = {source: 0 for source in ordered_sources}
    sequence: list[str] = []
    for slot_index in range(num_slots):
        prefix_size = slot_index + 1
        source = max(
            ordered_sources,
            key=lambda candidate: (
                normalized[candidate] * prefix_size - assigned[candidate],
                -ordered_sources.index(candidate),
            ),
        )
        assigned[source] += 1
        sequence.append(source)
    return sequence


def source_quota_for_update(
    probabilities: Mapping[str, float],
    *,
    batch_size: int,
    update_index: int,
) -> dict[str, int]:
    """Return the integer source quota for one optimizer update."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if update_index < 0:
        raise ValueError(f"update_index must be non-negative, got {update_index}")
    start = update_index * batch_size
    sequence = balanced_source_sequence(probabilities, start + batch_size)
    quota = Counter(sequence[start:])
    return {source: int(quota.get(source, 0)) for source in sorted(probabilities)}


def _source_counts_for_slots(probabilities: Mapping[str, float], num_slots: int) -> Counter:
    return Counter(balanced_source_sequence(probabilities, num_slots))


def _epoch_num_samples(
    source_counts: Mapping[str, int],
    probabilities: Mapping[str, float],
    *,
    batch_size: int,
    mode: str,
) -> int:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    mode = str(mode).lower()
    if mode == "dataset_size":
        return max((sum(int(count) for count in source_counts.values()) // batch_size) * batch_size, batch_size)
    if mode != "all_sources_once":
        raise ValueError(
            "task-sampling epoch_size_mode must be 'all_sources_once' or 'dataset_size', "
            f"got {mode!r}"
        )

    approximate_slots = max(
        float(source_counts[source]) / max(float(probabilities[source]), 1e-12)
        for source in source_counts
    )
    num_slots = max(int(approximate_slots // batch_size) * batch_size, batch_size)
    while True:
        allocated = _source_counts_for_slots(probabilities, num_slots)
        if all(allocated[source] >= int(source_counts[source]) for source in source_counts):
            return num_slots
        num_slots += batch_size


@dataclass(frozen=True)
class TaskSamplingPlan:
    """Immutable dataset-level task-sampling metadata."""

    indices_by_source: dict[str, tuple[int, ...]]
    source_counts: dict[str, int]
    probabilities: dict[str, float]
    num_samples: int
    batch_size: int
    alpha: float
    epoch_size_mode: str

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(sorted(self.source_counts))

    def quota_for_update(self, update_index: int) -> dict[str, int]:
        return source_quota_for_update(
            self.probabilities,
            batch_size=self.batch_size,
            update_index=update_index,
        )


@dataclass(frozen=True)
class TaskPromptDraw:
    """One prompt selected from an independently cycling task source."""

    dataset_index: int
    source: str
    epoch_index: int
    source_cycle: int
    source_draw_index: int
    total_source_draw_index: int

    @property
    def attempt_uid(self) -> str:
        return (
            f"task-prompt/e{self.epoch_index}/{self.source}/"
            f"c{self.source_cycle}/d{self.total_source_draw_index}"
        )


class CyclingTaskSourcePool:
    """Independent shuffled-without-replacement pools for every task source.

    A source is reshuffled immediately when another prompt is requested after
    its current permutation is exhausted.  Epoch completion is coverage based:
    every source must have exhausted its first permutation at least once.
    """

    _STATE_VERSION = 1

    def __init__(
        self,
        plan: TaskSamplingPlan,
        *,
        seed: int,
        epoch_index: int = 0,
    ) -> None:
        self.plan = plan
        self.seed = int(seed)
        self.epoch_index = int(epoch_index)
        if self.epoch_index < 0:
            raise ValueError(f"epoch_index must be non-negative, got {epoch_index}")
        self._generators: dict[str, torch.Generator] = {}
        self._states: dict[str, dict[str, Any]] = {}
        for source in self.plan.sources:
            generator = torch.Generator()
            generator.manual_seed(self._source_seed(source))
            self._generators[source] = generator
            self._states[source] = self._new_source_epoch_state(source, total_draws=0)

    def _source_seed(self, source: str) -> int:
        digest = hashlib.sha256(f"{self.seed}\0{source}".encode("utf-8")).digest()
        # torch.Generator.manual_seed accepts signed 64-bit-compatible values.
        return int.from_bytes(digest[:8], "big", signed=False) % (2**63 - 1)

    def _new_permutation(self, source: str) -> list[int]:
        source_indices = self.plan.indices_by_source[source]
        offsets = torch.randperm(
            len(source_indices),
            generator=self._generators[source],
        ).tolist()
        return [int(source_indices[offset]) for offset in offsets]

    def _new_source_epoch_state(self, source: str, *, total_draws: int) -> dict[str, Any]:
        return {
            "permutation": self._new_permutation(source),
            "cursor": 0,
            "cycles_completed": 0,
            "first_pass_completed": False,
            "draws_in_epoch": 0,
            "total_draws": int(total_draws),
        }

    @property
    def all_sources_first_pass_completed(self) -> bool:
        return all(
            bool(self._states[source]["first_pass_completed"])
            for source in self.plan.sources
        )

    def source_progress(self, source: str) -> dict[str, int | float | bool]:
        if source not in self._states:
            raise KeyError(f"Unknown task source {source!r}")
        state = self._states[source]
        source_size = len(self.plan.indices_by_source[source])
        first_pass_draws = source_size if state["first_pass_completed"] else int(state["cursor"])
        return {
            "source_size": source_size,
            "first_pass_draws": first_pass_draws,
            "first_pass_coverage": float(first_pass_draws) / float(source_size),
            "first_pass_completed": bool(state["first_pass_completed"]),
            "cycles_completed": int(state["cycles_completed"]),
            "draws_in_epoch": int(state["draws_in_epoch"]),
            "total_draws": int(state["total_draws"]),
        }

    def progress_by_source(self) -> dict[str, dict[str, int | float | bool]]:
        return {source: self.source_progress(source) for source in self.plan.sources}

    def take(self, requested_counts: Mapping[str, int]) -> tuple[list[TaskPromptDraw], dict[str, int]]:
        """Return exactly the requested source counts and completed-cycle deltas."""

        normalized = {
            str(source): int(count)
            for source, count in requested_counts.items()
            if int(count) > 0
        }
        unknown_sources = set(normalized) - set(self.plan.sources)
        if unknown_sources:
            raise KeyError(
                f"Requested unknown task sources {sorted(unknown_sources)}; "
                f"known sources are {list(self.plan.sources)}"
            )
        negative_counts = {
            str(source): int(count)
            for source, count in requested_counts.items()
            if int(count) < 0
        }
        if negative_counts:
            raise ValueError(f"Task source request counts must be non-negative, got {negative_counts}")
        if not normalized:
            return [], {}

        draws_by_source: dict[str, list[TaskPromptDraw]] = {}
        completed_cycles: dict[str, int] = defaultdict(int)
        for source, count in normalized.items():
            state = self._states[source]
            source_draws: list[TaskPromptDraw] = []
            for _ in range(count):
                if int(state["cursor"]) >= len(state["permutation"]):
                    state["permutation"] = self._new_permutation(source)
                    state["cursor"] = 0

                cursor = int(state["cursor"])
                source_cycle = int(state["cycles_completed"])
                source_draw_index = int(state["draws_in_epoch"])
                total_source_draw_index = int(state["total_draws"])
                source_draws.append(
                    TaskPromptDraw(
                        dataset_index=int(state["permutation"][cursor]),
                        source=source,
                        epoch_index=self.epoch_index,
                        source_cycle=source_cycle,
                        source_draw_index=source_draw_index,
                        total_source_draw_index=total_source_draw_index,
                    )
                )
                state["cursor"] = cursor + 1
                state["draws_in_epoch"] = source_draw_index + 1
                state["total_draws"] = total_source_draw_index + 1
                if int(state["cursor"]) == len(state["permutation"]):
                    state["cycles_completed"] = source_cycle + 1
                    state["first_pass_completed"] = True
                    completed_cycles[source] += 1
            draws_by_source[source] = source_draws

        # Preserve the quota while avoiding a permanent source-to-worker order.
        source_sequence = balanced_source_sequence(normalized, sum(normalized.values()))
        source_offsets = defaultdict(int)
        ordered_draws: list[TaskPromptDraw] = []
        for source in source_sequence:
            offset = source_offsets[source]
            ordered_draws.append(draws_by_source[source][offset])
            source_offsets[source] = offset + 1
        if Counter(draw.source for draw in ordered_draws) != Counter(normalized):
            raise RuntimeError(
                "Cycling task-source pool returned an incorrect quota: "
                f"requested={normalized}, returned={Counter(draw.source for draw in ordered_draws)}"
            )
        return ordered_draws, dict(completed_cycles)

    def start_next_epoch(self) -> None:
        if not self.all_sources_first_pass_completed:
            incomplete = [
                source
                for source in self.plan.sources
                if not self._states[source]["first_pass_completed"]
            ]
            raise RuntimeError(
                "Cannot start the next task-sampling epoch before every source "
                f"finishes its first pass; incomplete={incomplete}"
            )
        self.epoch_index += 1
        for source in self.plan.sources:
            total_draws = int(self._states[source]["total_draws"])
            self._states[source] = self._new_source_epoch_state(
                source,
                total_draws=total_draws,
            )

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self._STATE_VERSION,
            "seed": self.seed,
            "epoch_index": self.epoch_index,
            "sources": {
                source: {
                    **{
                        key: list(value) if key == "permutation" else value
                        for key, value in self._states[source].items()
                    },
                    "generator_state": self._generators[source].get_state(),
                }
                for source in self.plan.sources
            },
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        version = int(state_dict.get("version", 0))
        if version != self._STATE_VERSION:
            raise ValueError(
                f"Unsupported cycling task-source pool state version {version}; "
                f"expected {self._STATE_VERSION}"
            )
        if int(state_dict.get("seed")) != self.seed:
            raise ValueError(
                "Task-source pool seed does not match the checkpoint: "
                f"saved={state_dict.get('seed')}, current={self.seed}"
            )
        epoch_index = int(state_dict.get("epoch_index", -1))
        if epoch_index < 0:
            raise ValueError(f"Invalid restored task-source epoch index {epoch_index}")
        saved_sources = state_dict.get("sources")
        if not isinstance(saved_sources, Mapping) or set(saved_sources) != set(self.plan.sources):
            raise ValueError(
                "Task-source pool sources do not match the current plan: "
                f"saved={sorted(saved_sources) if isinstance(saved_sources, Mapping) else saved_sources}, "
                f"current={list(self.plan.sources)}"
            )

        restored_states: dict[str, dict[str, Any]] = {}
        for source in self.plan.sources:
            raw_state = saved_sources[source]
            if not isinstance(raw_state, Mapping):
                raise TypeError(f"Invalid state for task source {source!r}: {type(raw_state)}")
            permutation = [int(index) for index in raw_state.get("permutation", [])]
            expected_indices = list(self.plan.indices_by_source[source])
            if len(permutation) != len(expected_indices) or set(permutation) != set(expected_indices):
                raise ValueError(
                    f"Restored permutation for {source!r} is not an exact permutation "
                    "of the current dataset indices"
                )
            cursor = int(raw_state.get("cursor", -1))
            if cursor < 0 or cursor > len(permutation):
                raise ValueError(
                    f"Invalid restored cursor for {source!r}: {cursor}; "
                    f"expected [0, {len(permutation)}]"
                )
            cycles_completed = int(raw_state.get("cycles_completed", -1))
            draws_in_epoch = int(raw_state.get("draws_in_epoch", -1))
            total_draws = int(raw_state.get("total_draws", -1))
            first_pass_completed = bool(raw_state.get("first_pass_completed", False))
            if cycles_completed < 0 or draws_in_epoch < 0 or total_draws < draws_in_epoch:
                raise ValueError(f"Invalid restored counters for task source {source!r}: {dict(raw_state)}")
            if first_pass_completed != (cycles_completed > 0):
                raise ValueError(
                    f"Inconsistent first-pass state for {source!r}: "
                    f"completed={first_pass_completed}, cycles={cycles_completed}"
                )
            expected_draws = cycles_completed * len(permutation)
            if cursor < len(permutation):
                expected_draws += cursor
            if draws_in_epoch != expected_draws:
                raise ValueError(
                    f"Inconsistent draw position for {source!r}: draws_in_epoch={draws_in_epoch}, "
                    f"cycles={cycles_completed}, cursor={cursor}, source_size={len(permutation)}"
                )
            generator_state = raw_state.get("generator_state")
            if not isinstance(generator_state, torch.Tensor):
                raise TypeError(f"Missing generator state for task source {source!r}")
            self._generators[source].set_state(generator_state)
            restored_states[source] = {
                "permutation": permutation,
                "cursor": cursor,
                "cycles_completed": cycles_completed,
                "first_pass_completed": first_pass_completed,
                "draws_in_epoch": draws_in_epoch,
                "total_draws": total_draws,
            }

        self.epoch_index = epoch_index
        self._states = restored_states


def build_task_sampling_plan(
    dataset,
    *,
    alpha: float,
    batch_size: int,
    epoch_size_mode: str = "all_sources_once",
    source_key: str = "data_source",
) -> TaskSamplingPlan:
    """Build a task-sampling plan from an ``RLHFDataset``-like object."""

    dataframe = getattr(dataset, "dataframe", None)
    if dataframe is None:
        raise TypeError("Task-balanced sampling requires dataset.dataframe")
    if source_key not in dataframe.column_names:
        raise KeyError(
            f"Task-balanced sampling requires dataframe column {source_key!r}; "
            f"available columns are {dataframe.column_names}"
        )

    raw_sources: Sequence[Any] = dataframe[source_key]
    indices_by_source_lists: dict[str, list[int]] = defaultdict(list)
    for index, raw_source in enumerate(raw_sources):
        source = str(raw_source)
        if not source:
            raise ValueError(f"Empty data source at dataset index {index}")
        indices_by_source_lists[source].append(index)

    source_counts = {
        source: len(indices)
        for source, indices in sorted(indices_by_source_lists.items())
    }
    probabilities = _normalized_temperature_weights(source_counts, alpha=alpha)
    num_samples = _epoch_num_samples(
        source_counts,
        probabilities,
        batch_size=batch_size,
        mode=epoch_size_mode,
    )
    return TaskSamplingPlan(
        indices_by_source={
            source: tuple(indices)
            for source, indices in sorted(indices_by_source_lists.items())
        },
        source_counts=source_counts,
        probabilities=probabilities,
        num_samples=num_samples,
        batch_size=int(batch_size),
        alpha=float(alpha),
        epoch_size_mode=str(epoch_size_mode).lower(),
    )


class _TemperatureBalancedSamplerIterator(Iterator[int]):
    _GENERATOR_STATE = "generator_state"
    _YIELDED = "yielded"

    def __init__(self, sampler: "TemperatureBalancedSampler") -> None:
        self.sampler = sampler
        self.generator_state = sampler.generator.get_state()
        self.yielded = 0
        self._indices = self._build_indices()

    def __iter__(self) -> "_TemperatureBalancedSamplerIterator":
        return self

    def _build_indices(self) -> list[int]:
        source_sequence = balanced_source_sequence(
            self.sampler.plan.probabilities,
            self.sampler.plan.num_samples,
        )
        # Quotas are fixed per update, but task positions inside each batch
        # should not be permanently tied to the same rollout-worker order.
        for start in range(0, len(source_sequence), self.sampler.plan.batch_size):
            stop = min(start + self.sampler.plan.batch_size, len(source_sequence))
            block = source_sequence[start:stop]
            permutation = torch.randperm(
                len(block),
                generator=self.sampler.generator,
            ).tolist()
            source_sequence[start:stop] = [block[index] for index in permutation]
        required_counts = Counter(source_sequence)
        sampled_indices: dict[str, list[int]] = {}
        for source in self.sampler.plan.sources:
            source_indices = self.sampler.plan.indices_by_source[source]
            required = int(required_counts.get(source, 0))
            draws: list[int] = []
            while len(draws) < required:
                permutation = torch.randperm(
                    len(source_indices),
                    generator=self.sampler.generator,
                ).tolist()
                draws.extend(source_indices[offset] for offset in permutation)
            sampled_indices[source] = draws[:required]

        source_offsets = defaultdict(int)
        result: list[int] = []
        for source in source_sequence:
            offset = source_offsets[source]
            result.append(sampled_indices[source][offset])
            source_offsets[source] = offset + 1
        return result

    def __next__(self) -> int:
        if self.yielded >= len(self._indices):
            raise StopIteration
        value = self._indices[self.yielded]
        self.yielded += 1
        return value

    def state_dict(self) -> dict[str, Any]:
        return {
            self._GENERATOR_STATE: self.generator_state,
            self._YIELDED: int(self.yielded),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        yielded = int(state_dict[self._YIELDED])
        if yielded < 0 or yielded > len(self.sampler):
            raise ValueError(
                f"Invalid restored task-balanced sampler position {yielded}; "
                f"expected [0, {len(self.sampler)}]"
            )
        self.generator_state = state_dict[self._GENERATOR_STATE]
        self.sampler.generator.set_state(self.generator_state)
        self._indices = self._build_indices()
        self.yielded = yielded


class TemperatureBalancedSampler(Sampler[int]):
    """Stateful finite sampler with temperature-balanced source composition."""

    def __init__(self, plan: TaskSamplingPlan, *, seed: int | None = None) -> None:
        self.plan = plan
        self.generator = torch.Generator()
        if seed is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self.generator.manual_seed(int(seed))

    def __iter__(self) -> Iterator[int]:
        return _TemperatureBalancedSamplerIterator(self)

    def __len__(self) -> int:
        return int(self.plan.num_samples)


def create_temperature_balanced_sampler(
    dataset,
    *,
    alpha: float,
    batch_size: int,
    seed: int | None,
    epoch_size_mode: str = "all_sources_once",
    source_key: str = "data_source",
) -> TemperatureBalancedSampler:
    plan = build_task_sampling_plan(
        dataset,
        alpha=alpha,
        batch_size=batch_size,
        epoch_size_mode=epoch_size_mode,
        source_key=source_key,
    )
    return TemperatureBalancedSampler(plan, seed=seed)
