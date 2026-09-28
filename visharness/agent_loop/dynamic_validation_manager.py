"""Validation-specific continuous scheduling for VisHarness agent rollouts."""

import asyncio
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
import ray

from verl import DataProto
from verl.experimental.agent_loop import AgentLoopManager as VerlAgentLoopManager
from verl.experimental.agent_loop import AgentLoopWorker as VerlAgentLoopWorker
from verl.utils.ray_utils import auto_await


ValidationJob = tuple[int, DataProto, Any]
ValidationCallback = Callable[[int, Any, DataProto], None]
TrainingGroupJob = tuple[int, DataProto, Any]
TrainingGroupCallback = Callable[
    [int, Any, DataProto], TrainingGroupJob | None
]


def _normalize_reward_extra_infos(inputs: list[Any]) -> list[str]:
    """Give every trajectory the same reward-extra field set.

    Upstream verl 0.8.0 builds these arrays from the first trajectory's keys.
    VisHarness mixes tasks and valid/invalid trajectories, so optional reward
    metrics are not guaranteed to be identical within one worker batch.
    """

    reward_extra_infos: list[Mapping[str, Any]] = []
    for item in inputs:
        info = item.extra_fields.get("reward_extra_info", {})
        if info is None:
            info = {}
        if not isinstance(info, Mapping):
            raise TypeError(
                "reward_extra_info must be a mapping, got "
                f"{type(info).__name__}"
            )
        reward_extra_infos.append(info)

    reward_extra_keys = sorted(
        {key for info in reward_extra_infos for key in info}
    )
    for item, info in zip(inputs, reward_extra_infos, strict=True):
        item.extra_fields["reward_extra_info"] = {
            key: info.get(key) for key in reward_extra_keys
        }
    return reward_extra_keys


def _normalize_worker_output_fields(outputs: list[DataProto]) -> None:
    """Make optional fields concat-safe across agent-loop worker outputs."""

    reward_extra_keys = sorted(
        {
            key
            for output in outputs
            for key in output.meta_info.get("reward_extra_keys", [])
        }
    )
    non_tensor_keys = sorted(
        {key for output in outputs for key in output.non_tensor_batch}
    )

    for output in outputs:
        if reward_extra_keys:
            output.meta_info["reward_extra_keys"] = reward_extra_keys
        for key in reward_extra_keys:
            if key not in output.non_tensor_batch:
                output.non_tensor_batch[key] = np.array(
                    [None] * len(output), dtype=object
                )
        for key in non_tensor_keys:
            if key not in output.non_tensor_batch:
                output.non_tensor_batch[key] = np.array(
                    [None] * len(output), dtype=object
                )


class VisHarnessAgentLoopWorker(VerlAgentLoopWorker):
    """Use upstream rollout behavior with VisHarness-safe reward collation."""

    def _postprocess(
        self,
        inputs: list[Any],
        input_non_tensor_batch: dict | None = None,
        validate: bool = False,
    ) -> DataProto:
        reward_extra_keys = _normalize_reward_extra_infos(inputs)
        output = super()._postprocess(
            inputs,
            input_non_tensor_batch=input_non_tensor_batch,
            validate=validate,
        )
        # Preserve the previous VisHarness behavior for heterogeneous values
        # such as strings, None, nested dictionaries, and numeric metrics.
        for key in reward_extra_keys:
            output.non_tensor_batch[key] = np.asarray(
                output.non_tensor_batch[key], dtype=object
            )
        return output


class VisHarnessAgentLoopManager(VerlAgentLoopManager):
    """VisHarness rollout collation, progress reporting, and validation scheduling."""

    agent_loop_workers_class = ray.remote(VisHarnessAgentLoopWorker)

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Dispatch worker chunks while preserving input order and reporting progress."""

        chunks = prompts.chunk(len(self.agent_loop_workers))

        async def run_worker_chunk(index: int, worker, chunk: DataProto):
            return index, await worker.generate_sequences.remote(chunk)

        tasks = [
            asyncio.create_task(run_worker_chunk(index, worker, chunk))
            for index, (worker, chunk) in enumerate(
                zip(self.agent_loop_workers, chunks, strict=True)
            )
        ]
        show_progress = (
            not prompts.meta_info.get("validate", False)
            and bool(
                self.config.get("visharness", {}).get(
                    "rollout_progress", False
                )
            )
            and len(prompts) > 0
        )

        if show_progress:
            outputs: list[DataProto | None] = [None] * len(tasks)
            completed = 0
            total = len(prompts)
            total_chunks = len(tasks)
            started_at = time.perf_counter()
            last_report_at = started_at
            print(
                f"[rollout progress] started 0/{total} trajectories across "
                f"{total_chunks} worker chunks",
                flush=True,
            )
            for finished_chunks, task in enumerate(
                asyncio.as_completed(tasks), start=1
            ):
                index, chunk_output = await task
                outputs[index] = chunk_output
                completed += len(chunk_output)
                now = time.perf_counter()
                elapsed = now - started_at
                chunk_elapsed = now - last_report_at
                last_report_at = now
                average_per_trajectory = elapsed / max(completed, 1)
                remaining = max(total - completed, 0)
                eta = average_per_trajectory * remaining
                print(
                    f"[rollout progress] completed {completed}/{total} trajectories "
                    f"({finished_chunks}/{total_chunks} worker chunks), "
                    f"elapsed={elapsed:.1f}s, last_chunk={chunk_elapsed:.1f}s, "
                    f"avg={average_per_trajectory:.2f}s/traj, eta={eta:.1f}s",
                    flush=True,
                )
        else:
            indexed_outputs = await asyncio.gather(*tasks)
            outputs = [None] * len(indexed_outputs)
            for index, chunk_output in indexed_outputs:
                outputs[index] = chunk_output

        ordered_outputs = [output for output in outputs if output is not None]
        _normalize_worker_output_fields(ordered_outputs)
        output = DataProto.concat(ordered_outputs)

        metrics = [
            worker_output.meta_info.pop("metrics")
            for worker_output in ordered_outputs
        ]
        timing = self._performance_metrics(metrics, output)
        output.meta_info = {
            "timing": timing,
            **ordered_outputs[0].meta_info,
        }
        return output

    @staticmethod
    def _streaming_performance_metrics(
        metric_records: list[dict[str, Any]],
        sample_lengths: list[tuple[int, int]],
    ) -> dict[str, float]:
        """Aggregate upstream per-trajectory timings without retaining outputs.

        Streaming training may discard many completed prompt groups. Keeping all
        of their padded tensors until the actor batch is full would defeat the
        memory benefit of processing one group at a time, so only the lightweight
        timing records and sequence lengths are retained here.
        """

        if not metric_records:
            return {}
        if len(metric_records) != len(sample_lengths):
            raise RuntimeError(
                "Streaming rollout timing records do not match completed samples: "
                f"metrics={len(metric_records)}, lengths={len(sample_lengths)}"
            )

        timing: dict[str, float] = {}
        timing_fields = (
            "generate_sequences",
            "tool_calls",
            "compute_score",
            "num_preempted",
        )
        arrays: dict[str, np.ndarray] = {}
        for field in timing_fields:
            try:
                values = np.asarray(
                    [float(record[field]) for record in metric_records],
                    dtype=np.float64,
                )
            except KeyError as error:
                raise KeyError(
                    f"Streaming rollout metric is missing required field {field!r}"
                ) from error
            arrays[field] = values
            timing[f"agent_loop/{field}/min"] = float(values.min())
            timing[f"agent_loop/{field}/max"] = float(values.max())
            timing[f"agent_loop/{field}/mean"] = float(values.mean())

        total_times = (
            arrays["generate_sequences"]
            + arrays["tool_calls"]
            + arrays["compute_score"]
        )
        slowest = int(np.argmax(total_times))
        timing["agent_loop/slowest/generate_sequences"] = float(
            arrays["generate_sequences"][slowest]
        )
        timing["agent_loop/slowest/tool_calls"] = float(
            arrays["tool_calls"][slowest]
        )
        timing["agent_loop/slowest/compute_score"] = float(
            arrays["compute_score"][slowest]
        )
        timing["agent_loop/slowest/num_preempted"] = float(
            arrays["num_preempted"][slowest]
        )
        prompt_length, response_length = sample_lengths[slowest]
        timing["agent_loop/slowest/prompt_length"] = float(prompt_length)
        timing["agent_loop/slowest/response_length"] = float(response_length)
        return timing

    @auto_await
    async def generate_training_prompt_groups(
        self,
        jobs: Iterable[TrainingGroupJob],
        *,
        on_complete: TrainingGroupCallback,
        total_slots: int,
        trajectories_per_group: int,
        progress_interval: int = 1,
    ) -> dict[str, Any]:
        """Continuously refill training prompt-group slots as groups finish.

        One job contains every rollout trajectory for one prompt. The callback
        either accepts that quota slot by returning ``None`` or returns a
        replacement job for the same slot. Actor parameters are not updated
        while this scheduler is active, so all accepted and replacement groups
        are generated by one policy version.
        """

        if total_slots <= 0:
            raise ValueError(f"total_slots must be positive, got {total_slots}")
        if trajectories_per_group <= 0:
            raise ValueError(
                "trajectories_per_group must be positive, got "
                f"{trajectories_per_group}"
            )
        if progress_interval <= 0:
            raise ValueError(
                f"progress_interval must be positive, got {progress_interval}"
            )
        if not self.agent_loop_workers:
            raise RuntimeError("No agent-loop workers are available for training")

        initial_jobs = list(jobs)
        initial_slots = [int(job[0]) for job in initial_jobs]
        if len(initial_jobs) != total_slots or sorted(initial_slots) != list(
            range(total_slots)
        ):
            raise ValueError(
                "Streaming training requires exactly one initial job for every "
                f"quota slot [0, {total_slots}); got slots={initial_slots}"
            )

        ready_jobs: deque[TrainingGroupJob] = deque(initial_jobs)
        available_workers = deque(range(len(self.agent_loop_workers)))
        pending: set[asyncio.Task] = set()
        active_slots: set[int] = set()
        completed_slots: set[int] = set()
        worker_count = len(self.agent_loop_workers)
        submitted_groups = 0
        completed_groups = 0
        peak_in_flight = 0
        metric_records: list[dict[str, Any]] = []
        sample_lengths: list[tuple[int, int]] = []
        started_at = time.perf_counter()
        show_progress = bool(
            getattr(self, "config", {}).get("visharness", {}).get(
                "rollout_progress", False
            )
        )

        async def run_job(
            worker_index: int,
            slot_index: int,
            context: Any,
            batch: DataProto,
        ):
            output = await self.agent_loop_workers[worker_index].generate_sequences.remote(
                batch
            )
            return worker_index, slot_index, context, output

        def submit_job(worker_index: int, job: TrainingGroupJob) -> None:
            nonlocal submitted_groups, peak_in_flight
            slot_index, batch, context = job
            slot_index = int(slot_index)
            if slot_index < 0 or slot_index >= total_slots:
                raise ValueError(
                    f"Training rollout slot {slot_index} is outside [0, {total_slots})"
                )
            if slot_index in completed_slots:
                raise RuntimeError(
                    f"Training rollout attempted to refill completed slot {slot_index}"
                )
            if slot_index in active_slots:
                raise RuntimeError(
                    f"Training rollout slot {slot_index} already has an active group"
                )
            if len(batch) != trajectories_per_group:
                raise RuntimeError(
                    "Each streaming training job must contain one complete prompt "
                    f"group of {trajectories_per_group} trajectories, got {len(batch)}"
                )
            active_slots.add(slot_index)
            task = asyncio.create_task(
                run_job(worker_index, slot_index, context, batch)
            )
            pending.add(task)
            submitted_groups += 1
            peak_in_flight = max(peak_in_flight, len(pending))

        if show_progress:
            print(
                "[streaming rollout] started "
                f"0/{total_slots} accepted prompt groups, "
                f"in_flight={min(total_slots, worker_count)}, workers={worker_count}",
                flush=True,
            )

        try:
            while ready_jobs or pending:
                while ready_jobs and available_workers:
                    submit_job(
                        available_workers.popleft(),
                        ready_jobs.popleft(),
                    )
                if not pending:
                    raise RuntimeError(
                        "Streaming training has queued jobs but no runnable worker"
                    )

                done, still_pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                pending = set(still_pending)

                for task in done:
                    worker_index, slot_index, context, output = await task
                    available_workers.append(worker_index)
                    active_slots.remove(slot_index)
                    completed_groups += 1

                    raw_metrics = output.meta_info.pop("metrics", None)
                    if raw_metrics is None:
                        raise RuntimeError(
                            "Streaming training worker output has no per-trajectory metrics"
                        )
                    if not isinstance(raw_metrics, list):
                        raise TypeError(
                            "Streaming training worker metrics must be a list, got "
                            f"{type(raw_metrics).__name__}"
                        )
                    if len(raw_metrics) != len(output):
                        raise RuntimeError(
                            "Streaming training worker metrics do not match output: "
                            f"metrics={len(raw_metrics)}, trajectories={len(output)}"
                        )
                    metric_records.extend(raw_metrics)

                    prompt_width = output.batch["prompts"].shape[1]
                    if "attention_mask" in output.batch:
                        attention_mask = output.batch["attention_mask"]
                        for sample_index in range(len(output)):
                            sample_lengths.append(
                                (
                                    int(
                                        attention_mask[sample_index, :prompt_width]
                                        .sum()
                                        .item()
                                    ),
                                    int(
                                        attention_mask[sample_index, prompt_width:]
                                        .sum()
                                        .item()
                                    ),
                                )
                            )
                    else:
                        sample_lengths.extend(
                            [(int(prompt_width), 0)] * len(output)
                        )

                    replacement = on_complete(slot_index, context, output)
                    if replacement is None:
                        completed_slots.add(slot_index)
                    else:
                        replacement_slot = int(replacement[0])
                        if replacement_slot != slot_index:
                            raise RuntimeError(
                                "A failed training prompt group must refill the same "
                                f"quota slot: completed={slot_index}, replacement={replacement_slot}"
                            )
                        ready_jobs.append(replacement)

                if show_progress and (
                    len(completed_slots) == total_slots
                    or completed_groups % progress_interval == 0
                    or not pending
                ):
                    elapsed = time.perf_counter() - started_at
                    print(
                        "[streaming rollout] "
                        f"accepted={len(completed_slots)}/{total_slots}, "
                        f"completed_attempts={completed_groups}, "
                        f"submitted_attempts={submitted_groups}, "
                        f"in_flight={len(pending)}, elapsed={elapsed:.1f}s",
                        flush=True,
                    )
        finally:
            if pending:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

        if len(completed_slots) != total_slots:
            raise RuntimeError(
                "Streaming training stopped before all quota slots were accepted: "
                f"accepted={sorted(completed_slots)}, total_slots={total_slots}"
            )

        elapsed = time.perf_counter() - started_at
        return {
            "completed_slots": float(len(completed_slots)),
            "submitted_groups": float(submitted_groups),
            "completed_groups": float(completed_groups),
            "peak_in_flight_groups": float(peak_in_flight),
            "worker_count": float(worker_count),
            "wall_seconds": float(elapsed),
            "timing": self._streaming_performance_metrics(
                metric_records,
                sample_lengths,
            ),
        }

    @auto_await
    async def generate_validation_sequences(
        self,
        jobs: Iterable[ValidationJob],
        *,
        max_in_flight: int,
        on_complete: ValidationCallback,
        total_jobs: int | None = None,
        progress_interval: int = 10,
    ) -> dict[str, float]:
        """Run one-sample validation jobs with immediate slot refill.

        Unlike the inherited ``generate_sequences`` method, this method does
        not wait for a fixed validation batch. Each completed trajectory frees
        one slot and the next job is sent immediately. Ordinary training
        rollout continues to use the inherited method without modification.
        """

        if max_in_flight <= 0:
            raise ValueError(f"max_in_flight must be positive, got {max_in_flight}")
        if progress_interval <= 0:
            raise ValueError(f"progress_interval must be positive, got {progress_interval}")
        if not self.agent_loop_workers:
            raise RuntimeError("No agent-loop workers are available for validation")

        job_iterator = iter(jobs)
        pending: set[asyncio.Task] = set()
        worker_count = len(self.agent_loop_workers)
        completed = 0
        submitted = 0
        peak_in_flight = 0
        exhausted = False
        started_at = time.perf_counter()

        async def run_job(
            worker_index: int,
            sequence_index: int,
            context: Any,
            batch: DataProto,
        ):
            output = await self.agent_loop_workers[worker_index].generate_sequences.remote(batch)
            return worker_index, sequence_index, context, output

        def submit_next(worker_index: int) -> bool:
            nonlocal submitted, peak_in_flight, exhausted
            if exhausted:
                return False
            try:
                sequence_index, batch, context = next(job_iterator)
            except StopIteration:
                exhausted = True
                return False
            task = asyncio.create_task(run_job(worker_index, sequence_index, context, batch))
            pending.add(task)
            submitted += 1
            peak_in_flight = max(peak_in_flight, len(pending))
            return True

        for slot_index in range(max_in_flight):
            if not submit_next(slot_index % worker_count):
                break

        total_label = str(total_jobs) if total_jobs is not None else "?"
        print(
            f"[validation progress] started 0/{total_label} trajectories, "
            f"max_in_flight={max_in_flight}, workers={worker_count}",
            flush=True,
        )

        try:
            while pending:
                done, still_pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                pending = set(still_pending)

                completed_outputs = []
                for task in done:
                    worker_index, sequence_index, context, output = await task
                    completed_outputs.append((worker_index, sequence_index, context, output))

                # Refill every freed worker slot before doing driver-side reward
                # aggregation, so the rollout replicas do not wait for it.
                for worker_index, _, _, _ in completed_outputs:
                    submit_next(worker_index)

                for _, sequence_index, context, output in completed_outputs:
                    # The standard batch manager consumes this worker-only
                    # field before returning. Do the same here so it cannot
                    # collide with the original sample metadata during union.
                    output.meta_info.pop("metrics", None)
                    on_complete(sequence_index, context, output)
                    completed += 1

                if (
                    completed == total_jobs
                    or completed % progress_interval == 0
                    or not pending
                ):
                    elapsed = time.perf_counter() - started_at
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    remaining = (
                        max(total_jobs - completed, 0)
                        if total_jobs is not None
                        else max(submitted - completed, 0)
                    )
                    eta = remaining / rate if rate > 0 else 0.0
                    print(
                        f"[validation progress] completed {completed}/{total_label}, "
                        f"in_flight={len(pending)}, elapsed={elapsed:.1f}s, "
                        f"rate={rate:.2f} traj/s, eta={eta:.1f}s",
                        flush=True,
                    )
        finally:
            if pending:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

        if total_jobs is not None and completed != total_jobs:
            raise RuntimeError(
                f"Dynamic validation completed {completed} trajectories, expected {total_jobs}"
            )

        elapsed = time.perf_counter() - started_at
        return {
            "completed": float(completed),
            "submitted": float(submitted),
            "peak_in_flight": float(peak_in_flight),
            "max_in_flight": float(max_in_flight),
            "worker_count": float(worker_count),
            "wall_seconds": float(elapsed),
        }
