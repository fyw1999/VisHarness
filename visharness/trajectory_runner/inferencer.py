"""Multi-threaded VisHarness trajectory inference/data-generation loop."""

from __future__ import annotations

import copy
import json
import logging
import msgpack
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from statistics import mean
from typing import Any
from uuid import uuid4

from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from visharness.agent_loop.action_parser import ParsedAction, parse_action
from visharness.agent_loop.action_validator import validate_and_prepare_tool_call
from visharness.agent_loop.tool_response_processor import (
    ProcessedToolResponse,
    append_tool_response_guidance,
    archive_visible_images,
    build_error_observation,
    normalize_tool_response_closing_spacing,
    process_tool_response,
)
from visharness.agent_loop.trajectory_state import VisionTrajectoryState
from visharness.tools.errors import ToolOOMRetriesExhaustedError
from visharness.prompts import (
    PHRASE_TO_BOXMASK_PROMPT,
    PHRASE_TO_POINT_PROMPT,
    POINT_TO_BOXMASK_PROMPT,
    SPLIT_PROMPT,
    SR_PROMPT,
    VISION_RESULT_TOOLS,
    VISION_TOOL_PROMPT,
)

from .benchmark import InferenceResourceMonitor
from .errors import (
    TrajectoryContextLengthExceededError,
    TrajectoryModelRequestTimeoutError,
    TrajectoryVisionEncoderCacheExceededError,
)
from .item import TrajectoryItem
from .model_client import ModelTurnResponse
from .openai_action_parser import (
    OpenAIToolCall,
    SUBMIT_FINAL_ANSWER_TOOL,
    openai_message_tool_calls,
    parse_openai_action,
)
from .serializer import TrajectorySerializer, strip_tool_response_wrapper

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(
        ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
    )


def _strip_trailing_linebreaks(content: Any) -> Any:
    """Remove line breaks that the API tool-message template will add."""
    if isinstance(content, str):
        return content.rstrip("\r\n")
    if isinstance(content, list):
        for part in reversed(content):
            if (
                isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            ):
                part["text"] = part["text"].rstrip("\r\n")
                break
    return content


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "sample_count": len(values),
        "mean": float(mean(values)) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "min": float(min(values)) if values else None,
        "max": float(max(values)) if values else None,
        "sum": float(sum(values)) if values else None,
    }


def _response_tool_calls(response: ModelTurnResponse) -> list[OpenAIToolCall]:
    message = response.api_message
    if message is None:
        message = response.assistant_message
    return openai_message_tool_calls(message)


class SyncToolCaller:
    """Synchronous wrapper around the existing controller-backed ToolManager."""

    def __init__(
        self,
        controller_url_location: str | None = None,
        max_consecutive_oom: int = 5,
    ):
        from tool_server.tool_workers.tool_manager.base_manager import ToolManager

        self.manager = ToolManager(
            controller_url_location=controller_url_location,
            max_consecutive_oom=max_consecutive_oom,
        )
        self.available_tools = list(self.manager.available_tools)

    def call(self, tool_name: str, tool_parameters: dict[str, Any]) -> Any:
        payload = msgpack.packb(tool_parameters, use_bin_type=True)
        response = self.manager.dynamic_call_tool(tool_name, payload)
        if response.get("status") != "success":
            if response.get("error_type") == "tool_oom_retry_exhausted":
                raise ToolOOMRetriesExhaustedError(response)
            raise RuntimeError(f"Visual tool {tool_name} failed: {response.get('message', response)!r}")
        return copy.deepcopy(response["results"])


def _identity_collate(data):
    return data[0]


# Compatibility alias for callers that imported the old private helper. The
# implementation belongs to the trajectory runner, not the RL text parser.
_parse_output_format = parse_openai_action


class BaseTrajectoryInferencer:
    def __init__(
        self,
        tp_model: Any = None,
        batch_size: int = 1,
        max_rounds: int = 3,
        mode: str = "inference",
        model_family: str = "qwen3_vl",
        archive_previous_images: bool = True,
        serializer: TrajectorySerializer | None = None,
        tool_caller: Any = None,
        controller_url_location: str | None = None,
        max_consecutive_oom: int = 5,
        benchmark_config: dict[str, Any] | None = None,
    ):
        self.tp_model = tp_model
        self.batch_size = int(batch_size)
        self.max_rounds = int(max_rounds)
        self.mode = str(mode).lower()
        if self.mode not in {"inference", "data_generation"}:
            raise ValueError("mode should be either data_generation or inference")
        self.model_family = model_family
        self.archive_previous_images = archive_previous_images
        self.serializer = serializer or TrajectorySerializer(None)
        self.benchmark_config = dict(benchmark_config or {})
        self._benchmark_run_started_perf_counter: float | None = None
        self.tool_caller = tool_caller or SyncToolCaller(
            controller_url_location=controller_url_location,
            max_consecutive_oom=max_consecutive_oom,
        )
        self.available_tools = list(getattr(self.tool_caller, "available_tools", []))

    def _make_item(self, meta_data: dict[str, Any]) -> TrajectoryItem:
        image = meta_data["image"]
        if not isinstance(image, Image.Image):
            raise TypeError("Trajectory runner expects dataset items to contain a PIL image under 'image'")
        conversation = self.tp_model.generate_conversation_fn(meta_data["question"], image)
        state = VisionTrajectoryState.from_initial_image(image)
        return TrajectoryItem(
            max_rounds=self.max_rounds,
            current_round=0,
            meta_data=meta_data,
            conversation=conversation,
            state=state,
            status="processing",
            current_images=[image],
            current_image_names=["img_0"],
        )

    def _add_tool_result_guidance(
        self,
        observation: ProcessedToolResponse,
        tool_name: str | None,
        original_prompt: str,
    ) -> ProcessedToolResponse:
        if self.mode == "inference":
            if tool_name in VISION_RESULT_TOOLS:
                append_tool_response_guidance(observation, VISION_TOOL_PROMPT)
            return observation

        guidance_parts = []
        if tool_name in VISION_RESULT_TOOLS:
            guidance_parts.append(VISION_TOOL_PROMPT)
            if tool_name == "PhraseToBoxMask":
                guidance_parts.append(PHRASE_TO_BOXMASK_PROMPT)
            elif tool_name == "PhraseToPoint":
                guidance_parts.append(PHRASE_TO_POINT_PROMPT)
            elif tool_name == "PointToBoxMask":
                guidance_parts.append(POINT_TO_BOXMASK_PROMPT)
        elif tool_name == "SplitImageIntoPatches":
            guidance_parts.append(SPLIT_PROMPT)
        elif tool_name == "SuperResolution":
            guidance_parts.append(SR_PROMPT)
        guidance_parts.append(
            f"Continue the reasoning process to answer the original question: {original_prompt}\n"
        )
        append_tool_response_guidance(observation, "".join(guidance_parts))
        return observation

    def _message_for_observation(
        self,
        observation: ProcessedToolResponse,
        turn_record: dict[str, Any],
    ) -> dict[str, Any]:
        normalize_tool_response_closing_spacing(observation)
        message = copy.deepcopy(observation.message)
        unwrapped_content = strip_tool_response_wrapper(message.get("content"))
        if unwrapped_content is None:
            raise ValueError(
                "Internal trajectory-runner error: every environment observation must contain exactly one outer "
                "<tool_response>...</tool_response> wrapper before it is adapted to the model-facing protocol."
            )

        if self.mode != "data_generation":
            if message.get("role") != "user":
                raise ValueError(
                    "Internal trajectory-runner error: inference environment observations must use role=user."
                )
            return message

        message["role"] = "tool"
        message["tool_call_id"] = turn_record.get("tool_call_id")
        message["content"] = _strip_trailing_linebreaks(unwrapped_content)
        return message

    def _append_observation(
        self,
        item: TrajectoryItem,
        observation: ProcessedToolResponse,
        turn_record: dict[str, Any],
    ) -> None:
        is_error_observation = bool(getattr(observation, "is_error", False))
        should_archive_previous = (
            self.archive_previous_images and bool(item.current_image_names) and not is_error_observation
        )
        if should_archive_previous:
            archive_visible_images(item.conversation, item.current_image_names)

        observation_message_index = len(item.conversation)
        item.conversation.append(self._message_for_observation(observation, turn_record))
        if self.archive_previous_images:
            if not is_error_observation:
                item.current_images = list(observation.images)
                item.current_image_names = list(observation.image_names)
        else:
            item.current_images = item.current_images + list(observation.images)
            item.current_image_names = item.current_image_names + list(observation.image_names)

        turn_record["observation_appended"] = True
        turn_record.setdefault("observation_message_indices", []).append(
            observation_message_index
        )
        turn_record["visible_image_names_after_step"] = list(item.current_image_names)

    def _new_turn_record(
        self,
        item: TrajectoryItem,
        turn_index: int,
        response: ModelTurnResponse,
        action: Any,
        assistant_message_index: int,
    ) -> dict[str, Any]:
        api_tool_calls = _response_tool_calls(response)
        submit_call = next(
            (
                tool_call
                for tool_call in api_tool_calls
                if tool_call.function_name == SUBMIT_FINAL_ANSWER_TOOL
            ),
            None,
        )
        primary_tool_call = api_tool_calls[0] if len(api_tool_calls) == 1 else None
        if (
            self.mode == "data_generation"
            and primary_tool_call is not None
            and primary_tool_call.call_id is None
        ):
            raise ValueError(
                "Data-generation tool calls require tool_call_id, but the model response omitted it."
            )
        submit_attempt = submit_call is not None
        recorded_completion = (
            response.completion if response.used_token_completion else None
        )
        return {
            "turn_index": int(turn_index),
            "completion": recorded_completion,
            "finish_reason": response.finish_reason,
            "backend_stop_reason": response.stop_reason,
            "stop_reason": response.effective_stop_reason,
            "turn_truncated_by_length": response.truncated_by_length,
            "turn_max_tokens": response.turn_max_tokens,
            "turn_response_length": response.completion_token_count,
            "model_latency_seconds": response.model_latency_seconds,
            "prompt_token_count": response.prompt_token_count,
            "visual_token_count": response.visual_token_count,
            "used_token_completion": response.used_token_completion,
            "token_ids": list(response.token_ids) if response.token_ids is not None else None,
            "action_type": action.action_type,
            "tool_name": action.tool_name,
            "tool_arguments": copy.deepcopy(action.arguments) if action.arguments is not None else None,
            "answer": action.answer,
            "api_tool_calls": [
                {
                    "id": tool_call.call_id,
                    "type": tool_call.call_type,
                    "function_name": tool_call.function_name,
                    "raw_arguments": copy.deepcopy(tool_call.raw_arguments),
                }
                for tool_call in api_tool_calls
            ],
            "api_tool_call_count": len(api_tool_calls),
            "tool_call_id": (
                primary_tool_call.call_id
                if primary_tool_call is not None
                else None
            ),
            "submit_final_answer_attempt": submit_attempt,
            "assistant_message_index": int(assistant_message_index),
            "observation_message_indices": [],
            "observation_appended": False,
            "output_format_success": action.action_type != "invalid",
            "sft_eligible": False,
            "exclude_from_sft_history": (
                submit_attempt and action.action_type == "invalid"
            ),
            "tool_args_success": False,
            "tool_execution_success": False,
            "tool_execution_error_type": None,
            "tool_oom_attempts": 0,
            "tool_oom_worker_history": [],
            "action_error": action.error,
            "tool_args_error": None,
            "tool_execution_error": None,
            "tool_latency_seconds": 0.0,
            "visible_image_names": list(item.current_image_names),
            "final_results_before_turn": copy.deepcopy(item.state.final_results),
        }

    def _parse_model_action(self, response: ModelTurnResponse) -> ParsedAction:
        effective_stop_reason = response.effective_stop_reason
        if self.mode == "inference":
            if not response.used_token_completion or not isinstance(response.completion, str):
                raise RuntimeError(
                    "Internal inference error: inference mode requires a token-decoded completion."
                )
            return parse_action(
                response.completion,
                model_family=self.model_family,
                finish_reason=effective_stop_reason,
                max_output_tokens=response.turn_max_tokens,
            )
        if self.mode == "data_generation":
            if response.api_message is None:
                raise RuntimeError(
                    "Internal data-generation error: the structured API message is missing."
                )
            return parse_openai_action(
                response.api_message,
                finish_reason=effective_stop_reason,
                max_output_tokens=response.turn_max_tokens,
            )
        raise RuntimeError(
            f"Internal inference error: unsupported mode {self.mode!r}."
        )

    def _conversation_message_for_response(
        self,
        response: ModelTurnResponse,
    ) -> Any:
        if self.mode == "data_generation":
            if response.api_message is None:
                raise RuntimeError(
                    "Internal data-generation error: the structured API message is missing."
                )
            return response.api_message
        if response.assistant_message is not None:
            return copy.deepcopy(response.assistant_message)
        if isinstance(response.completion, str):
            return {"role": "assistant", "content": response.completion}
        raise RuntimeError(
            "Internal inference error: the assistant response is missing."
        )

    def _save_valid_turn(
        self,
        item: TrajectoryItem,
        turn_record: dict[str, Any],
        turn_index: int,
    ) -> None:
        turn_record["sft_eligible"] = True
        if self.serializer.save_trajectory:
            self.serializer.save_item_trajectory(
                item,
                step_index=turn_index,
                data_generation=self.mode == "data_generation",
            )

    @staticmethod
    def _abort_invalid_trajectory(
        item: TrajectoryItem,
        *,
        reason: str,
        stage: str,
        turn_index: int,
        error: Exception,
    ) -> TrajectoryItem:
        error_text = str(error)
        item.status = "failed"
        item.error = f"{type(error).__name__}: {error_text}"
        item.trajectory_invalid = True
        item.invalid_reason = reason
        item.invalid_stage = stage
        item.invalid_turn_index = int(turn_index)
        item.invalid_error = error_text
        logger.warning(
            "Trajectory %s aborted without stopping the run: reason=%s stage=%s turn=%s error=%s",
            item.meta_data.get("id"),
            reason,
            stage,
            turn_index,
            error_text,
        )
        return item

    def process_single_trajectory(self, item: TrajectoryItem) -> TrajectoryItem:
        trajectory_started = time.perf_counter()
        run_started = self._benchmark_run_started_perf_counter
        efficiency_metrics: dict[str, Any] = {
            "schema_version": 1,
            "archive_previous_images": bool(self.archive_previous_images),
            "trajectory_started_since_run_seconds": (
                max(trajectory_started - run_started, 0.0)
                if run_started is not None
                else None
            ),
            "model_call_count": 0,
            "successful_model_call_count": 0,
            "tool_call_count": 0,
            "successful_tool_call_count": 0,
            "llm_generate_seconds": 0.0,
            "tool_seconds": 0.0,
            "cumulative_prompt_tokens": 0,
            "cumulative_visual_tokens": 0,
            "cumulative_completion_tokens": 0,
            "max_visual_tokens_per_call": 0,
            "prompt_token_metrics_missing_calls": 0,
            "visual_token_metrics_missing_calls": 0,
            "completion_token_metrics_missing_calls": 0,
        }
        item.efficiency_metrics = efficiency_metrics
        try:
            for turn_index in range(1, item.max_rounds + 1):
                efficiency_metrics["model_call_count"] += 1
                model_started = time.perf_counter()
                try:
                    response = self.tp_model.generate_one_item(item)
                finally:
                    model_latency = time.perf_counter() - model_started
                    efficiency_metrics["llm_generate_seconds"] += model_latency
                response.model_latency_seconds = float(model_latency)
                efficiency_metrics["successful_model_call_count"] += 1
                for response_value, total_key, missing_key in (
                    (
                        response.prompt_token_count,
                        "cumulative_prompt_tokens",
                        "prompt_token_metrics_missing_calls",
                    ),
                    (
                        response.visual_token_count,
                        "cumulative_visual_tokens",
                        "visual_token_metrics_missing_calls",
                    ),
                    (
                        response.completion_token_count,
                        "cumulative_completion_tokens",
                        "completion_token_metrics_missing_calls",
                    ),
                ):
                    if response_value is None:
                        efficiency_metrics[missing_key] += 1
                    else:
                        efficiency_metrics[total_key] += int(response_value)
                if response.visual_token_count is not None:
                    efficiency_metrics["max_visual_tokens_per_call"] = max(
                        int(efficiency_metrics["max_visual_tokens_per_call"]),
                        int(response.visual_token_count),
                    )
                conversation_message = self._conversation_message_for_response(response)
                assistant_message_index = len(item.conversation)
                item.conversation.append(conversation_message)
                item.current_round = turn_index

                action = self._parse_model_action(response)
                turn_record = self._new_turn_record(
                    item,
                    turn_index,
                    response,
                    action,
                    assistant_message_index,
                )
                item.state.turns.append(turn_record)

                if action.action_type == "answer":
                    turn_record["tool_args_success"] = True
                    self._save_valid_turn(item, turn_record, turn_index)
                    item.answer = action.answer
                    item.state.finished = True
                    item.status = "finished"
                    break

                if action.action_type == "invalid":
                    observation = build_error_observation(action.error or "Incorrect output format.")
                    self._append_observation(item, observation, turn_record)
                    continue

                validation = validate_and_prepare_tool_call(
                    action=action,
                    state=item.state,
                    available_tools=self.available_tools,
                )
                if not validation.is_valid:
                    turn_record["tool_args_error"] = validation.error
                    item.tool_response.append(validation.error)
                    observation = build_error_observation(validation.error or "Incorrect tool call parameters.")
                    self._append_observation(item, observation, turn_record)
                    continue

                turn_record["tool_args_success"] = True
                self._save_valid_turn(item, turn_record, turn_index)
                validated_call = validation.call
                efficiency_metrics["tool_call_count"] += 1
                tool_started = time.perf_counter()
                try:
                    tool_result = self.tool_caller.call(validated_call.tool_name, validated_call.tool_parameters)
                    item.tool_response.append(tool_result)
                    observation = process_tool_response(
                        tool_name=validated_call.tool_name,
                        tool_response=tool_result,
                        state=item.state,
                        turn_index=turn_index,
                    )
                except ToolOOMRetriesExhaustedError as exc:
                    turn_record["tool_execution_error_type"] = (
                        "tool_oom_retry_exhausted"
                    )
                    turn_record["tool_execution_error"] = str(exc)
                    turn_record["tool_oom_attempts"] = int(exc.oom_attempts)
                    turn_record["tool_oom_worker_history"] = copy.deepcopy(
                        exc.worker_history
                    )
                    raise
                except Exception as exc:
                    turn_record["tool_execution_error_type"] = type(exc).__name__
                    turn_record["tool_execution_error"] = str(exc)
                    raise
                finally:
                    tool_latency = time.perf_counter() - tool_started
                    turn_record["tool_latency_seconds"] = float(tool_latency)
                    efficiency_metrics["tool_seconds"] += tool_latency

                turn_record["tool_execution_success"] = True
                efficiency_metrics["successful_tool_call_count"] += 1
                turn_record["tool_result_summary"] = copy.deepcopy(observation.result_summary)
                turn_record["artifact_events"] = copy.deepcopy(observation.artifact_events)
                observation = self._add_tool_result_guidance(
                    observation,
                    tool_name=validated_call.tool_name,
                    original_prompt=item.meta_data.get("question", ""),
                )
                self._append_observation(item, observation, turn_record)

            # ``status`` describes the processing lifecycle: exhausting the
            # configured turn budget is still a finished processing attempt.
            # ``TrajectoryItem.termination_reason`` separately distinguishes
            # this case from an explicit final answer.
            if item.status != "finished":
                item.status = "finished"
            return item
        except ToolOOMRetriesExhaustedError as exc:
            return self._abort_invalid_trajectory(
                item,
                reason="tool_oom_retry_exhausted",
                stage="tool_execution",
                turn_index=max(int(item.current_round), 1),
                error=exc,
            )
        except TrajectoryContextLengthExceededError as exc:
            return self._abort_invalid_trajectory(
                item,
                reason="context_length_exceeded",
                stage="model_generation",
                turn_index=int(item.current_round) + 1,
                error=exc,
            )
        except TrajectoryModelRequestTimeoutError as exc:
            return self._abort_invalid_trajectory(
                item,
                reason="model_request_timeout",
                stage="model_generation",
                turn_index=int(item.current_round) + 1,
                error=exc,
            )
        except TrajectoryVisionEncoderCacheExceededError as exc:
            return self._abort_invalid_trajectory(
                item,
                reason="vision_encoder_cache_exceeded",
                stage="model_generation",
                turn_index=int(item.current_round) + 1,
                error=exc,
            )
        except Exception:
            logger.error(
                "Fatal trajectory-runner error for sample %s",
                item.meta_data.get("id"),
            )
            logger.error(traceback.format_exc())
            raise
        finally:
            trajectory_completed = time.perf_counter()
            successful_model_calls = int(
                efficiency_metrics["successful_model_call_count"]
            )
            for total_key, missing_key in (
                (
                    "cumulative_prompt_tokens",
                    "prompt_token_metrics_missing_calls",
                ),
                (
                    "cumulative_visual_tokens",
                    "visual_token_metrics_missing_calls",
                ),
                (
                    "cumulative_completion_tokens",
                    "completion_token_metrics_missing_calls",
                ),
            ):
                if (
                    successful_model_calls == 0
                    or int(efficiency_metrics[missing_key]) > 0
                ):
                    efficiency_metrics[total_key] = None
            if (
                successful_model_calls == 0
                or int(
                    efficiency_metrics[
                        "visual_token_metrics_missing_calls"
                    ]
                )
                > 0
            ):
                efficiency_metrics["max_visual_tokens_per_call"] = None
            efficiency_metrics["trajectory_elapsed_seconds"] = float(
                trajectory_completed - trajectory_started
            )
            efficiency_metrics["trajectory_completed_since_run_seconds"] = (
                float(max(trajectory_completed - run_started, 0.0))
                if run_started is not None
                else None
            )
            efficiency_metrics["actual_rounds"] = int(item.current_round)
            efficiency_metrics["trajectory_finished"] = bool(
                item.trajectory_finished
            )
            efficiency_metrics["status"] = item.status

    def parallel_batch_inference(self, dataset: Any) -> list[TrajectoryItem]:
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=_identity_collate)
        all_items = [self._make_item(meta_data) for meta_data in dataloader]
        results: list[TrajectoryItem] = []
        progress_bar = tqdm(total=len(all_items), desc="Model Responding")
        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "-"
            + uuid4().hex[:8]
        )
        resource_monitor = InferenceResourceMonitor(
            self.benchmark_config,
            trace_path=self.serializer.resource_trace_path,
            run_id=run_id,
        )
        resource_monitor.start()
        run_started_at = _utc_now()
        run_started = time.perf_counter()
        self._benchmark_run_started_perf_counter = run_started
        serialization_seconds = 0.0
        try:
            # Serial debug version: comment out the ThreadPoolExecutor block
            # below and uncomment this block when step-by-step debugging is
            # needed. Keep the same fail-fast persistence semantics.
            # for item in all_items:
            #     try:
            #         finished_item = self.process_single_trajectory(item)
            #         serialization_started = time.perf_counter()
            #         self.serializer.store_result(
            #             finished_item.result_dict(),
            #         )
            #         serialization_seconds += (
            #             time.perf_counter() - serialization_started
            #         )
            #         results.append(finished_item)
            #     finally:
            #         progress_bar.update(1)

            with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
                future_to_item = {
                    executor.submit(self.process_single_trajectory, item): item
                    for item in all_items
                }
                try:
                    for future in as_completed(future_to_item):
                        try:
                            finished_item = future.result()
                            serialization_started = time.perf_counter()
                            self.serializer.store_result(
                                finished_item.result_dict(),
                            )
                            serialization_seconds += (
                                time.perf_counter() - serialization_started
                            )
                            results.append(finished_item)
                        finally:
                            progress_bar.update(1)
                except BaseException:
                    for pending_future in future_to_item:
                        pending_future.cancel()
                    raise
        finally:
            end_to_end_wall_seconds = time.perf_counter() - run_started
            self._benchmark_run_started_perf_counter = None
            resource_summary = resource_monitor.stop()
            progress_bar.close()

        completion_offsets = [
            float(
                item.efficiency_metrics[
                    "trajectory_completed_since_run_seconds"
                ]
            )
            for item in results
            if item.efficiency_metrics.get(
                "trajectory_completed_since_run_seconds"
            )
            is not None
        ]
        rollout_wall_seconds = (
            max(completion_offsets) if completion_offsets else 0.0
        )
        trajectory_metric_fields = (
            "trajectory_elapsed_seconds",
            "llm_generate_seconds",
            "tool_seconds",
            "cumulative_prompt_tokens",
            "cumulative_visual_tokens",
            "cumulative_completion_tokens",
            "max_visual_tokens_per_call",
            "model_call_count",
            "tool_call_count",
        )
        trajectory_metric_summary = {
            field_name: _numeric_summary(
                [
                    float(item.efficiency_metrics[field_name])
                    for item in results
                    if item.efficiency_metrics.get(field_name) is not None
                ]
            )
            for field_name in trajectory_metric_fields
        }
        processed_count = len(results)
        successful_answer_count = sum(
            item.trajectory_finished for item in results
        )
        invalid_reason_counts: dict[str, int] = {}
        for item in results:
            if not item.trajectory_invalid:
                continue
            reason = item.invalid_reason or "unknown"
            invalid_reason_counts[reason] = (
                invalid_reason_counts.get(reason, 0) + 1
            )
        run_summary = {
            "schema_version": 1,
            "run_id": run_id,
            "run_started_at": run_started_at,
            "run_finished_at": _utc_now(),
            "mode": self.mode,
            "model_name": getattr(self.tp_model, "model_name", None),
            "prediction_checkpoint_path": (
                str(self.serializer.save_ckpt_path.resolve())
                if self.serializer.save_ckpt_path is not None
                else None
            ),
            "archive_previous_images": bool(self.archive_previous_images),
            "batch_size": int(self.batch_size),
            "max_rounds": int(self.max_rounds),
            "generation_config": copy.deepcopy(
                getattr(self.tp_model, "generation_config", {})
            ),
            "metric_definitions": {
                "trajectory_elapsed_seconds": (
                    "Active wall time from a trajectory worker starting until "
                    "that trajectory finishes; checkpoint serialization is "
                    "excluded."
                ),
                "cumulative_visual_tokens": (
                    "Sum across all model calls in a trajectory of the "
                    "Qwen <|image_pad|> tokens in returned prompt_token_ids."
                ),
                "rollout_wall_seconds": (
                    "Wall time from dispatching the run until the final "
                    "trajectory worker completes; checkpoint serialization is "
                    "excluded using worker completion timestamps."
                ),
                "rollout_throughput": (
                    "Processed trajectory attempts divided by "
                    "rollout_wall_seconds; failed attempts remain included."
                ),
                "gpu_peak_memory": (
                    "Maximum NVML device memory used over configured serving "
                    "GPUs during resource sampling."
                ),
                "peak_active_kv_cache_memory": (
                    "Peak vLLM KV-cache utilization multiplied by the "
                    "allocated KV-cache pool capacity per serving GPU."
                ),
            },
            "visual_token_id": getattr(
                self.tp_model,
                "_image_token_id",
                None,
            ),
            "attempted_trajectories": len(all_items),
            "processed_trajectories": processed_count,
            "explicit_answer_trajectories": successful_answer_count,
            "failed_trajectories": sum(
                item.status == "failed" for item in results
            ),
            "invalid_trajectories": sum(
                item.trajectory_invalid for item in results
            ),
            "invalid_reason_counts": invalid_reason_counts,
            "rollout_wall_seconds": float(rollout_wall_seconds),
            "serialization_wall_seconds": float(serialization_seconds),
            "end_to_end_wall_seconds": float(end_to_end_wall_seconds),
            "rollout_throughput_trajectories_per_second": (
                processed_count / rollout_wall_seconds
                if rollout_wall_seconds > 0
                else 0.0
            ),
            "rollout_throughput_trajectories_per_minute": (
                processed_count * 60.0 / rollout_wall_seconds
                if rollout_wall_seconds > 0
                else 0.0
            ),
            "end_to_end_throughput_trajectories_per_minute": (
                processed_count * 60.0 / end_to_end_wall_seconds
                if end_to_end_wall_seconds > 0
                else 0.0
            ),
            "trajectory_metrics": trajectory_metric_summary,
            "resource_metrics": resource_summary,
        }
        self.serializer.store_benchmark_summary(run_summary)
        benchmark_preview = {
            "processed_trajectories": processed_count,
            "average_trajectory_elapsed_seconds": (
                trajectory_metric_summary["trajectory_elapsed_seconds"]["mean"]
            ),
            "p95_trajectory_elapsed_seconds": (
                trajectory_metric_summary["trajectory_elapsed_seconds"]["p95"]
            ),
            "average_cumulative_visual_tokens": (
                trajectory_metric_summary["cumulative_visual_tokens"]["mean"]
            ),
            "rollout_throughput_trajectories_per_minute": (
                run_summary[
                    "rollout_throughput_trajectories_per_minute"
                ]
            ),
            "peak_gpu_memory_per_gpu_gib": resource_summary.get(
                "peak_gpu_memory_per_gpu_gib"
            ),
            "incremental_peak_gpu_memory_per_gpu_gib": (
                resource_summary.get(
                    "incremental_peak_gpu_memory_per_gpu_gib"
                )
            ),
            "peak_kv_cache_usage": resource_summary.get(
                "peak_kv_cache_usage"
            ),
            "peak_active_kv_cache_memory_per_gpu_gib": (
                resource_summary.get(
                    "peak_active_kv_cache_memory_per_gpu_gib"
                )
            ),
        }
        logger.info(
            "Efficiency benchmark preview:\n%s",
            json.dumps(benchmark_preview, ensure_ascii=False, indent=2),
        )
        return results
