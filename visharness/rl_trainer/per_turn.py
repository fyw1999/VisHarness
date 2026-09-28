"""Convert complete trajectories into the exact per-turn samples seen during rollout."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.utils.tokenizer import build_multimodal_processor_inputs, get_processor_token_id

from .per_turn_loss import TRAJECTORY_LOSS_WEIGHT_KEY


SUPPORTED_LOCAL_COST_MODES = frozenset(
    {
        "trajectory_weighted_additive",
        "per_event_floor",
    }
)


class NoTrainablePerTurnSamplesError(ValueError):
    """Raised when rollout succeeded but every per-turn training sample is filtered."""


def _object_array(values: list[Any]) -> np.ndarray:
    array = np.empty(len(values), dtype=object)
    array[:] = values
    return array


def _pad_tokens(token_ids: list[int], width: int, pad_token_id: int, *, left: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if len(token_ids) > width:
        side = "prompt" if left else "response"
        raise ValueError(f"Per-turn {side} has {len(token_ids)} tokens, exceeding configured width {width}")

    ids = torch.full((width,), pad_token_id, dtype=torch.long)
    mask = torch.zeros((width,), dtype=torch.long)
    if token_ids:
        values = torch.tensor(token_ids, dtype=torch.long)
        if left:
            ids[-len(token_ids) :] = values
            mask[-len(token_ids) :] = 1
        else:
            ids[: len(token_ids)] = values
            mask[: len(token_ids)] = 1
    return ids, mask


def _build_multi_modal_inputs(
    *,
    tokenizer,
    processor,
    input_ids: torch.Tensor,
    multi_modal_data: dict[str, Any],
    mm_processor_kwargs: dict[str, Any] | None,
) -> dict[str, torch.Tensor]:
    if processor is None:
        return {}

    current_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    processor_inputs = build_multimodal_processor_inputs(
        processor,
        text=[current_text],
        images=multi_modal_data.get("images") or None,
        videos=multi_modal_data.get("videos") or None,
        audio=multi_modal_data.get("audios") or None,
        mm_processor_kwargs=mm_processor_kwargs,
    )
    processor_inputs.pop("input_ids", None)
    processor_inputs.pop("attention_mask", None)
    if hasattr(processor_inputs, "convert_to_tensors"):
        processor_inputs = processor_inputs.convert_to_tensors("pt")
    multi_modal_inputs = dict(processor_inputs)

    image_grid_thw = multi_modal_inputs.get("image_grid_thw")
    if image_grid_thw is not None:
        multi_modal_inputs["images_seqlens"] = torch.repeat_interleave(
            image_grid_thw[:, 1] * image_grid_thw[:, 2],
            image_grid_thw[:, 0],
        )
    return multi_modal_inputs


def _compute_position_ids(
    *,
    processor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    multi_modal_inputs: dict[str, torch.Tensor],
    prompt_width: int | None = None,
) -> torch.Tensor:
    input_ids = input_ids.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)
    if processor is None:
        return compute_position_id_with_mask(attention_mask).squeeze(0)

    image_grid_thw = multi_modal_inputs.get("image_grid_thw")
    video_grid_thw = multi_modal_inputs.get("video_grid_thw")
    multi_modal_kwargs = {
        "image_grid_thw": image_grid_thw,
        "video_grid_thw": video_grid_thw,
    }
    if multi_modal_inputs.pop("mm_token_type_ids", None) is not None:
        mm_token_type_ids = torch.zeros_like(input_ids)
        image_token_id = get_processor_token_id(processor, "image")
        video_token_id = get_processor_token_id(processor, "video")

        # The sequence here is prompt + response. Qwen3-VL uses modality type
        # ids to consume one image/video grid for every visual-token span. The
        # model may occasionally generate visual special tokens in the response;
        # those are invalid text outputs, not real images with grid metadata. If
        # we mark the full sequence by token id, get_rope_index will try to read
        # a non-existent image_grid_thw entry and crash long-running training.
        prompt_end = input_ids.shape[-1] if prompt_width is None else int(prompt_width)
        prompt_end = max(0, min(prompt_end, input_ids.shape[-1]))
        prompt_ids = input_ids[0, :prompt_end]

        if image_token_id is not None:
            prompt_image_mask = prompt_ids == image_token_id
            if prompt_image_mask.any():
                if image_grid_thw is None:
                    raise ValueError(
                        "Per-turn prompt contains image tokens but image_grid_thw is missing. "
                        "This means the saved turn prompt and multi_modal_data['images'] are inconsistent."
                    )
                prompt_mm_token_type_ids = mm_token_type_ids[0, :prompt_end]
                prompt_mm_token_type_ids[prompt_image_mask] = 1
        if video_token_id is not None:
            prompt_video_mask = prompt_ids == video_token_id
            if prompt_video_mask.any():
                if video_grid_thw is None:
                    raise ValueError(
                        "Per-turn prompt contains video tokens but video_grid_thw is missing. "
                        "This means the saved turn prompt and multi_modal_data['videos'] are inconsistent."
                    )
                prompt_mm_token_type_ids = mm_token_type_ids[0, :prompt_end]
                prompt_mm_token_type_ids[prompt_video_mask] = 2
        multi_modal_kwargs["mm_token_type_ids"] = mm_token_type_ids

    if not hasattr(processor, "get_rope_index"):
        raise ValueError("The multimodal processor must expose get_rope_index for per-turn training")

    vision_position_ids, _ = processor.get_rope_index(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **multi_modal_kwargs,
    )
    vision_position_ids = vision_position_ids.transpose(0, 1)

    valid_mask = attention_mask[0].bool()
    text_position_ids = torch.ones((1, input_ids.shape[-1]), dtype=torch.long)
    text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
    text_position_ids = text_position_ids.unsqueeze(0)
    return torch.cat((text_position_ids, vision_position_ids), dim=1).squeeze(0)


def _as_float(value: Any, *, default: float = 0.0) -> float:
    if value is None:
        return default
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return default
    return float(array.reshape(-1)[0])


def _truncate_response_before_generated_modality_token(
    response_ids: list[int],
    *,
    processor,
) -> tuple[list[int], bool]:
    """Keep only the valid text prefix before a generated image/video token.

    Real image/video tokens belong to the prompt and have matching processor
    features. If the language model emits one in its response, the model
    forward would count it as another visual feature placeholder even though
    no corresponding feature exists.
    """

    modality_token_ids = {
        token_id
        for token_id in (
            get_processor_token_id(processor, "image"),
            get_processor_token_id(processor, "video"),
        )
        if token_id is not None
    }
    if not modality_token_ids:
        return response_ids, False

    for index, token_id in enumerate(response_ids):
        if int(token_id) in modality_token_ids:
            return response_ids[:index], True
    return response_ids, False


def _trajectory_metric_values(batch: DataProto, name: str) -> list[float]:
    if name in batch.non_tensor_batch:
        return [_as_float(value) for value in batch.non_tensor_batch[name]]
    if name == "task_reward" and "token_level_scores" in batch.batch:
        return batch.batch["token_level_scores"].sum(dim=-1).detach().cpu().to(torch.float32).tolist()
    raise KeyError(f"Cannot find trajectory metric {name!r} in batch.non_tensor_batch")


def compute_soft_overlong_cost(
    *,
    response_length: int,
    max_response_length: int,
    buffer_length: int,
    cost_coef: float,
    truncated_by_length: bool,
) -> float:
    """Return a non-negative DAPO-style soft length cost for one turn."""

    if max_response_length <= 0:
        raise ValueError(f"max_response_length must be positive, got {max_response_length}")
    cost_coef = abs(float(cost_coef))
    if not np.isfinite(cost_coef):
        raise ValueError(f"cost_coef must be finite, got {cost_coef!r}")
    if cost_coef == 0:
        return 0.0

    buffer_length = max(int(buffer_length), 0)
    penalty_start = max_response_length - buffer_length
    if buffer_length == 0:
        return cost_coef if truncated_by_length or response_length >= max_response_length else 0.0
    if response_length <= penalty_start and not truncated_by_length:
        return 0.0
    if truncated_by_length or response_length >= max_response_length:
        return cost_coef
    ratio = (response_length - penalty_start) / buffer_length
    ratio = min(max(ratio, 0.0), 1.0)
    return cost_coef * ratio


def _strict_hard_error_cost(value: float, *, config_name: str) -> float:
    """Return a finite positive hard-error cost or fail before actor update."""

    cost = abs(float(value))
    if not np.isfinite(cost) or cost <= 0:
        raise ValueError(
            f"{config_name} must be finite and > 0 so a hard-error turn always has "
            f"strictly negative advantage, got {value!r}"
        )
    return cost


def compute_turn_advantage_shaping(
    turn_record: dict[str, Any],
    *,
    max_response_length: int,
    output_format_error_cost: float,
    tool_args_error_cost: float,
    truncation_error_cost: float,
    overlong_buffer_length: int,
    overlong_cost_coef: float,
    soft_overlong_enabled: bool = False,
) -> dict[str, Any]:
    """Classify one turn and return the mutually exclusive local advantage cost.

    Truncation takes precedence over format errors, and format errors take
    precedence over tool-argument errors. This avoids charging the same root
    failure multiple times (for example, an unparseable action has no
    meaningful tool arguments to validate).
    """

    response_ids = list(turn_record.get("response_ids") or [])
    response_length = int(turn_record.get("turn_response_length", len(response_ids)))
    truncated_by_length = bool(turn_record.get("turn_truncated_by_length", False))
    raw_output_format_failure = not bool(turn_record.get("output_format_success", False))
    raw_tool_args_failure = bool(turn_record.get("output_format_success", False)) and not bool(
        turn_record.get("tool_args_success", True)
    )
    soft_overlong_cost = 0.0
    if soft_overlong_enabled:
        soft_overlong_cost = compute_soft_overlong_cost(
            response_length=response_length,
            max_response_length=max_response_length,
            buffer_length=overlong_buffer_length,
            cost_coef=overlong_cost_coef,
            truncated_by_length=truncated_by_length,
        )

    format_cost = 0.0
    tool_args_cost = 0.0
    truncation_cost = 0.0
    applied_soft_overlong_cost = 0.0
    error_type = "none"
    if truncated_by_length:
        truncation_cost = _strict_hard_error_cost(
            truncation_error_cost,
            config_name="truncation_error_cost",
        )
        error_type = "truncation"
    elif raw_output_format_failure:
        format_cost = _strict_hard_error_cost(
            output_format_error_cost,
            config_name="output_format_error_cost",
        )
        error_type = "output_format"
    elif raw_tool_args_failure:
        tool_args_cost = _strict_hard_error_cost(
            tool_args_error_cost,
            config_name="tool_args_error_cost",
        )
        error_type = "tool_args"
    else:
        applied_soft_overlong_cost = soft_overlong_cost
        if applied_soft_overlong_cost > 0:
            error_type = "soft_overlong"

    output_format_error = error_type == "output_format"
    tool_args_error = error_type == "tool_args"
    local_cost = format_cost + tool_args_cost + truncation_cost + applied_soft_overlong_cost
    return {
        "hard_error": error_type in {"truncation", "output_format", "tool_args"},
        "error_type": error_type,
        # Public error flags are mutually exclusive and match the applied local
        # cost. Keep parser/validator outcomes under explicit raw names so a
        # truncated response can be diagnosed without also inflating the format
        # or tool-argument error rates.
        "output_format_error": output_format_error,
        "tool_args_error": tool_args_error,
        "truncated_by_length": truncated_by_length,
        "raw_output_format_failure": raw_output_format_failure,
        "raw_tool_args_failure": raw_tool_args_failure,
        "soft_overlong": applied_soft_overlong_cost > 0,
        "format_cost": format_cost,
        "tool_args_cost": tool_args_cost,
        "truncation_cost": truncation_cost,
        "soft_overlong_cost": applied_soft_overlong_cost,
        "local_cost": local_cost,
    }


def _compute_trajectory_grpo_advantages(
    batch: DataProto,
    *,
    trajectory_step_cost: float,
    norm_adv_by_std: bool,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Compute one normalized GRPO advantage per complete trajectory."""

    if "turn_records" not in batch.non_tensor_batch:
        raise KeyError("Cannot compute trajectory advantages because rollout output has no turn_records")

    task_rewards = _trajectory_metric_values(batch, "task_reward")
    if len(task_rewards) != len(batch):
        raise ValueError(f"Expected {len(batch)} task rewards, got {len(task_rewards)}")

    step_cost_coef = abs(float(trajectory_step_cost))
    trajectory_rewards = []
    trajectory_step_costs = []
    for trajectory_index, turn_records in enumerate(batch.non_tensor_batch["turn_records"]):
        num_turns = len(turn_records or [])
        step_cost = step_cost_coef * num_turns
        trajectory_step_costs.append(step_cost)
        trajectory_rewards.append(float(task_rewards[trajectory_index]) - step_cost)

    # The trainer computes this field before filtering. Reuse it only after
    # verifying that it has the exact new semantics.
    stored_rewards = batch.non_tensor_batch.get("trajectory_reward")
    if stored_rewards is not None:
        stored_rewards = np.asarray(stored_rewards, dtype=np.float64).reshape(-1)
        expected_rewards = np.asarray(trajectory_rewards, dtype=np.float64)
        if stored_rewards.shape != expected_rewards.shape or not np.allclose(
            stored_rewards,
            expected_rewards,
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ValueError(
                "batch.non_tensor_batch['trajectory_reward'] is inconsistent with "
                "task_reward - trajectory_step_cost * num_turns"
            )

    uids = batch.non_tensor_batch.get("uid", np.arange(len(batch), dtype=object))
    advantage_suppressed = batch.non_tensor_batch.get("trajectory_advantage_suppressed")
    if advantage_suppressed is None:
        advantage_suppressed = np.zeros(len(batch), dtype=bool)
    else:
        advantage_suppressed = np.asarray(advantage_suppressed, dtype=bool).reshape(-1)
        if advantage_suppressed.size != len(batch):
            raise ValueError(
                "Expected one trajectory_advantage_suppressed flag per trajectory, "
                f"got {advantage_suppressed.size} flags for {len(batch)} trajectories"
            )

    grouped_indices: dict[Any, list[int]] = defaultdict(list)
    for trajectory_index, uid in enumerate(uids):
        grouped_indices[uid].append(trajectory_index)

    trajectory_advantages = np.zeros(len(batch), dtype=np.float64)
    group_reward_stds = []
    group_reward_ranges = []
    group_sizes = []
    suppressed_group_count = 0
    suppressed_trajectory_count = 0
    for indices in grouped_indices.values():
        reward_array = np.asarray([trajectory_rewards[index] for index in indices], dtype=np.float64)
        reward_mean = float(reward_array.mean())
        reward_std = float(reward_array.std(ddof=1)) if reward_array.size > 1 else 0.0
        group_reward_stds.append(reward_std)
        group_reward_ranges.append(float(reward_array.max() - reward_array.min()))
        group_sizes.append(float(reward_array.size))

        centered = reward_array - reward_mean
        group_suppression_flags = {bool(advantage_suppressed[index]) for index in indices}
        if len(group_suppression_flags) != 1:
            raise ValueError(
                "All trajectories from the same prompt uid must have the same "
                "trajectory_advantage_suppressed flag"
            )
        suppress_group_advantage = group_suppression_flags.pop()
        if suppress_group_advantage:
            group_advantages = np.zeros_like(centered)
            suppressed_group_count += 1
            suppressed_trajectory_count += len(indices)
        elif norm_adv_by_std:
            group_advantages = centered / (reward_std + 1e-6) if reward_std > 0 else np.zeros_like(centered)
        else:
            group_advantages = centered
        for index, advantage in zip(indices, group_advantages, strict=True):
            trajectory_advantages[index] = float(advantage)

    trajectory_infos = [
        {
            "task_reward": float(task_rewards[index]),
            "trajectory_step_cost": float(trajectory_step_costs[index]),
            "trajectory_reward": float(trajectory_rewards[index]),
            "trajectory_advantage": float(trajectory_advantages[index]),
            "trajectory_advantage_suppressed": bool(advantage_suppressed[index]),
        }
        for index in range(len(batch))
    ]
    reward_array = np.asarray(trajectory_rewards, dtype=np.float64)
    advantage_array = np.asarray(trajectory_advantages, dtype=np.float64)
    metrics = {
        "visharness/per_turn/trajectory_group_count": float(len(grouped_indices)),
        "visharness/per_turn/trajectory_reward_mean": float(reward_array.mean()) if reward_array.size else 0.0,
        "visharness/per_turn/trajectory_advantage_mean": (
            float(advantage_array.mean()) if advantage_array.size else 0.0
        ),
        "visharness/per_turn/trajectory_advantage_std": (
            float(advantage_array.std()) if advantage_array.size else 0.0
        ),
        "visharness/per_turn/trajectory_advantage_suppressed_group_count": float(suppressed_group_count),
        "visharness/per_turn/trajectory_advantage_suppressed_trajectory_count": float(
            suppressed_trajectory_count
        ),
        "visharness/per_turn/trajectory_advantage_suppressed_trajectory_ratio": (
            suppressed_trajectory_count / max(len(batch), 1)
        ),
        "visharness/per_turn/mean_group_trajectory_reward_std": (
            float(np.mean(group_reward_stds)) if group_reward_stds else 0.0
        ),
        "visharness/per_turn/mean_group_trajectory_reward_range": (
            float(np.mean(group_reward_ranges)) if group_reward_ranges else 0.0
        ),
        "visharness/per_turn/trajectory_group_size_mean": (
            float(np.mean(group_sizes)) if group_sizes else 0.0
        ),
    }
    return trajectory_infos, metrics


def _build_turn_advantage_records(
    batch: DataProto,
    *,
    processor,
    max_prompt_length: int,
    max_response_length: int,
    trajectory_step_cost: float,
    output_format_error_cost: float,
    tool_args_error_cost: float,
    truncation_error_cost: float,
    overlong_buffer_length: int,
    overlong_cost_coef: float,
    soft_overlong_enabled: bool,
    norm_adv_by_std: bool,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    trajectory_infos, metrics = _compute_trajectory_grpo_advantages(
        batch,
        trajectory_step_cost=trajectory_step_cost,
        norm_adv_by_std=norm_adv_by_std,
    )
    records: list[dict[str, Any]] = []
    prompt_overlong_count = 0
    generated_modality_token_count = 0
    generated_modality_token_empty_prefix_count = 0
    total_turn_count = 0
    assigned_trajectory_advantages = []
    base_advantages = []
    final_turn_advantages = []
    local_costs = []
    format_costs = []
    tool_args_costs = []
    truncation_costs = []
    soft_overlong_costs = []
    hard_error_count = 0
    positive_advantage_clamped_count = 0
    output_format_error_count = 0
    tool_args_error_count = 0
    truncation_error_count = 0
    raw_output_format_failure_count = 0
    raw_tool_args_failure_count = 0
    soft_overlong_count = 0

    for trajectory_index, turn_records in enumerate(batch.non_tensor_batch["turn_records"]):
        if not turn_records:
            continue

        trajectory_info = trajectory_infos[trajectory_index]
        uid = (
            batch.non_tensor_batch["uid"][trajectory_index]
            if "uid" in batch.non_tensor_batch
            else f"prompt-{trajectory_index}"
        )
        trajectory_uid = (
            batch.non_tensor_batch["trajectory_uid"][trajectory_index]
            if "trajectory_uid" in batch.non_tensor_batch
            else f"trajectory-{trajectory_index}"
        )
        task_type = (
            str(batch.non_tensor_batch["task_type"][trajectory_index])
            if "task_type" in batch.non_tensor_batch
            else "unknown"
        )
        data_source = (
            str(batch.non_tensor_batch["data_source"][trajectory_index])
            if "data_source" in batch.non_tensor_batch
            else "unknown"
        )

        for turn_record in turn_records:
            response_ids = list(turn_record.get("response_ids") or [])
            if not response_ids:
                continue
            train_response_ids, generated_modality_token = _truncate_response_before_generated_modality_token(
                response_ids,
                processor=processor,
            )
            generated_modality_token_count += int(generated_modality_token)
            generated_modality_token_empty_prefix_count += int(
                generated_modality_token and not train_response_ids
            )

            total_turn_count += 1
            prompt_ids = list(turn_record["prompt_ids"])
            is_prompt_overlong = len(prompt_ids) > max_prompt_length
            prompt_overlong_count += int(is_prompt_overlong)

            shaping = compute_turn_advantage_shaping(
                turn_record,
                max_response_length=max_response_length,
                output_format_error_cost=output_format_error_cost,
                tool_args_error_cost=tool_args_error_cost,
                truncation_error_cost=truncation_error_cost,
                overlong_buffer_length=overlong_buffer_length,
                overlong_cost_coef=overlong_cost_coef,
                soft_overlong_enabled=soft_overlong_enabled,
            )
            trajectory_advantage = float(trajectory_info["trajectory_advantage"])
            base_advantage = (
                min(trajectory_advantage, 0.0)
                if shaping["hard_error"]
                else trajectory_advantage
            )
            turn_advantage = base_advantage - float(shaping["local_cost"])

            assigned_trajectory_advantages.append(trajectory_advantage)
            base_advantages.append(base_advantage)
            final_turn_advantages.append(turn_advantage)
            local_costs.append(float(shaping["local_cost"]))
            format_costs.append(float(shaping["format_cost"]))
            tool_args_costs.append(float(shaping["tool_args_cost"]))
            truncation_costs.append(float(shaping["truncation_cost"]))
            soft_overlong_costs.append(float(shaping["soft_overlong_cost"]))
            hard_error_count += int(shaping["hard_error"])
            positive_advantage_clamped_count += int(shaping["hard_error"] and trajectory_advantage > 0)
            output_format_error_count += int(shaping["output_format_error"])
            tool_args_error_count += int(shaping["tool_args_error"])
            truncation_error_count += int(shaping["truncated_by_length"])
            raw_output_format_failure_count += int(shaping["raw_output_format_failure"])
            raw_tool_args_failure_count += int(shaping["raw_tool_args_failure"])
            soft_overlong_count += int(shaping["soft_overlong"])

            records.append(
                {
                    "trajectory_index": trajectory_index,
                    "uid": uid,
                    "trajectory_uid": trajectory_uid,
                    "task_type": task_type,
                    "data_source": data_source,
                    "turn_record": turn_record,
                    "train_response_ids": train_response_ids,
                    "generated_modality_token": generated_modality_token,
                    "trajectory_num_turns": len(turn_records),
                    **trajectory_info,
                    "assigned_trajectory_advantage": trajectory_advantage,
                    "base_advantage": base_advantage,
                    "turn_advantage": turn_advantage,
                    **shaping,
                    "is_prompt_overlong": is_prompt_overlong,
                }
            )

    exclusive_hard_error_count = (
        output_format_error_count + tool_args_error_count + truncation_error_count
    )
    if exclusive_hard_error_count != hard_error_count:
        raise RuntimeError(
            "Mutually exclusive hard-error counts must sum to hard_error_count, got "
            f"format={output_format_error_count}, args={tool_args_error_count}, "
            f"truncation={truncation_error_count}, hard={hard_error_count}"
        )

    metrics.update(
        {
            "visharness/per_turn/total_records": float(total_turn_count),
            "visharness/per_turn/prompt_overlong_dropped": float(prompt_overlong_count),
            "visharness/per_turn/generated_modality_token_truncated": float(generated_modality_token_count),
            "visharness/per_turn/generated_modality_token_empty_prefix_dropped": float(
                generated_modality_token_empty_prefix_count
            ),
            "visharness/per_turn/assigned_trajectory_advantage_mean": (
                float(np.mean(assigned_trajectory_advantages)) if assigned_trajectory_advantages else 0.0
            ),
            "visharness/per_turn/base_advantage_after_clamp_mean": (
                float(np.mean(base_advantages)) if base_advantages else 0.0
            ),
            "visharness/per_turn/final_turn_advantage_mean": (
                float(np.mean(final_turn_advantages)) if final_turn_advantages else 0.0
            ),
            "visharness/per_turn/local_advantage_cost_mean": (
                float(np.mean(local_costs)) if local_costs else 0.0
            ),
            "visharness/per_turn/format_error_cost_mean": (
                float(np.mean(format_costs)) if format_costs else 0.0
            ),
            "visharness/per_turn/tool_args_error_cost_mean": (
                float(np.mean(tool_args_costs)) if tool_args_costs else 0.0
            ),
            "visharness/per_turn/truncation_error_cost_mean": (
                float(np.mean(truncation_costs)) if truncation_costs else 0.0
            ),
            "visharness/per_turn/soft_overlong_cost_mean": (
                float(np.mean(soft_overlong_costs)) if soft_overlong_costs else 0.0
            ),
            "visharness/per_turn/hard_error_turn_rate": hard_error_count / max(total_turn_count, 1),
            "visharness/per_turn/positive_trajectory_advantage_clamped_rate": (
                positive_advantage_clamped_count / max(total_turn_count, 1)
            ),
            "visharness/per_turn/output_format_error_rate": output_format_error_count / max(total_turn_count, 1),
            "visharness/per_turn/tool_args_error_rate": tool_args_error_count / max(total_turn_count, 1),
            "visharness/per_turn/truncation_error_rate": truncation_error_count / max(total_turn_count, 1),
            "visharness/per_turn/raw_output_format_failure_rate": (
                raw_output_format_failure_count / max(total_turn_count, 1)
            ),
            "visharness/per_turn/raw_tool_args_failure_rate": (
                raw_tool_args_failure_count / max(total_turn_count, 1)
            ),
            "visharness/per_turn/soft_overlong_rate": soft_overlong_count / max(total_turn_count, 1),
            "visharness/per_turn/soft_overlong_enabled": float(bool(soft_overlong_enabled)),
        }
    )
    return records, metrics


def _summarize_trainable_turn_advantages(records: list[dict[str, Any]]) -> dict[str, float]:
    """Summarize the exact turn records that will be tensorized for actor update."""

    count = len(records)
    assigned = [float(record["assigned_trajectory_advantage"]) for record in records]
    base = [float(record["base_advantage"]) for record in records]
    legacy_additive_final = [float(record["turn_advantage"]) for record in records]
    effective_final = [
        float(record.get("effective_policy_advantage", record["turn_advantage"]))
        for record in records
    ]
    local = [float(record["local_cost"]) for record in records]
    hard_error_count = sum(bool(record["hard_error"]) for record in records)
    output_format_error_count = sum(bool(record["output_format_error"]) for record in records)
    tool_args_error_count = sum(bool(record["tool_args_error"]) for record in records)
    truncation_error_count = sum(bool(record["truncated_by_length"]) for record in records)
    if output_format_error_count + tool_args_error_count + truncation_error_count != hard_error_count:
        raise RuntimeError(
            "Trainable mutually exclusive hard-error counts must sum to hard_error_count"
        )

    def mean(key: str) -> float:
        return float(np.mean([float(record[key]) for record in records])) if records else 0.0

    metrics = {
        "visharness/per_turn/assigned_trajectory_advantage_mean": float(np.mean(assigned)) if assigned else 0.0,
        "visharness/per_turn/base_advantage_after_clamp_mean": float(np.mean(base)) if base else 0.0,
        # "final" now means the actual post-q policy coefficient. Keep the
        # historical base-local value under an explicit counterfactual name.
        "visharness/per_turn/final_turn_advantage_mean": (
            float(np.mean(effective_final)) if effective_final else 0.0
        ),
        "visharness/per_turn/legacy_additive_turn_advantage_mean": (
            float(np.mean(legacy_additive_final)) if legacy_additive_final else 0.0
        ),
        "visharness/per_turn/local_advantage_cost_mean": float(np.mean(local)) if local else 0.0,
        "visharness/per_turn/format_error_cost_mean": mean("format_cost"),
        "visharness/per_turn/tool_args_error_cost_mean": mean("tool_args_cost"),
        "visharness/per_turn/truncation_error_cost_mean": mean("truncation_cost"),
        "visharness/per_turn/soft_overlong_cost_mean": mean("soft_overlong_cost"),
        "visharness/per_turn/hard_error_turn_rate": (
            hard_error_count / max(count, 1)
        ),
        "visharness/per_turn/positive_trajectory_advantage_clamped_rate": (
            sum(
                bool(record["hard_error"]) and float(record["assigned_trajectory_advantage"]) > 0
                for record in records
            )
            / max(count, 1)
        ),
        "visharness/per_turn/output_format_error_rate": (
            output_format_error_count / max(count, 1)
        ),
        "visharness/per_turn/tool_args_error_rate": (
            tool_args_error_count / max(count, 1)
        ),
        "visharness/per_turn/truncation_error_rate": (
            truncation_error_count / max(count, 1)
        ),
        "visharness/per_turn/raw_output_format_failure_rate": (
            sum(bool(record["raw_output_format_failure"]) for record in records)
            / max(count, 1)
        ),
        "visharness/per_turn/raw_tool_args_failure_rate": (
            sum(bool(record["raw_tool_args_failure"]) for record in records)
            / max(count, 1)
        ),
        "visharness/per_turn/soft_overlong_rate": (
            sum(bool(record["soft_overlong"]) for record in records) / max(count, 1)
        ),
    }
    return metrics


def _assign_trajectory_loss_weights(
    records: list[dict[str, Any]],
    *,
    enabled: bool,
    local_cost_mode: str,
) -> dict[str, float]:
    """Assign trajectory weights and compose the PPO-only turn advantage.

    With ``M`` trainable turns from ``N`` trainable trajectories and ``K_i``
    trainable turns in trajectory ``i``, the enabled weight is

    ``w_it = M / (N * K_i)``.

    The existing ``seq-mean-token-mean`` actor loss then becomes exactly
    ``(1/N) * sum_i (1/K_i) * sum_t loss_it`` while preserving the previous
    overall loss scale because the mean row weight is one.  Counts are taken
    after prompt/response filtering, so a dropped turn receives no share of a
    trajectory's training budget.

    ``trajectory_weighted_additive`` preserves the previous objective
    ``w_i * (base_advantage - local_cost)``. ``per_event_floor`` keeps the
    task component trajectory-equal while applying a trajectory-weight-free
    local constraint: hard-error turns become
    ``min(w_i * base_advantage, -cost)`` and soft-overlong turns become
    ``w_i * base_advantage - cost``.
    """

    if not records:
        raise ValueError("Cannot assign trajectory weights to an empty turn list")
    local_cost_mode = str(local_cost_mode).strip().lower()
    if local_cost_mode not in SUPPORTED_LOCAL_COST_MODES:
        raise ValueError(
            "local_cost_mode must be one of "
            f"{sorted(SUPPORTED_LOCAL_COST_MODES)}, got {local_cost_mode!r}"
        )

    counts: dict[int, int] = defaultdict(int)
    for record in records:
        counts[int(record["trajectory_index"])] += 1

    num_turns = len(records)
    num_trajectories = len(counts)
    mean_turns = num_turns / num_trajectories
    weights = []
    assigned_advantages = []
    final_advantages = []
    ppo_advantages = []
    effective_advantages = []
    effective_local_costs = []
    local_floor_applied = []
    trajectory_advantages: dict[int, float] = {}

    for record in records:
        trajectory_index = int(record["trajectory_index"])
        trainable_turn_count = counts[trajectory_index]
        loss_weight = mean_turns / trainable_turn_count if enabled else 1.0
        if not np.isfinite(loss_weight) or loss_weight <= 0:
            raise ValueError(
                "Invalid trajectory loss weight for "
                f"trajectory {trajectory_index}: {loss_weight}"
            )
        record["trajectory_trainable_turn_count"] = trainable_turn_count
        record["trajectory_loss_weight"] = float(loss_weight)
        assigned_advantage = float(record["assigned_trajectory_advantage"])
        base_advantage = float(record["base_advantage"])
        local_cost = float(record["local_cost"])
        if local_cost_mode == "trajectory_weighted_additive":
            ppo_turn_advantage = float(record["turn_advantage"])
        elif bool(record["hard_error"]):
            ppo_turn_advantage = min(
                base_advantage,
                -local_cost / float(loss_weight),
            )
        else:
            ppo_turn_advantage = base_advantage - local_cost / float(loss_weight)

        effective_policy_advantage = float(loss_weight) * ppo_turn_advantage
        if bool(record["hard_error"]) and not (
            np.isfinite(effective_policy_advantage) and effective_policy_advantage < 0
        ):
            raise ValueError(
                "Hard-error turn must have a finite strictly negative effective policy advantage, "
                f"got {effective_policy_advantage!r} for trajectory {trajectory_index}"
            )
        weighted_base_advantage = float(loss_weight) * base_advantage
        effective_local_cost = max(weighted_base_advantage - effective_policy_advantage, 0.0)
        floor_applied = bool(
            local_cost_mode == "per_event_floor"
            and record["hard_error"]
            and effective_policy_advantage < weighted_base_advantage - 1e-12
        )

        record["local_cost_mode"] = local_cost_mode
        record["ppo_turn_advantage"] = float(ppo_turn_advantage)
        record["effective_policy_advantage"] = float(effective_policy_advantage)
        record["effective_local_cost"] = float(effective_local_cost)
        record["local_floor_applied"] = floor_applied
        record["weighted_turn_advantage"] = float(effective_policy_advantage)
        weights.append(float(loss_weight))
        assigned_advantages.append(assigned_advantage)
        final_advantages.append(float(record["turn_advantage"]))
        ppo_advantages.append(float(ppo_turn_advantage))
        effective_advantages.append(float(effective_policy_advantage))
        effective_local_costs.append(float(effective_local_cost))
        local_floor_applied.append(floor_applied)
        trajectory_advantages.setdefault(
            trajectory_index,
            assigned_advantage,
        )

    weight_array = np.asarray(weights, dtype=np.float64)
    assigned_array = np.asarray(assigned_advantages, dtype=np.float64)
    final_array = np.asarray(final_advantages, dtype=np.float64)
    ppo_array = np.asarray(ppo_advantages, dtype=np.float64)
    effective_array = np.asarray(effective_advantages, dtype=np.float64)
    effective_local_cost_array = np.asarray(effective_local_costs, dtype=np.float64)
    floor_applied_array = np.asarray(local_floor_applied, dtype=bool)
    hard_error_array = np.asarray([bool(record["hard_error"]) for record in records], dtype=bool)
    advantage_eps = 1e-12
    positive_effective_mask = effective_array > advantage_eps
    negative_effective_mask = effective_array < -advantage_eps
    zero_effective_mask = ~(positive_effective_mask | negative_effective_mask)
    positive_effective_count = int(np.count_nonzero(positive_effective_mask))
    negative_effective_count = int(np.count_nonzero(negative_effective_mask))
    zero_effective_count = int(np.count_nonzero(zero_effective_mask))
    positive_effective_mass = float(effective_array[positive_effective_mask].sum())
    negative_effective_mass = float(-effective_array[negative_effective_mask].sum())
    absolute_effective_mass = positive_effective_mass + negative_effective_mass
    turn_count_array = np.asarray(list(counts.values()), dtype=np.float64)
    trajectory_advantage_array = np.asarray(
        [trajectory_advantages[index] for index in counts],
        dtype=np.float64,
    )
    trajectory_mass = defaultdict(float)
    for record in records:
        trajectory_mass[int(record["trajectory_index"])] += float(record["trajectory_loss_weight"])
    trajectory_mass_array = np.asarray(list(trajectory_mass.values()), dtype=np.float64)

    covariance = float(
        np.mean(
            (turn_count_array - turn_count_array.mean())
            * (trajectory_advantage_array - trajectory_advantage_array.mean())
        )
    )
    weight_sum = float(weight_array.sum())
    return {
        "visharness/per_turn/trajectory_equal_weight_enabled": float(bool(enabled)),
        "visharness/per_turn/trainable_trajectory_count": float(num_trajectories),
        "visharness/per_turn/trainable_turns_per_trajectory_mean": float(turn_count_array.mean()),
        "visharness/per_turn/trainable_turns_per_trajectory_min": float(turn_count_array.min()),
        "visharness/per_turn/trainable_turns_per_trajectory_max": float(turn_count_array.max()),
        "visharness/per_turn/trajectory_loss_weight_mean": float(weight_array.mean()),
        "visharness/per_turn/trajectory_loss_weight_min": float(weight_array.min()),
        "visharness/per_turn/trajectory_loss_weight_max": float(weight_array.max()),
        "visharness/per_turn/trajectory_loss_weight_std": float(weight_array.std()),
        "visharness/per_turn/trajectory_loss_mass_mean": float(trajectory_mass_array.mean()),
        "visharness/per_turn/trajectory_loss_mass_std": float(trajectory_mass_array.std()),
        "visharness/per_turn/local_cost_per_event_floor_enabled": float(
            local_cost_mode == "per_event_floor"
        ),
        "visharness/per_turn/ppo_turn_advantage_mean": float(ppo_array.mean()),
        "visharness/per_turn/ppo_turn_advantage_abs_mean": float(np.abs(ppo_array).mean()),
        "visharness/per_turn/effective_policy_advantage_mean": float(effective_array.mean()),
        "visharness/per_turn/effective_policy_advantage_abs_mean": float(
            np.abs(effective_array).mean()
        ),
        "visharness/per_turn/effective_policy_advantage_std": float(effective_array.std()),
        "visharness/per_turn/effective_policy_advantage_min": float(effective_array.min()),
        "visharness/per_turn/effective_policy_advantage_max": float(effective_array.max()),
        # Counts/rates describe how many trainable assistant turns receive each
        # sign. Masses describe the total pre-PPO coefficient magnitude after
        # trajectory equal weighting and local-error shaping; they are not
        # token counts or final gradient norms.
        "visharness/per_turn/effective_positive_advantage_count": float(
            positive_effective_count
        ),
        "visharness/per_turn/effective_negative_advantage_count": float(
            negative_effective_count
        ),
        "visharness/per_turn/effective_zero_advantage_count": float(
            zero_effective_count
        ),
        "visharness/per_turn/effective_positive_advantage_rate": float(
            positive_effective_count / num_turns
        ),
        "visharness/per_turn/effective_negative_advantage_rate": float(
            negative_effective_count / num_turns
        ),
        "visharness/per_turn/effective_zero_advantage_rate": float(
            zero_effective_count / num_turns
        ),
        "visharness/per_turn/effective_positive_advantage_mass": positive_effective_mass,
        "visharness/per_turn/effective_negative_advantage_mass": negative_effective_mass,
        "visharness/per_turn/effective_net_advantage_mass": (
            positive_effective_mass - negative_effective_mass
        ),
        "visharness/per_turn/effective_positive_advantage_mass_share": (
            positive_effective_mass / absolute_effective_mass
            if absolute_effective_mass > advantage_eps
            else 0.0
        ),
        "visharness/per_turn/effective_negative_advantage_mass_share": (
            negative_effective_mass / absolute_effective_mass
            if absolute_effective_mass > advantage_eps
            else 0.0
        ),
        "visharness/per_turn/effective_local_cost_mean": float(effective_local_cost_array.mean()),
        "visharness/per_turn/local_floor_applied_rate": float(floor_applied_array.mean()),
        "visharness/per_turn/hard_error_local_floor_applied_rate": float(
            floor_applied_array[hard_error_array].mean() if np.any(hard_error_array) else 0.0
        ),
        "visharness/per_turn/trajectory_weighted_assigned_advantage_mean": float(
            np.dot(weight_array, assigned_array) / weight_sum
        ),
        "visharness/per_turn/trajectory_weighted_final_advantage_mean": float(
            effective_array.sum() / weight_sum
        ),
        "visharness/per_turn/trajectory_weighted_legacy_final_advantage_mean": float(
            np.dot(weight_array, final_array) / weight_sum
        ),
        "visharness/per_turn/trainable_turn_count_advantage_covariance": covariance,
    }


def _tensorize_turn_records(
    records: list[dict[str, Any]],
    *,
    tokenizer,
    processor,
    max_prompt_length: int,
    max_response_length: int,
    meta_info: dict[str, Any],
    num_trajectories: int,
) -> DataProto:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer pad_token_id must be set before per-turn training")

    tensor_rows: dict[str, list[torch.Tensor]] = {
        "prompts": [],
        "responses": [],
        "response_mask": [],
        "input_ids": [],
        "attention_mask": [],
        "position_ids": [],
        "advantages": [],
        "returns": [],
        "token_level_scores": [],
        "token_level_rewards": [],
        TRAJECTORY_LOSS_WEIGHT_KEY: [],
    }
    rollout_log_prob_rows: list[torch.Tensor] = []
    all_turns_have_rollout_log_probs = True
    non_tensor_rows: dict[str, list[Any]] = {
        "uid": [],
        "trajectory_uid": [],
        "task_type": [],
        "data_source": [],
        "turn_index": [],
        "turn_tool_name": [],
        "trajectory_num_turns": [],
        "trajectory_trainable_turn_count": [],
        "trajectory_loss_weight": [],
        "visible_image_names": [],
        "multi_modal_inputs": [],
        "trajectory_reward": [],
        "trajectory_advantage": [],
        "trajectory_advantage_suppressed": [],
        "assigned_trajectory_advantage": [],
        "base_advantage_after_clamp": [],
        "turn_advantage": [],
        "ppo_turn_advantage": [],
        "effective_policy_advantage": [],
        "weighted_turn_advantage": [],
        "effective_local_cost": [],
        "local_floor_applied": [],
        "local_cost_mode": [],
        "trajectory_task_reward": [],
        "trajectory_step_cost": [],
        "turn_local_advantage_cost": [],
        "turn_format_error_cost": [],
        "turn_tool_args_error_cost": [],
        "turn_truncation_error_cost": [],
        "turn_soft_overlong_cost": [],
        "turn_error_type": [],
        "turn_hard_error": [],
        "turn_raw_output_format_failure": [],
        "turn_raw_tool_args_failure": [],
    }

    for record in records:
        if record["is_prompt_overlong"]:
            continue
        turn_record = record["turn_record"]
        prompt_ids = list(turn_record["prompt_ids"])
        response_ids = list(record.get("train_response_ids", turn_record["response_ids"]))[:max_response_length]
        if not response_ids:
            continue

        prompts, prompt_attention = _pad_tokens(prompt_ids, max_prompt_length, pad_token_id, left=True)
        responses, response_attention = _pad_tokens(response_ids, max_response_length, pad_token_id, left=False)
        response_mask = response_attention.clone()
        input_ids = torch.cat((prompts, responses), dim=0)
        attention_mask = torch.cat((prompt_attention, response_attention), dim=0)

        multi_modal_data = copy.deepcopy(turn_record.get("multi_modal_data") or {})
        multi_modal_inputs = _build_multi_modal_inputs(
            tokenizer=tokenizer,
            processor=processor,
            input_ids=input_ids,
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=turn_record.get("mm_processor_kwargs"),
        )
        position_ids = _compute_position_ids(
            processor=processor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            multi_modal_inputs=multi_modal_inputs,
            prompt_width=max_prompt_length,
        )

        # The actor loss later multiplies this scalar by trajectory_loss_weight.
        # In per_event_floor mode, _assign_trajectory_loss_weights has already
        # divided the fixed local term by that weight. Consequently the actor
        # sees q * base - cost for soft costs and min(q * base, -cost)
        # for hard errors, rather than q * (base - cost).
        advantage_scalar = torch.tensor(
            float(record.get("ppo_turn_advantage", record["turn_advantage"])),
            dtype=torch.float32,
        )
        trajectory_loss_weight = float(record.get("trajectory_loss_weight", 1.0))
        return_scalar = advantage_scalar.clone()
        token_scores = torch.zeros(max_response_length, dtype=torch.float32)
        active_positions = response_mask.bool().nonzero(as_tuple=False).flatten()
        if active_positions.numel() > 0:
            # Preserve the historical/counterfactual additive score for rollout
            # diagnostics. The actual PPO input is advantage_scalar, and its
            # post-q coefficient is recorded as effective_policy_advantage.
            token_scores[active_positions[-1]] = float(record["turn_advantage"])

        tensor_rows["prompts"].append(prompts)
        tensor_rows["responses"].append(responses)
        tensor_rows["response_mask"].append(response_mask)
        tensor_rows["input_ids"].append(input_ids)
        tensor_rows["attention_mask"].append(attention_mask)
        tensor_rows["position_ids"].append(position_ids)
        tensor_rows["advantages"].append(response_mask * advantage_scalar)
        tensor_rows["returns"].append(response_mask * return_scalar)
        tensor_rows["token_level_scores"].append(token_scores)
        tensor_rows["token_level_rewards"].append(token_scores.clone())
        tensor_rows[TRAJECTORY_LOSS_WEIGHT_KEY].append(
            response_mask.to(dtype=torch.float32) * trajectory_loss_weight
        )

        response_logprobs = turn_record.get("response_logprobs")
        if response_logprobs is None:
            all_turns_have_rollout_log_probs = False
        else:
            response_logprobs = list(response_logprobs)[: len(response_ids)]
            padded_logprobs = torch.zeros(max_response_length, dtype=torch.float32)
            padded_logprobs[: len(response_logprobs)] = torch.tensor(response_logprobs, dtype=torch.float32)
            rollout_log_prob_rows.append(padded_logprobs)

        non_tensor_rows["uid"].append(record["uid"])
        non_tensor_rows["trajectory_uid"].append(record["trajectory_uid"])
        non_tensor_rows["task_type"].append(record["task_type"])
        non_tensor_rows["data_source"].append(record["data_source"])
        non_tensor_rows["turn_index"].append(turn_record["turn_index"])
        non_tensor_rows["turn_tool_name"].append(turn_record.get("tool_name"))
        non_tensor_rows["trajectory_num_turns"].append(record["trajectory_num_turns"])
        non_tensor_rows["trajectory_trainable_turn_count"].append(
            int(record.get("trajectory_trainable_turn_count", record["trajectory_num_turns"]))
        )
        non_tensor_rows["trajectory_loss_weight"].append(trajectory_loss_weight)
        non_tensor_rows["visible_image_names"].append(copy.deepcopy(turn_record.get("visible_image_names", [])))
        non_tensor_rows["multi_modal_inputs"].append(multi_modal_inputs)
        non_tensor_rows["trajectory_reward"].append(record["trajectory_reward"])
        non_tensor_rows["trajectory_advantage"].append(record["trajectory_advantage"])
        non_tensor_rows["trajectory_advantage_suppressed"].append(
            record["trajectory_advantage_suppressed"]
        )
        non_tensor_rows["assigned_trajectory_advantage"].append(record["assigned_trajectory_advantage"])
        non_tensor_rows["base_advantage_after_clamp"].append(record["base_advantage"])
        non_tensor_rows["turn_advantage"].append(record["turn_advantage"])
        non_tensor_rows["ppo_turn_advantage"].append(
            float(record.get("ppo_turn_advantage", record["turn_advantage"]))
        )
        non_tensor_rows["effective_policy_advantage"].append(
            float(
                record.get(
                    "effective_policy_advantage",
                    trajectory_loss_weight * record["turn_advantage"],
                )
            )
        )
        non_tensor_rows["weighted_turn_advantage"].append(
            float(record.get("weighted_turn_advantage", trajectory_loss_weight * record["turn_advantage"]))
        )
        non_tensor_rows["effective_local_cost"].append(
            float(record.get("effective_local_cost", trajectory_loss_weight * record["local_cost"]))
        )
        non_tensor_rows["local_floor_applied"].append(
            bool(record.get("local_floor_applied", False))
        )
        non_tensor_rows["local_cost_mode"].append(
            str(record.get("local_cost_mode", "trajectory_weighted_additive"))
        )
        non_tensor_rows["trajectory_task_reward"].append(record["task_reward"])
        non_tensor_rows["trajectory_step_cost"].append(record["trajectory_step_cost"])
        non_tensor_rows["turn_local_advantage_cost"].append(record["local_cost"])
        non_tensor_rows["turn_format_error_cost"].append(record["format_cost"])
        non_tensor_rows["turn_tool_args_error_cost"].append(record["tool_args_cost"])
        non_tensor_rows["turn_truncation_error_cost"].append(record["truncation_cost"])
        non_tensor_rows["turn_soft_overlong_cost"].append(record["soft_overlong_cost"])
        non_tensor_rows["turn_error_type"].append(record["error_type"])
        non_tensor_rows["turn_hard_error"].append(record["hard_error"])
        non_tensor_rows["turn_raw_output_format_failure"].append(
            record["raw_output_format_failure"]
        )
        non_tensor_rows["turn_raw_tool_args_failure"].append(
            record["raw_tool_args_failure"]
        )

    if not tensor_rows["prompts"]:
        raise NoTrainablePerTurnSamplesError(
            "No trainable per-turn samples remain after applying the training prompt-length limit"
        )

    tensors = {key: torch.stack(rows, dim=0) for key, rows in tensor_rows.items()}
    if all_turns_have_rollout_log_probs and len(rollout_log_prob_rows) == len(tensor_rows["prompts"]):
        tensors["rollout_log_probs"] = torch.stack(rollout_log_prob_rows, dim=0)

    meta_info = copy.deepcopy(meta_info)
    meta_info["global_token_num"] = tensors["attention_mask"].sum(dim=-1).tolist()
    meta_info["visharness_num_trajectories"] = num_trajectories
    meta_info["visharness_num_turns"] = len(tensor_rows["prompts"])

    return DataProto.from_dict(
        tensors=tensors,
        non_tensors={key: _object_array(values) for key, values in non_tensor_rows.items()},
        meta_info=meta_info,
    )


def build_per_turn_grpo_batch(
    batch: DataProto,
    *,
    tokenizer,
    processor=None,
    max_prompt_length: int,
    max_response_length: int,
    trajectory_step_cost: float,
    output_format_error_cost: float,
    tool_args_error_cost: float,
    truncation_error_cost: float,
    overlong_buffer_length: int,
    overlong_cost_coef: float,
    norm_adv_by_std: bool = True,
    trajectory_equal_weight: bool = False,
    local_cost_mode: str = "trajectory_weighted_additive",
    soft_overlong_enabled: bool = False,
) -> tuple[DataProto, dict[str, float]]:
    """Build per-turn samples from trajectory-level GRPO advantages.

    Trajectory rewards contain task reward and, when configured, total
    interaction step cost. The normalized trajectory advantage is assigned to
    every turn. Local costs are applied afterwards. Hard-error turns
    clamp a positive trajectory advantage to zero. ``local_cost_mode`` then
    controls whether the mutually exclusive local cost is trajectory-weighted
    (legacy) or enforced as a fixed per-event floor.
    """

    if max_prompt_length <= 0 or max_response_length <= 0:
        raise ValueError(
            f"max_prompt_length and max_response_length must be positive, got "
            f"{max_prompt_length=} {max_response_length=}"
        )

    records, metrics = _build_turn_advantage_records(
        batch,
        processor=processor,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        trajectory_step_cost=trajectory_step_cost,
        output_format_error_cost=output_format_error_cost,
        tool_args_error_cost=tool_args_error_cost,
        truncation_error_cost=truncation_error_cost,
        overlong_buffer_length=overlong_buffer_length,
        overlong_cost_coef=overlong_cost_coef,
        soft_overlong_enabled=soft_overlong_enabled,
        norm_adv_by_std=norm_adv_by_std,
    )
    if not records:
        raise NoTrainablePerTurnSamplesError("No assistant turns were produced by the trajectory batch")
    trainable_records = [
        record
        for record in records
        if not record["is_prompt_overlong"] and bool(record.get("train_response_ids"))
    ]
    if not trainable_records:
        raise NoTrainablePerTurnSamplesError(
            "No trainable per-turn samples remain after applying the training prompt-length limit"
        )
    metrics.update(
        _assign_trajectory_loss_weights(
            trainable_records,
            enabled=bool(trajectory_equal_weight),
            local_cost_mode=local_cost_mode,
        )
    )
    metrics.update(_summarize_trainable_turn_advantages(trainable_records))
    per_turn_batch = _tensorize_turn_records(
        trainable_records,
        tokenizer=tokenizer,
        processor=processor,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        meta_info=batch.meta_info,
        num_trajectories=len(batch),
    )
    metrics["visharness/per_turn/trainable_records"] = float(len(per_turn_batch))
    metrics["visharness/per_turn/prompt_overlong_drop_ratio"] = (
        metrics["visharness/per_turn/prompt_overlong_dropped"]
        / max(metrics["visharness/per_turn/total_records"], 1.0)
    )
    return per_turn_batch, metrics


def pad_per_turn_batch_to_divisor(batch: DataProto, divisor: int) -> tuple[DataProto, int]:
    """Pad a per-turn batch for data-parallel dispatch without adding loss."""

    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}")
    remainder = len(batch) % divisor
    if remainder == 0:
        return batch, 0

    pad_size = divisor - remainder
    padding_parts = []
    remaining = pad_size
    while remaining > 0:
        take = min(remaining, len(batch))
        padding_parts.append(batch[:take])
        remaining -= take

    padded = DataProto.concat([batch, *padding_parts])
    padding_slice = slice(len(batch), len(padded))
    for key in (
        "response_mask",
        "advantages",
        "returns",
        "rollout_log_probs",
        "token_level_scores",
        "token_level_rewards",
        TRAJECTORY_LOSS_WEIGHT_KEY,
    ):
        if key in padded.batch:
            padded.batch[key][padding_slice] = 0
    padded.meta_info["global_token_num"] = padded.batch["attention_mask"].sum(dim=-1).tolist()
    return padded, pad_size


def plan_fixed_per_turn_mini_batches(
    batch_size: int,
    num_mini_batches: int,
    dp_size: int,
) -> tuple[int, int, int]:
    """Plan equal global mini-batches without shrinking them to a small divisor.

    Ordinary VisHarness GRPO fixes the number of optimizer mini-batches per
    rollout update. The per-turn batch is padded to ``dp_size *
    num_mini_batches`` so every DP rank receives the same number of rows in
    every optimizer mini-batch. Padding rows are later fully loss-masked by
    :func:`pad_per_turn_batch_to_divisor`.

    Returns:
        ``(padded_batch_size, padding_size, global_mini_batch_size)``.
    """

    if batch_size <= 0 or num_mini_batches <= 0 or dp_size <= 0:
        raise ValueError(
            "batch_size, num_mini_batches, and dp_size must be positive, got "
            f"{batch_size=}, {num_mini_batches=}, {dp_size=}"
        )

    divisor = num_mini_batches * dp_size
    padding_size = (-batch_size) % divisor
    padded_batch_size = batch_size + padding_size
    global_mini_batch_size = padded_batch_size // num_mini_batches

    if global_mini_batch_size % dp_size != 0:
        raise RuntimeError(
            "Internal per-turn mini-batch planning error: global mini-batch "
            f"size {global_mini_batch_size} is not divisible by DP size {dp_size}"
        )
    return padded_batch_size, padding_size, global_mini_batch_size
