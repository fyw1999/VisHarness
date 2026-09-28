import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.utils import tensordict_utils as tu
from visharness.rl_trainer.per_turn import (
    _assign_trajectory_loss_weights,
    _summarize_trainable_turn_advantages,
    build_per_turn_grpo_batch,
    compute_turn_advantage_shaping,
    pad_per_turn_batch_to_divisor,
    plan_fixed_per_turn_mini_batches,
)
from visharness.rl_trainer.per_turn_loss import (
    EFFECTIVE_GLOBAL_BATCH_SIZE_KEY,
    TRAJECTORY_LOSS_WEIGHT_KEY,
)
from visharness.rl_trainer.visharness_trainer import VisHarnessTrainer


class FakeTokenizer:
    pad_token_id = 0

    def decode(self, token_ids, **kwargs):
        return " ".join(str(int(token_id)) for token_id in token_ids)


class FakeBatch(dict):
    def convert_to_tensors(self, tensor_type):
        assert tensor_type == "pt"
        return self


class FakeProcessor:
    def __call__(self, *, text, images, videos, return_tensors, **kwargs):
        assert videos is None
        assert return_tensors == "pt"
        image_value = images[0] if images else -1
        return FakeBatch({"input_ids": torch.tensor([[1]]), "pixel_values": torch.tensor([[image_value]])})

    def get_rope_index(self, *, input_ids, attention_mask, image_grid_thw, video_grid_thw):
        del attention_mask, image_grid_thw, video_grid_thw
        return torch.zeros((3, 1, input_ids.shape[-1]), dtype=torch.long), None


class FakeQwen3Processor(FakeProcessor):
    image_token_id = 999
    video_token_id = 998

    def __call__(self, *, text, images, videos, return_tensors, **kwargs):
        batch = super().__call__(text=text, images=images, videos=videos, return_tensors=return_tensors, **kwargs)
        batch["mm_token_type_ids"] = torch.zeros((1, 1), dtype=torch.long)
        if images:
            batch["image_grid_thw"] = torch.tensor([[1, 1, 1]], dtype=torch.long)
        return batch

    def get_rope_index(
        self,
        *,
        input_ids,
        attention_mask,
        mm_token_type_ids,
        image_grid_thw,
        video_grid_thw,
    ):
        del attention_mask, video_grid_thw
        if image_grid_thw is None:
            assert not torch.any(mm_token_type_ids == 1)
        return torch.zeros((3, 1, input_ids.shape[-1]), dtype=torch.long), None


def test_normal_grpo_truncates_response_before_generated_visual_special_token():
    turn_records = np.empty(1, dtype=object)
    turn_records[0] = [
        {
            "turn_index": 1,
            "prompt_ids": [10],
            "response_ids": [20, FakeQwen3Processor.image_token_id, 21],
            "output_format_success": False,
            "tool_args_success": False,
            "turn_response_length": 3,
            "turn_truncated_by_length": False,
            "multi_modal_data": {"images": []},
            "mm_processor_kwargs": {},
        }
    ]
    batch = DataProto.from_dict(
        tensors={"dummy": torch.zeros(1, dtype=torch.long)},
        non_tensors={
            "uid": np.array(["prompt-a"], dtype=object),
            "trajectory_uid": np.array(["trajectory-a"], dtype=object),
            "task_reward": np.array([0.0], dtype=np.float32),
            "turn_records": turn_records,
        },
    )

    result, metrics = build_per_turn_grpo_batch(
        batch,
        tokenizer=FakeTokenizer(),
        processor=FakeQwen3Processor(),
        max_prompt_length=4,
        max_response_length=4,
        trajectory_step_cost=0.0,
        output_format_error_cost=0.5,
        tool_args_error_cost=0.5,
        truncation_error_cost=0.75,
        overlong_buffer_length=0,
        overlong_cost_coef=0.0,
        norm_adv_by_std=False,
    )

    assert result.batch["responses"][0].tolist() == [20, 0, 0, 0]
    assert result.batch["response_mask"][0].tolist() == [1, 0, 0, 0]
    assert metrics["visharness/per_turn/generated_modality_token_truncated"] == 1.0
    assert metrics["visharness/per_turn/generated_modality_token_empty_prefix_dropped"] == 0.0


def test_soft_overlong_switch_does_not_disable_hard_truncation():
    clean_long_turn = {
        "response_ids": [20, 21, 22],
        "turn_response_length": 3,
        "turn_truncated_by_length": False,
        "output_format_success": True,
        "tool_args_success": True,
    }
    common = {
        "max_response_length": 4,
        "output_format_error_cost": 0.5,
        "tool_args_error_cost": 0.5,
        "truncation_error_cost": 0.75,
        "overlong_buffer_length": 2,
        "overlong_cost_coef": 0.6,
    }

    disabled = compute_turn_advantage_shaping(
        clean_long_turn,
        soft_overlong_enabled=False,
        **common,
    )
    enabled = compute_turn_advantage_shaping(
        clean_long_turn,
        soft_overlong_enabled=True,
        **common,
    )

    assert disabled["soft_overlong"] is False
    assert disabled["soft_overlong_cost"] == 0.0
    assert disabled["local_cost"] == 0.0
    assert enabled["soft_overlong"] is True
    assert enabled["soft_overlong_cost"] == pytest.approx(0.3)
    assert enabled["local_cost"] == pytest.approx(0.3)

    truncated_turn = {
        **clean_long_turn,
        "response_ids": [20, 21, 22, 23],
        "turn_response_length": 4,
        "turn_truncated_by_length": True,
        # Real length truncation also fails parsing. It must remain a raw
        # parser failure without becoming an exclusive format-error event.
        "output_format_success": False,
        "tool_args_success": False,
    }
    hard = compute_turn_advantage_shaping(
        truncated_turn,
        soft_overlong_enabled=False,
        **common,
    )
    assert hard["hard_error"] is True
    assert hard["error_type"] == "truncation"
    assert hard["truncated_by_length"] is True
    assert hard["output_format_error"] is False
    assert hard["raw_output_format_failure"] is True
    assert hard["truncation_cost"] == pytest.approx(0.75)
    assert hard["soft_overlong_cost"] == 0.0
    assert hard["local_cost"] == pytest.approx(0.75)


def test_hard_error_rejects_zero_cost_instead_of_producing_nonnegative_advantage():
    with pytest.raises(ValueError, match="must be finite and > 0"):
        compute_turn_advantage_shaping(
            {
                "response_ids": [20],
                "turn_response_length": 1,
                "turn_truncated_by_length": False,
                "output_format_success": False,
                "tool_args_success": False,
            },
            max_response_length=4,
            output_format_error_cost=0.0,
            tool_args_error_cost=0.5,
            truncation_error_cost=0.75,
            overlong_buffer_length=2,
            overlong_cost_coef=0.6,
            soft_overlong_enabled=False,
        )


def test_grpo_error_metrics_are_mutually_exclusive_and_keep_raw_failures():
    turn_records = np.empty(4, dtype=object)
    classifications = [
        # Truncation also fails parsing, as it does in the real agent loop.
        (True, False, False),
        (False, False, False),
        (False, True, False),
        (False, True, True),
    ]
    for index, (truncated, format_success, args_success) in enumerate(classifications):
        response_length = 4 if truncated else 1
        turn_records[index] = [
            {
                "turn_index": 1,
                "prompt_ids": [10 + index],
                "response_ids": list(range(20, 20 + response_length)),
                "turn_response_length": response_length,
                "turn_truncated_by_length": truncated,
                "output_format_success": format_success,
                "tool_args_success": args_success,
                "multi_modal_data": {"images": []},
                "mm_processor_kwargs": {},
            }
        ]
    batch = DataProto.from_dict(
        tensors={"dummy": torch.zeros(4, dtype=torch.long)},
        non_tensors={
            "uid": np.array(["prompt-a"] * 4, dtype=object),
            "trajectory_uid": np.array([f"trajectory-{index}" for index in range(4)], dtype=object),
            "data_source": np.array(["source-a", "source-a", "source-b", "source-b"], dtype=object),
            "task_reward": np.ones(4, dtype=np.float32),
            "turn_records": turn_records,
        },
    )

    result, metrics = build_per_turn_grpo_batch(
        batch,
        tokenizer=FakeTokenizer(),
        max_prompt_length=4,
        max_response_length=4,
        trajectory_step_cost=0.0,
        output_format_error_cost=0.5,
        tool_args_error_cost=0.5,
        truncation_error_cost=0.75,
        overlong_buffer_length=0,
        overlong_cost_coef=0.0,
        norm_adv_by_std=False,
        trajectory_equal_weight=True,
        local_cost_mode="per_event_floor",
        soft_overlong_enabled=False,
    )

    assert result.non_tensor_batch["turn_error_type"].tolist() == [
        "truncation",
        "output_format",
        "tool_args",
        "none",
    ]
    assert result.non_tensor_batch["turn_raw_output_format_failure"].tolist() == [
        True,
        True,
        False,
        False,
    ]
    assert metrics["visharness/per_turn/truncation_error_rate"] == pytest.approx(0.25)
    assert metrics["visharness/per_turn/output_format_error_rate"] == pytest.approx(0.25)
    assert metrics["visharness/per_turn/tool_args_error_rate"] == pytest.approx(0.25)
    assert metrics["visharness/per_turn/hard_error_turn_rate"] == pytest.approx(0.75)
    assert metrics["visharness/per_turn/raw_output_format_failure_rate"] == pytest.approx(0.5)
    assert metrics["visharness/per_turn/raw_tool_args_failure_rate"] == pytest.approx(0.25)

    # Padding rows must not affect the per-source update rates.
    padded, pad_size = pad_per_turn_batch_to_divisor(result, divisor=8)
    assert pad_size == 4
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    update_metrics = {}
    trainer._add_actor_update_sample_metrics(update_metrics, padded)
    assert update_metrics["train/update/per_turn_samples"] == 4.0
    assert update_metrics["train/update/truncation_error_rate"] == pytest.approx(0.25)
    assert update_metrics["train/update/output_format_error_rate"] == pytest.approx(0.25)
    assert update_metrics["train/update/raw_output_format_failure_rate"] == pytest.approx(0.5)
    assert update_metrics[
        "train/update/by_data_source/source-a/truncation_error_rate"
    ] == pytest.approx(0.5)
    assert update_metrics[
        "train/update/by_data_source/source-a/output_format_error_rate"
    ] == pytest.approx(0.5)
    assert update_metrics[
        "train/update/by_data_source/source-a/raw_output_format_failure_rate"
    ] == pytest.approx(1.0)
    assert update_metrics[
        "train/update/by_data_source/source-b/tool_args_error_rate"
    ] == pytest.approx(0.5)
    assert update_metrics[
        "train/update/by_data_source/source-b/hard_error_turn_rate"
    ] == pytest.approx(0.5)


def test_soft_overlong_switch_is_applied_by_ordinary_grpo_batch_builder():
    turn_records = np.empty(1, dtype=object)
    turn_records[0] = [
        {
            "turn_index": 1,
            "prompt_ids": [10],
            "response_ids": [20, 21, 22],
            "turn_response_length": 3,
            "turn_truncated_by_length": False,
            "output_format_success": True,
            "tool_args_success": True,
            "multi_modal_data": {"images": []},
            "mm_processor_kwargs": {},
        }
    ]
    batch = DataProto.from_dict(
        tensors={"dummy": torch.zeros(1, dtype=torch.long)},
        non_tensors={
            "uid": np.array(["prompt-a"], dtype=object),
            "trajectory_uid": np.array(["trajectory-a"], dtype=object),
            "task_reward": np.array([1.0], dtype=np.float32),
            "turn_records": turn_records,
        },
    )
    common = {
        "tokenizer": FakeTokenizer(),
        "max_prompt_length": 4,
        "max_response_length": 4,
        "trajectory_step_cost": 0.0,
        "output_format_error_cost": 0.5,
        "tool_args_error_cost": 0.5,
        "truncation_error_cost": 0.75,
        "overlong_buffer_length": 2,
        "overlong_cost_coef": 0.6,
        "norm_adv_by_std": False,
        "trajectory_equal_weight": True,
        "local_cost_mode": "per_event_floor",
    }

    disabled, disabled_metrics = build_per_turn_grpo_batch(
        batch,
        soft_overlong_enabled=False,
        **common,
    )
    enabled, enabled_metrics = build_per_turn_grpo_batch(
        batch,
        soft_overlong_enabled=True,
        **common,
    )

    assert disabled.non_tensor_batch["turn_error_type"].tolist() == ["none"]
    assert disabled.non_tensor_batch["effective_policy_advantage"].tolist() == [0.0]
    assert disabled_metrics["visharness/per_turn/soft_overlong_enabled"] == 0.0
    assert enabled.non_tensor_batch["turn_error_type"].tolist() == ["soft_overlong"]
    assert enabled.non_tensor_batch["effective_policy_advantage"].tolist() == pytest.approx([-0.3])
    assert enabled_metrics["visharness/per_turn/soft_overlong_enabled"] == 1.0


def _make_trajectory_weighting_batch():
    turn_records = np.empty(2, dtype=object)
    turn_records[0] = [
        {
            "turn_index": 1,
            "prompt_ids": [10],
            "response_ids": [20],
            "response_logprobs": [-0.1],
            "output_format_success": False,
            "tool_args_success": False,
            "turn_response_length": 1,
            "turn_truncated_by_length": False,
            "visible_image_names": [],
            "multi_modal_data": {"images": []},
            "mm_processor_kwargs": {},
        },
        {
            "turn_index": 2,
            "prompt_ids": [10, 20],
            "response_ids": [21],
            "response_logprobs": [-0.2],
            "output_format_success": True,
            "tool_args_success": True,
            "turn_response_length": 1,
            "turn_truncated_by_length": False,
            "visible_image_names": [],
            "multi_modal_data": {"images": []},
            "mm_processor_kwargs": {},
        },
    ]
    turn_records[1] = [
        {
            "turn_index": 1,
            "prompt_ids": [10],
            "response_ids": [22],
            "output_format_success": True,
            "tool_args_success": True,
            "turn_response_length": 1,
            "turn_truncated_by_length": False,
            "visible_image_names": [],
            "multi_modal_data": {"images": []},
            "mm_processor_kwargs": {},
        },
        {
            "turn_index": 2,
            "prompt_ids": [10, 22],
            "response_ids": [23],
            "output_format_success": True,
            "tool_args_success": True,
            "turn_response_length": 1,
            "turn_truncated_by_length": False,
            "multi_modal_data": {"images": []},
            "mm_processor_kwargs": {},
        },
        {
            "turn_index": 3,
            "prompt_ids": [10, 22, 23],
            "response_ids": [24],
            "output_format_success": True,
            "tool_args_success": True,
            "turn_response_length": 1,
            "turn_truncated_by_length": False,
            "multi_modal_data": {"images": []},
            "mm_processor_kwargs": {},
        },
    ]
    return DataProto.from_dict(
        tensors={"dummy": torch.zeros(2, dtype=torch.long)},
        non_tensors={
            "uid": np.array(["prompt-a", "prompt-a"], dtype=object),
            "trajectory_uid": np.array(["trajectory-a", "trajectory-b"], dtype=object),
            "data_source": np.array(
                ["visharness/rec8k", "visharness/rec8k"],
                dtype=object,
            ),
            "task_reward": np.array([5.0, 5.0], dtype=np.float32),
            "turn_records": turn_records,
        },
    )


def test_grpo_assigns_trajectory_advantage_and_clamps_hard_error_turns():
    batch = _make_trajectory_weighting_batch()

    result, metrics = build_per_turn_grpo_batch(
        batch,
        tokenizer=FakeTokenizer(),
        processor=FakeProcessor(),
        max_prompt_length=6,
        max_response_length=4,
        trajectory_step_cost=0.2,
        output_format_error_cost=0.5,
        tool_args_error_cost=0.5,
        truncation_error_cost=0.75,
        overlong_buffer_length=0,
        overlong_cost_coef=0.0,
        norm_adv_by_std=False,
        trajectory_equal_weight=True,
    )

    assert result.non_tensor_batch["trajectory_reward"].tolist() == pytest.approx([4.6, 4.6, 4.4, 4.4, 4.4])
    assert result.non_tensor_batch["trajectory_advantage"].tolist() == pytest.approx([0.1, 0.1, -0.1, -0.1, -0.1])
    assert result.non_tensor_batch["turn_error_type"].tolist() == [
        "output_format",
        "none",
        "none",
        "none",
        "none",
    ]
    assert result.non_tensor_batch["turn_local_advantage_cost"].tolist() == pytest.approx(
        [0.5, 0.0, 0.0, 0.0, 0.0]
    )
    assert result.non_tensor_batch["turn_advantage"].tolist() == pytest.approx(
        [-0.5, 0.1, -0.1, -0.1, -0.1]
    )
    assert result.non_tensor_batch["trajectory_trainable_turn_count"].tolist() == [2, 2, 3, 3, 3]
    assert result.non_tensor_batch["trajectory_loss_weight"].tolist() == pytest.approx(
        [1.25, 1.25, 5.0 / 6.0, 5.0 / 6.0, 5.0 / 6.0]
    )
    assert result.non_tensor_batch["weighted_turn_advantage"].tolist() == pytest.approx(
        [-0.625, 0.125, -1.0 / 12.0, -1.0 / 12.0, -1.0 / 12.0]
    )
    assert result.batch["advantages"][:, 0].tolist() == pytest.approx([-0.5, 0.1, -0.1, -0.1, -0.1])
    assert result.batch[TRAJECTORY_LOSS_WEIGHT_KEY][:, 0].tolist() == pytest.approx(
        [1.25, 1.25, 5.0 / 6.0, 5.0 / 6.0, 5.0 / 6.0]
    )
    assert metrics["visharness/per_turn/trajectory_loss_weight_mean"] == pytest.approx(1.0)
    assert metrics["visharness/per_turn/trajectory_loss_mass_mean"] == pytest.approx(2.5)
    assert metrics["visharness/per_turn/trajectory_loss_mass_std"] == pytest.approx(0.0)

    padded, pad_size = pad_per_turn_batch_to_divisor(result, divisor=8)
    assert pad_size == 3
    assert padded.batch[TRAJECTORY_LOSS_WEIGHT_KEY][-3:].sum().item() == 0.0


def test_per_event_floor_keeps_local_cost_outside_trajectory_weight():
    result, metrics = build_per_turn_grpo_batch(
        _make_trajectory_weighting_batch(),
        tokenizer=FakeTokenizer(),
        processor=FakeProcessor(),
        max_prompt_length=6,
        max_response_length=4,
        trajectory_step_cost=0.2,
        output_format_error_cost=0.5,
        tool_args_error_cost=0.5,
        truncation_error_cost=0.75,
        overlong_buffer_length=0,
        overlong_cost_coef=0.0,
        norm_adv_by_std=False,
        trajectory_equal_weight=True,
        local_cost_mode="per_event_floor",
        soft_overlong_enabled=False,
    )

    # The first trajectory has two trainable turns, so q = 5 / (2 * 2) = 1.25.
    # Its hard-error task advantage is positive and therefore the final actor
    # coefficient is min(q * 0.1, -0.5) = -0.5, not q * (-0.5) = -0.625.
    assert result.non_tensor_batch["turn_advantage"].tolist() == pytest.approx(
        [-0.5, 0.1, -0.1, -0.1, -0.1]
    )
    assert result.non_tensor_batch["local_cost_mode"].tolist() == ["per_event_floor"] * 5
    assert result.non_tensor_batch["ppo_turn_advantage"].tolist() == pytest.approx(
        [-0.4, 0.1, -0.1, -0.1, -0.1]
    )
    assert result.non_tensor_batch["effective_policy_advantage"].tolist() == pytest.approx(
        [-0.5, 0.125, -1.0 / 12.0, -1.0 / 12.0, -1.0 / 12.0]
    )
    assert result.non_tensor_batch["effective_policy_advantage"][0] < 0
    assert metrics["visharness/per_turn/soft_overlong_enabled"] == 0.0
    assert result.non_tensor_batch["effective_local_cost"].tolist() == pytest.approx(
        [0.5, 0.0, 0.0, 0.0, 0.0]
    )
    assert result.non_tensor_batch["local_floor_applied"].tolist() == [True, False, False, False, False]
    assert result.non_tensor_batch["weighted_turn_advantage"].tolist() == pytest.approx(
        [-0.5, 0.125, -1.0 / 12.0, -1.0 / 12.0, -1.0 / 12.0]
    )

    # The tensor sent into PPO contains the pre-compensated value -c / q.
    # Multiplying it once by the trajectory loss weight recovers the desired
    # per-event coefficient. This catches accidental double weighting.
    tensor_advantages = result.batch["advantages"][:, 0]
    tensor_weights = result.batch[TRAJECTORY_LOSS_WEIGHT_KEY][:, 0]
    assert tensor_advantages.tolist() == pytest.approx([-0.4, 0.1, -0.1, -0.1, -0.1])
    assert (tensor_advantages * tensor_weights).tolist() == pytest.approx(
        [-0.5, 0.125, -1.0 / 12.0, -1.0 / 12.0, -1.0 / 12.0]
    )
    assert result.batch["returns"][:, 0].tolist() == pytest.approx(
        [-0.4, 0.1, -0.1, -0.1, -0.1]
    )
    assert metrics["visharness/per_turn/local_cost_per_event_floor_enabled"] == 1.0
    assert metrics["visharness/per_turn/final_turn_advantage_mean"] == pytest.approx(-0.125)
    assert metrics["visharness/per_turn/legacy_additive_turn_advantage_mean"] == pytest.approx(-0.14)
    assert metrics["visharness/per_turn/local_floor_applied_rate"] == pytest.approx(0.2)
    assert metrics["visharness/per_turn/hard_error_local_floor_applied_rate"] == 1.0


def test_per_event_floor_composes_hard_soft_and_clean_advantages_exactly():
    # Four turns across three trajectories give q = 4/3 for each one-turn
    # trajectory and q = 2/3 for each turn in the two-turn trajectory.
    records = [
        {
            "trajectory_index": 0,
            "assigned_trajectory_advantage": 0.25,
            "base_advantage": 0.0,
            "local_cost": 0.5,
            "turn_advantage": -0.5,
            "hard_error": True,
        },
        {
            "trajectory_index": 1,
            "assigned_trajectory_advantage": -1.0,
            "base_advantage": -1.0,
            "local_cost": 0.5,
            "turn_advantage": -1.5,
            "hard_error": True,
        },
        {
            "trajectory_index": 2,
            "assigned_trajectory_advantage": 0.25,
            "base_advantage": 0.25,
            "local_cost": 0.2,
            "turn_advantage": 0.05,
            "hard_error": False,
        },
        {
            "trajectory_index": 2,
            "assigned_trajectory_advantage": 0.25,
            "base_advantage": 0.25,
            "local_cost": 0.0,
            "turn_advantage": 0.25,
            "hard_error": False,
        },
    ]

    metrics = _assign_trajectory_loss_weights(
        records,
        enabled=True,
        local_cost_mode="per_event_floor",
    )

    assert [record["trajectory_loss_weight"] for record in records] == pytest.approx(
        [4.0 / 3.0, 4.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]
    )
    # Hard positive: min(q*A, -c) = min(1/3, -1/2) = -1/2.
    # Hard strong negative: min(-4/3, -1/2) = -4/3, so c is not
    # subtracted again. Soft: q*base-c = (2/3)*(1/4)-1/5 = -1/30.
    # Clean: only the task/base term is weighted, yielding 1/6.
    assert [record["effective_policy_advantage"] for record in records] == pytest.approx(
        [-0.5, -4.0 / 3.0, -1.0 / 30.0, 1.0 / 6.0]
    )
    assert [record["ppo_turn_advantage"] for record in records] == pytest.approx(
        [-0.375, -1.0, -0.05, 0.25]
    )
    assert [record["effective_local_cost"] for record in records] == pytest.approx(
        [0.5, 0.0, 0.2, 0.0]
    )
    assert [record["local_floor_applied"] for record in records] == [True, False, False, False]
    assert metrics["visharness/per_turn/effective_positive_advantage_count"] == 1.0
    assert metrics["visharness/per_turn/effective_negative_advantage_count"] == 3.0
    assert metrics["visharness/per_turn/effective_zero_advantage_count"] == 0.0
    assert metrics["visharness/per_turn/effective_positive_advantage_rate"] == pytest.approx(0.25)
    assert metrics["visharness/per_turn/effective_negative_advantage_rate"] == pytest.approx(0.75)
    assert metrics["visharness/per_turn/effective_zero_advantage_rate"] == 0.0
    assert metrics["visharness/per_turn/effective_positive_advantage_mass"] == pytest.approx(1.0 / 6.0)
    assert metrics["visharness/per_turn/effective_negative_advantage_mass"] == pytest.approx(28.0 / 15.0)
    assert metrics["visharness/per_turn/effective_net_advantage_mass"] == pytest.approx(-1.7)
    assert metrics["visharness/per_turn/effective_positive_advantage_mass_share"] == pytest.approx(5.0 / 61.0)
    assert metrics["visharness/per_turn/effective_negative_advantage_mass_share"] == pytest.approx(56.0 / 61.0)


def test_effective_advantage_metrics_distinguish_sample_counts_from_signal_mass():
    records = [
        {
            "trajectory_index": trajectory_index,
            "assigned_trajectory_advantage": advantage,
            "base_advantage": advantage,
            "local_cost": 0.0,
            "turn_advantage": advantage,
            "hard_error": False,
        }
        for trajectory_index, advantage in enumerate([1.0, 1.0, -3.0, 0.0])
    ]

    metrics = _assign_trajectory_loss_weights(
        records,
        enabled=True,
        local_cost_mode="per_event_floor",
    )

    # Positive turns are more numerous (2 vs 1), but the negative coefficient
    # mass is larger (3 vs 2), so this actor batch is negative-signal dominated.
    assert metrics["visharness/per_turn/effective_positive_advantage_count"] == 2.0
    assert metrics["visharness/per_turn/effective_negative_advantage_count"] == 1.0
    assert metrics["visharness/per_turn/effective_zero_advantage_count"] == 1.0
    assert metrics["visharness/per_turn/effective_positive_advantage_rate"] == pytest.approx(0.5)
    assert metrics["visharness/per_turn/effective_negative_advantage_rate"] == pytest.approx(0.25)
    assert metrics["visharness/per_turn/effective_zero_advantage_rate"] == pytest.approx(0.25)
    assert metrics["visharness/per_turn/effective_positive_advantage_mass"] == pytest.approx(2.0)
    assert metrics["visharness/per_turn/effective_negative_advantage_mass"] == pytest.approx(3.0)
    assert metrics["visharness/per_turn/effective_net_advantage_mass"] == pytest.approx(-1.0)
    assert metrics["visharness/per_turn/effective_positive_advantage_mass_share"] == pytest.approx(0.4)
    assert metrics["visharness/per_turn/effective_negative_advantage_mass_share"] == pytest.approx(0.6)


def test_same_hard_event_cost_is_fixed_for_short_and_long_trajectories():
    records = [
        {
            "trajectory_index": 0,
            "assigned_trajectory_advantage": 0.25,
            "base_advantage": 0.0,
            "local_cost": 0.5,
            "turn_advantage": -0.5,
            "hard_error": True,
        },
        {
            "trajectory_index": 1,
            "assigned_trajectory_advantage": 0.25,
            "base_advantage": 0.0,
            "local_cost": 0.5,
            "turn_advantage": -0.5,
            "hard_error": True,
        },
        {
            "trajectory_index": 1,
            "assigned_trajectory_advantage": 0.25,
            "base_advantage": 0.25,
            "local_cost": 0.0,
            "turn_advantage": 0.25,
            "hard_error": False,
        },
        {
            "trajectory_index": 1,
            "assigned_trajectory_advantage": 0.25,
            "base_advantage": 0.25,
            "local_cost": 0.0,
            "turn_advantage": 0.25,
            "hard_error": False,
        },
    ]

    _assign_trajectory_loss_weights(
        records,
        enabled=True,
        local_cost_mode="per_event_floor",
    )

    # M=4, N=2: the one-turn trajectory has q=2 while every turn in the
    # three-turn trajectory has q=2/3. The same hard event remains exactly
    # -0.5 in both cases.
    assert [record["trajectory_loss_weight"] for record in records] == pytest.approx(
        [2.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]
    )
    assert [record["effective_policy_advantage"] for record in records[:2]] == pytest.approx(
        [-0.5, -0.5]
    )
    assert [record["ppo_turn_advantage"] for record in records[:2]] == pytest.approx(
        [-0.25, -0.75]
    )


def test_trainer_local_cost_mode_defaults_validates_and_controls_weight_correction():
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.config = OmegaConf.create(
        {
            "visharness": {
                "per_turn": {"trajectory_equal_weight": False},
                "per_turn_advantage": {},
            }
        }
    )
    assert trainer._get_per_turn_local_cost_mode() == "per_event_floor"
    assert trainer._per_turn_loss_weight_correction_enabled() is True

    trainer.config.visharness.per_turn_advantage.local_cost_mode = "trajectory_weighted_additive"
    assert trainer._get_per_turn_local_cost_mode() == "trajectory_weighted_additive"
    assert trainer._per_turn_loss_weight_correction_enabled() is False

    trainer.config.visharness.per_turn.trajectory_equal_weight = True
    with pytest.raises(ValueError, match="trajectory_equal_weight=true requires"):
        trainer._get_per_turn_local_cost_mode()
    trainer.config.visharness.per_turn.trajectory_equal_weight = False

    trainer.config.visharness.per_turn_advantage.local_cost_mode = "unknown"
    with pytest.raises(ValueError, match="local_cost_mode must be one of"):
        trainer._get_per_turn_local_cost_mode()


def test_trainer_soft_overlong_setting_defaults_off_and_can_be_enabled():
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.config = OmegaConf.create(
        {
            "visharness": {
                "per_turn": {"max_response_length": 8},
                "per_turn_advantage": {
                    "output_format_error_cost": 0.5,
                    "tool_args_error_cost": 0.5,
                    "truncation_error_cost": 0.75,
                },
            }
        }
    )

    assert trainer._get_per_turn_advantage_settings()["soft_overlong_enabled"] is False
    trainer.config.visharness.per_turn_advantage.soft_overlong_enabled = True
    assert trainer._get_per_turn_advantage_settings()["soft_overlong_enabled"] is True


def test_grpo_step_cost_is_applied_once_at_trajectory_level():
    turn_records = np.empty(2, dtype=object)
    turn_records[0] = [
        {
            "turn_index": 1,
            "prompt_ids": [10],
            "response_ids": [20],
            "output_format_success": True,
            "tool_args_success": True,
            "turn_response_length": 1,
            "turn_truncated_by_length": False,
            "multi_modal_data": {"images": []},
            "mm_processor_kwargs": {},
        },
    ]
    turn_records[1] = [
        {
            "turn_index": 1,
            "prompt_ids": [10, 20],
            "response_ids": [21],
            "output_format_success": True,
            "tool_args_success": True,
            "turn_response_length": 1,
            "turn_truncated_by_length": False,
            "multi_modal_data": {"images": []},
            "mm_processor_kwargs": {},
        },
        {
            "turn_index": 2,
            "prompt_ids": [10, 20, 21],
            "response_ids": [22],
            "output_format_success": True,
            "tool_args_success": True,
            "turn_response_length": 1,
            "turn_truncated_by_length": False,
            "multi_modal_data": {"images": []},
            "mm_processor_kwargs": {},
        },
        {
            "turn_index": 3,
            "prompt_ids": [10, 20, 21, 22],
            "response_ids": [23],
            "output_format_success": True,
            "tool_args_success": True,
            "turn_response_length": 1,
            "turn_truncated_by_length": False,
            "multi_modal_data": {"images": []},
            "mm_processor_kwargs": {},
        },
    ]
    batch = DataProto.from_dict(
        tensors={"dummy": torch.zeros(2, dtype=torch.long)},
        non_tensors={
            "uid": np.array(["prompt-a", "prompt-a"], dtype=object),
            "trajectory_uid": np.array(["trajectory-a", "trajectory-b"], dtype=object),
            "task_reward": np.array([5.0, 5.0], dtype=np.float32),
            "turn_records": turn_records,
        },
    )

    result, _ = build_per_turn_grpo_batch(
        batch,
        tokenizer=FakeTokenizer(),
        processor=FakeProcessor(),
        max_prompt_length=6,
        max_response_length=4,
        trajectory_step_cost=0.2,
        output_format_error_cost=0.5,
        tool_args_error_cost=0.5,
        truncation_error_cost=0.75,
        overlong_buffer_length=0,
        overlong_cost_coef=0.0,
        norm_adv_by_std=False,
    )

    assert result.non_tensor_batch["trajectory_step_cost"].tolist() == pytest.approx([0.2, 0.6, 0.6, 0.6])
    assert result.non_tensor_batch["trajectory_reward"].tolist() == pytest.approx([4.8, 4.4, 4.4, 4.4])
    assert result.batch["advantages"][:, 0].tolist() == pytest.approx([0.2, -0.2, -0.2, -0.2])
    assert result.batch[TRAJECTORY_LOSS_WEIGHT_KEY][:, 0].tolist() == pytest.approx([1.0] * 4)


def test_grpo_trajectory_baseline_is_computed_before_prompt_overlong_turns_are_dropped():
    turn_records = np.empty(2, dtype=object)
    turn_records[0] = [
        {
            "turn_index": 1,
            "prompt_ids": [10],
            "response_ids": [20],
            "output_format_success": True,
            "tool_args_success": True,
            "turn_response_length": 1,
            "turn_truncated_by_length": False,
            "multi_modal_data": {"images": []},
            "mm_processor_kwargs": {},
        },
        {
            "turn_index": 2,
            "prompt_ids": [10],
            "response_ids": [21],
            "output_format_success": True,
            "tool_args_success": False,
            "turn_response_length": 1,
            "turn_truncated_by_length": False,
            "multi_modal_data": {"images": []},
            "mm_processor_kwargs": {},
        },
        {
            "turn_index": 3,
            "prompt_ids": [10, 11, 12],
            "response_ids": [22],
            "output_format_success": True,
            "tool_args_success": True,
            "turn_response_length": 1,
            "turn_truncated_by_length": False,
            "multi_modal_data": {"images": []},
            "mm_processor_kwargs": {},
        },
    ]
    turn_records[1] = [
        {
            "turn_index": 1,
            "prompt_ids": [30],
            "response_ids": [31],
            "output_format_success": True,
            "tool_args_success": True,
            "turn_response_length": 1,
            "turn_truncated_by_length": False,
            "multi_modal_data": {"images": []},
            "mm_processor_kwargs": {},
        }
    ]
    batch = DataProto.from_dict(
        tensors={"dummy": torch.zeros(2, dtype=torch.long)},
        non_tensors={
            "uid": np.array(["prompt-a", "prompt-a"], dtype=object),
            "trajectory_uid": np.array(["trajectory-a", "trajectory-b"], dtype=object),
            "task_reward": np.array([2.0, 0.0], dtype=np.float32),
            "turn_records": turn_records,
        },
    )

    result, metrics = build_per_turn_grpo_batch(
        batch,
        tokenizer=FakeTokenizer(),
        processor=FakeProcessor(),
        max_prompt_length=2,
        max_response_length=4,
        trajectory_step_cost=0.0,
        output_format_error_cost=0.5,
        tool_args_error_cost=0.5,
        truncation_error_cost=0.75,
        overlong_buffer_length=0,
        overlong_cost_coef=0.0,
        norm_adv_by_std=False,
        trajectory_equal_weight=True,
    )

    assert result.non_tensor_batch["trajectory_uid"].tolist() == ["trajectory-a", "trajectory-a", "trajectory-b"]
    assert result.non_tensor_batch["turn_index"].tolist() == [1, 2, 1]
    assert result.non_tensor_batch["trajectory_advantage"].tolist() == pytest.approx([1.0, 1.0, -1.0])
    assert result.batch["advantages"][:, 0].tolist() == pytest.approx([1.0, -0.5, -1.0])
    assert result.non_tensor_batch["trajectory_trainable_turn_count"].tolist() == [2, 2, 1]
    assert result.batch[TRAJECTORY_LOSS_WEIGHT_KEY][:, 0].tolist() == pytest.approx([0.75, 0.75, 1.5])
    assert metrics["visharness/per_turn/trainable_turns_per_trajectory_mean"] == pytest.approx(1.5)
    assert metrics["visharness/per_turn/trajectory_loss_mass_std"] == pytest.approx(0.0)
    assert metrics["visharness/per_turn/prompt_overlong_dropped"] == 1.0



@pytest.mark.parametrize(
    ("batch_size", "dp_size", "expected"),
    [
        (409, 4, (416, 7, 208)),
        (534, 4, (536, 2, 268)),
        (409, 8, (416, 7, 208)),
        (534, 8, (544, 10, 272)),
    ],
)
def test_fixed_per_turn_mini_batch_plan_keeps_two_optimizer_updates(batch_size, dp_size, expected):
    plan = plan_fixed_per_turn_mini_batches(
        batch_size=batch_size,
        num_mini_batches=2,
        dp_size=dp_size,
    )

    assert plan == expected
    padded_batch_size, _, global_mini_batch_size = plan
    assert padded_batch_size // global_mini_batch_size == 2
    assert global_mini_batch_size % dp_size == 0


def test_fixed_per_turn_mini_batch_plan_rejects_invalid_values():
    with pytest.raises(ValueError, match="must be positive"):
        plan_fixed_per_turn_mini_batches(batch_size=0, num_mini_batches=2, dp_size=4)
    with pytest.raises(ValueError, match="must be positive"):
        plan_fixed_per_turn_mini_batches(batch_size=64, num_mini_batches=0, dp_size=4)


def test_trainer_derives_fixed_per_turn_mini_batch_count_from_prompt_batch():
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": 8},
            "actor_rollout_ref": {"actor": {"ppo_mini_batch_size": 4}},
            "visharness": {"per_turn": {"num_mini_batches": None}},
        }
    )

    assert trainer._get_per_turn_num_mini_batches() == 2


def test_trainer_prefers_explicit_fixed_per_turn_mini_batch_count():
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": 8},
            "actor_rollout_ref": {"actor": {"ppo_mini_batch_size": 4}},
            "visharness": {"per_turn": {"num_mini_batches": 3}},
        }
    )

    assert trainer._get_per_turn_num_mini_batches() == 3


def make_filter_batch(uids, metric_vals):
    return DataProto.from_dict(
        tensors={"dummy": torch.zeros(len(uids), dtype=torch.long)},
        non_tensors={
            "uid": np.array(uids, dtype=object),
            "trajectory_reward": np.array(metric_vals, dtype=np.float64),
        },
    )


def test_drop_invalid_trajectories_keeps_valid_rows_and_counts_reasons():
    batch = DataProto.from_dict(
        tensors={"dummy": torch.arange(4, dtype=torch.long)},
        non_tensors={
            "uid": np.array(["prompt-a", "prompt-b", "prompt-c", "prompt-d"], dtype=object),
            "trajectory_invalid": np.array([False, True, True, None], dtype=object),
            "invalid_reason": np.array(
                [None, "rollout_prompt_overlong", "tool_oom_retry_exhausted", None],
                dtype=object,
            ),
        },
    )

    kept, dropped, reason_counts = VisHarnessTrainer._drop_invalid_trajectories(batch)

    assert dropped == 2
    assert reason_counts == {
        "rollout_prompt_overlong": 1,
        "tool_oom_retry_exhausted": 1,
    }
    assert kept is not None
    assert kept.batch["dummy"].tolist() == [0, 3]
    assert kept.non_tensor_batch["uid"].tolist() == ["prompt-a", "prompt-d"]


def test_drop_invalid_trajectories_returns_none_when_all_dropped():
    batch = DataProto.from_dict(
        tensors={"dummy": torch.arange(2, dtype=torch.long)},
        non_tensors={
            "uid": np.array(["prompt-a", "prompt-b"], dtype=object),
            "trajectory_invalid": np.array([True, True], dtype=object),
            "invalid_reason": np.array(
                ["tool_oom_retry_exhausted", "rollout_prompt_overlong"], dtype=object
            ),
        },
    )

    kept, dropped, reason_counts = VisHarnessTrainer._drop_invalid_trajectories(batch)

    assert kept is None
    assert dropped == 2
    assert reason_counts == {
        "tool_oom_retry_exhausted": 1,
        "rollout_prompt_overlong": 1,
    }


def test_drop_invalid_trajectories_supports_legacy_prompt_overlong_flag():
    batch = DataProto.from_dict(
        tensors={"dummy": torch.arange(2, dtype=torch.long)},
        non_tensors={
            "uid": np.array(["prompt-a", "prompt-b"], dtype=object),
            "rollout_prompt_overlong": np.array([False, True], dtype=object),
        },
    )

    kept, dropped, reason_counts = VisHarnessTrainer._drop_invalid_trajectories(batch)

    assert kept is not None
    assert kept.batch["dummy"].tolist() == [0]
    assert dropped == 1
    assert reason_counts == {"rollout_prompt_overlong": 1}


def test_drop_undersized_prompt_groups_drops_entire_group_below_minimum():
    batch = DataProto.from_dict(
        tensors={"dummy": torch.arange(9, dtype=torch.long)},
        non_tensors={
            "uid": np.array(
                ["prompt-a"] * 3 + ["prompt-b"] * 4 + ["prompt-c"] * 2,
                dtype=object,
            ),
        },
    )

    kept, dropped_trajectories, dropped_prompt_groups = VisHarnessTrainer._drop_undersized_prompt_groups(
        batch,
        min_trajectories=4,
    )

    assert kept is not None
    assert kept.batch["dummy"].tolist() == [3, 4, 5, 6]
    assert kept.non_tensor_batch["uid"].tolist() == ["prompt-b"] * 4
    assert dropped_trajectories == 5
    assert dropped_prompt_groups == 2


def test_drop_undersized_prompt_groups_returns_none_when_every_group_is_too_small():
    batch = DataProto.from_dict(
        tensors={"dummy": torch.arange(5, dtype=torch.long)},
        non_tensors={
            "uid": np.array(["prompt-a"] * 3 + ["prompt-b"] * 2, dtype=object),
        },
    )

    kept, dropped_trajectories, dropped_prompt_groups = VisHarnessTrainer._drop_undersized_prompt_groups(
        batch,
        min_trajectories=4,
    )

    assert kept is None
    assert dropped_trajectories == 5
    assert dropped_prompt_groups == 2


def test_disabled_group_filter_keeps_undersized_surviving_prompt_groups():
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"filter_groups": {"enable": False}},
            "visharness": {
                "filter_groups": {"min_trajectories_after_rollout_filter": 4},
                "task_sampling": {"enable": False},
            },
        }
    )
    trainer._task_sampling_plan = None
    batch = DataProto.from_dict(
        tensors={"dummy": torch.arange(3, dtype=torch.long)},
        non_tensors={"uid": np.array(["prompt-a"] * 3, dtype=object)},
    )

    kept, dropped_trajectories, dropped_prompt_groups = trainer._maybe_drop_undersized_prompt_groups(
        batch
    )

    assert kept is batch
    assert kept.batch["dummy"].tolist() == [0, 1, 2]
    assert dropped_trajectories == 0
    assert dropped_prompt_groups == 0

    trainer.config.algorithm.filter_groups.enable = True
    kept, dropped_trajectories, dropped_prompt_groups = trainer._maybe_drop_undersized_prompt_groups(
        batch
    )
    assert kept is None
    assert dropped_trajectories == 3
    assert dropped_prompt_groups == 1


def test_task_sampling_keeps_minimum_group_size_filter_when_reward_filter_is_disabled():
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer._task_sampling_plan = object()
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"filter_groups": {"enable": False}},
            "visharness": {
                "filter_groups": {"min_trajectories_after_rollout_filter": 4},
                "task_sampling": {"enable": True},
            },
        }
    )
    batch = DataProto.from_dict(
        tensors={"dummy": torch.arange(7, dtype=torch.long)},
        non_tensors={
            "uid": np.array(["prompt-small"] * 3 + ["prompt-valid"] * 4)
        },
    )

    kept, dropped_trajectories, dropped_prompt_groups = (
        trainer._maybe_drop_undersized_prompt_groups(batch)
    )

    assert kept.non_tensor_batch["uid"].tolist() == ["prompt-valid"] * 4
    assert kept.batch["dummy"].tolist() == [3, 4, 5, 6]
    assert dropped_trajectories == 3
    assert dropped_prompt_groups == 1


def test_restored_raw_prompt_buffer_is_consumed_before_next_dataloader_batch():
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer._visharness_raw_prompt_buffer = DataProto.from_dict(
        tensors={"prompt_id": torch.arange(2, 8, dtype=torch.long)},
        non_tensors={"uid": np.array([f"prompt-{idx}" for idx in range(2, 8)], dtype=object)},
    )
    future_batches = iter(
        [
            DataProto.from_dict(
                tensors={"prompt_id": torch.arange(8, 16, dtype=torch.long)},
                non_tensors={
                    "uid": np.array([f"prompt-{idx}" for idx in range(8, 16)], dtype=object)
                },
            )
        ]
    )

    selected = trainer._take_raw_prompts(8, lambda: next(future_batches, None))

    assert selected is not None
    assert selected.batch["prompt_id"].tolist() == list(range(2, 10))
    assert selected.non_tensor_batch["uid"].tolist() == [f"prompt-{idx}" for idx in range(2, 10)]
    assert trainer._visharness_raw_prompt_buffer is not None
    assert trainer._visharness_raw_prompt_buffer.batch["prompt_id"].tolist() == list(range(10, 16))


def test_visharness_resume_state_round_trips_raw_prompt_buffer(tmp_path):
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.global_steps = 12
    trainer._visharness_train_epoch = 0
    trainer.config = OmegaConf.create(
        {"trainer": {"default_local_dir": str(tmp_path / "checkpoints")}}
    )
    trainer._visharness_raw_prompt_buffer = DataProto.from_dict(
        tensors={"prompt_id": torch.tensor([6, 7], dtype=torch.long)},
        non_tensors={
            "uid": np.array(["prompt-6", "prompt-7"], dtype=object),
            "payload": np.array([{"image": b"abc"}, {"image": b"def"}], dtype=object),
        },
        meta_info={"temperature": 1.0},
    )

    trainer._save_visharness_resume_state()

    state_path = tmp_path / "checkpoints" / "global_step_12" / "visharness_resume_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    restored = trainer._raw_prompt_buffer_from_resume_state(state)
    assert state["version"] == 5
    assert state["global_steps"] == 12
    assert state["train_epoch"] == 0
    assert state["epoch_start_global_step"] == 0
    assert state["task_sampling"] == {"enabled": False}
    assert state["task_source_pool_state"] is None
    assert state["train_dataset_signature"] is None
    assert state["dataloader_state_signature"] is None
    assert state["training_completed"] is False
    assert state["completion_reason"] is None
    assert state["last_validation_step"] is None
    assert restored is not None
    assert restored.batch["prompt_id"].tolist() == [6, 7]
    assert restored.non_tensor_batch["uid"].tolist() == ["prompt-6", "prompt-7"]
    assert restored.non_tensor_batch["payload"][1]["image"] == b"def"
    assert restored.meta_info == {"temperature": 1.0}


def test_visharness_resume_state_is_atomically_replaced(tmp_path, monkeypatch):
    state_path = tmp_path / "resume.pt"
    VisHarnessTrainer._atomic_torch_save({"value": "old"}, str(state_path))

    def fail_during_save(value, file):
        del value
        file.write(b"partial-new-state")
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(torch, "save", fail_during_save)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        VisHarnessTrainer._atomic_torch_save({"value": "new"}, str(state_path))

    assert torch.load(state_path, map_location="cpu", weights_only=False) == {"value": "old"}
    assert list(tmp_path.glob("resume.pt.tmp-*")) == []


def test_checkpoint_persists_visharness_state_before_base_publication(monkeypatch):
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    events = []
    monkeypatch.setattr(
        trainer,
        "_save_visharness_resume_state",
        lambda: events.append("visharness_state"),
    )
    monkeypatch.setattr(
        "recipe.dapo.dapo_ray_trainer.RayDAPOTrainer._save_checkpoint",
        lambda self: events.append("base_checkpoint"),
    )
    monkeypatch.setattr(trainer, "_archive_hf_checkpoint_enabled", lambda: False)
    monkeypatch.setattr(
        trainer,
        "_prune_resume_checkpoint_roles",
        lambda: events.append("checkpoint_retention"),
    )

    trainer._save_checkpoint()

    assert events == [
        "visharness_state",
        "base_checkpoint",
        "checkpoint_retention",
    ]


def test_resume_rejects_checkpoint_missing_visharness_state(tmp_path):
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.global_steps = 12
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(tmp_path / "checkpoints"),
                "resume_mode": "auto",
            }
        }
    )

    with pytest.raises(FileNotFoundError, match="cannot be resumed safely"):
        trainer._load_visharness_resume_state()


def test_natural_exhaustion_finalizes_last_completed_step():
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.global_steps = 11
    trainer.config = OmegaConf.create({"trainer": {"test_freq": 10}})
    events = []
    trainer._save_checkpoint = lambda: events.append(("save", trainer.global_steps))
    trainer._save_visharness_resume_state = lambda: events.append(
        ("validation-state", trainer.global_steps)
    )
    trainer._validate = lambda: events.append(("validate", trainer.global_steps)) or {"val": 1.0}
    trainer._filter_logged_metrics = lambda metrics: metrics

    class Logger:
        def log(self, *, data, step):
            events.append(("log", step, dict(data)))

    class ProgressBar:
        def close(self):
            events.append(("close", trainer.global_steps))

    trainer._finalize_naturally_exhausted_training(
        last_completed_step=10,
        last_validation_step=0,
        last_saved_step=0,
        logger=Logger(),
        progress_bar=ProgressBar(),
        refresh_update_progress=lambda step, consumed: events.append(
            ("progress", step, consumed)
        ),
        consumed_prompt_examples=123,
    )

    assert trainer.global_steps == 10
    assert ("save", 10) in events
    assert ("validate", 10) in events
    assert ("validation-state", 10) in events
    assert ("progress", 10, 123) in events
    assert ("close", 10) in events
    assert any(event[0] == "log" and event[1] == 10 for event in events)


def test_natural_exhaustion_refreshes_existing_final_checkpoint_runtime_state():
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.global_steps = 11
    trainer.config = OmegaConf.create({"trainer": {"test_freq": 10}})
    events = []
    trainer._save_checkpoint = lambda: events.append(("full-save", trainer.global_steps))
    trainer._refresh_checkpoint_runtime_state = lambda: events.append(
        ("runtime-state", trainer.global_steps)
    )
    trainer._validate = lambda: events.append(("validate", trainer.global_steps)) or {"val": 1.0}
    trainer._filter_logged_metrics = lambda metrics: metrics

    class Logger:
        def log(self, *, data, step):
            events.append(("log", step, dict(data)))

    class ProgressBar:
        def close(self):
            events.append(("close", trainer.global_steps))

    trainer._finalize_naturally_exhausted_training(
        last_completed_step=10,
        last_validation_step=10,
        last_saved_step=10,
        logger=Logger(),
        progress_bar=ProgressBar(),
        refresh_update_progress=lambda step, consumed: None,
        consumed_prompt_examples=123,
    )

    assert trainer.global_steps == 10
    assert ("runtime-state", 10) in events
    assert not any(event[0] == "full-save" for event in events)
    assert not any(event[0] == "validate" for event in events)
    assert trainer._visharness_training_completed is True
    assert trainer._visharness_completion_reason == "data_exhausted"


def test_refresh_checkpoint_runtime_state_updates_data_and_resume_state(tmp_path):
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.global_steps = 10
    checkpoint_dir = tmp_path / "checkpoints" / "global_step_10"
    checkpoint_dir.mkdir(parents=True)
    trainer.config = OmegaConf.create(
        {"trainer": {"default_local_dir": str(tmp_path / "checkpoints")}}
    )

    class Loader:
        def state_dict(self):
            return {"samples_yielded": 17, "generator": torch.tensor([1, 2, 3])}

    trainer.train_dataloader = Loader()
    captured = {}

    def save_resume_state(*, dataloader_state=None):
        captured["state"] = dataloader_state

    trainer._save_visharness_resume_state = save_resume_state
    trainer._refresh_checkpoint_runtime_state()

    saved_data = torch.load(
        checkpoint_dir / "data.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert saved_data["samples_yielded"] == 17
    assert saved_data["generator"].tolist() == [1, 2, 3]
    assert captured["state"]["samples_yielded"] == 17


def test_resume_completion_is_explicit_and_legacy_inference_is_safe():
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.global_steps = 12
    trainer.total_training_steps = 100
    trainer.config = OmegaConf.create({"trainer": {"total_epochs": 1}})

    assert trainer._resume_training_completion(
        {"training_completed": True, "completion_reason": "data_exhausted"}
    ) == (True, "data_exhausted")
    assert trainer._resume_training_completion(
        {"training_completed": False, "train_epoch": 1}
    ) == (False, None)
    assert trainer._resume_training_completion(
        {"version": 3, "train_epoch": 1, "raw_prompt_buffers": {}}
    ) == (True, "legacy_data_exhausted")

    buffered = DataProto.from_dict(
        tensors={"prompt_id": torch.tensor([1])},
        non_tensors={"uid": np.array(["prompt-1"], dtype=object)},
    )
    assert trainer._resume_training_completion(
        {"version": 3, "train_epoch": 1, "raw_prompt_buffers": {"source": buffered}}
    ) == (False, None)


def test_global_step_zero_checkpoint_loads_visharness_completion_state(tmp_path):
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.global_steps = 0
    trainer._visharness_checkpoint_loaded = True
    checkpoint_dir = tmp_path / "checkpoints" / "global_step_0"
    checkpoint_dir.mkdir(parents=True)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(tmp_path / "checkpoints"),
                "resume_mode": "auto",
            }
        }
    )
    torch.save(
        {
            "version": 4,
            "global_steps": 0,
            "training_completed": True,
            "completion_reason": "data_exhausted",
        },
        checkpoint_dir / "visharness_resume_state.pt",
    )

    state = trainer._load_visharness_resume_state()

    assert state["training_completed"] is True
    assert trainer._resume_training_completion(state) == (True, "data_exhausted")


def test_legacy_validation_artifact_prevents_duplicate_resume_validation(tmp_path):
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.global_steps = 150
    checkpoint_dir = tmp_path / "checkpoints" / "global_step_150"
    checkpoint_dir.mkdir(parents=True)
    validation_dir = tmp_path / "validation" / "global_step_150"
    validation_dir.mkdir(parents=True)
    trainer.val_dataset = [None, None, None]
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(tmp_path / "checkpoints"),
                "resume_mode": "auto",
            },
            "actor_rollout_ref": {"rollout": {"val_kwargs": {"n": 1}}},
            "visharness": {
                "validation": {"results_dir": str(tmp_path / "validation")}
            },
        }
    )
    (validation_dir / "metrics.json").write_text(
        json.dumps(
            {
                "global_step": 150,
                "sample_count": 3,
                "metadata": {"checkpoint_path": str(checkpoint_dir)},
            }
        ),
        encoding="utf-8",
    )

    assert trainer._restored_last_validation_step({"version": 3}) == 150
    assert trainer._restored_last_validation_step(
        {"version": 4, "last_validation_step": None}
    ) is None


def test_raw_prompt_buffer_resume_is_backward_compatible_and_corrects_progress():
    assert VisHarnessTrainer._raw_prompt_buffer_from_resume_state(
        {"version": 1, "global_steps": 10, "train_epoch": 0}
    ) is None

    restored = DataProto.from_dict(
        tensors={"prompt_id": torch.arange(6, dtype=torch.long)},
        non_tensors={"uid": np.array([f"prompt-{idx}" for idx in range(6)], dtype=object)},
    )
    assert VisHarnessTrainer._effective_restored_prompt_examples(408, restored) == 402
    assert VisHarnessTrainer._effective_restored_prompt_examples(None, restored) is None
    assert VisHarnessTrainer._effective_restored_prompt_examples(4, restored) == 0

    with pytest.raises(ValueError, match="raw_prompt_buffer must be a DataProto"):
        VisHarnessTrainer._raw_prompt_buffer_from_resume_state({"raw_prompt_buffer": ["prompt-1"]})


def test_trajectory_reward_group_filter_uses_only_std_threshold():
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.config = OmegaConf.create(
        {
            "visharness": {
                "filter_groups": {
                    "min_reward_std": 0.2,
                }
            }
        }
    )
    batch = make_filter_batch(
        uids=["large-range-low-std"] * 4 + ["enough-std"] * 4 + ["single"],
        # The first group's range is 0.30 but its sample std is only 0.15.
        # The second group's sample std is 0.25.
        metric_vals=[0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.5, 1.0],
    )

    kept_prompt_uids, metrics = trainer._select_filter_group_uids(batch, "trajectory_reward")

    assert kept_prompt_uids == ["enough-std"]
    assert batch.non_tensor_batch["trajectory_advantage_suppressed"].tolist() == [False] * 9
    assert metrics == {}


def test_soft_reward_group_filter_zero_threshold_preserves_dapo_std_behavior():
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.config = OmegaConf.create(
        {
            "visharness": {
                "filter_groups": {
                    "min_reward_std": 0.0,
                }
            }
        }
    )
    batch = make_filter_batch(
        uids=["different"] * 4 + ["same"] * 4,
        metric_vals=[0.10, 0.10, 0.10, 0.11, 0.20, 0.20, 0.20, 0.20],
    )

    kept_prompt_uids, _ = trainer._select_filter_group_uids(batch, "trajectory_reward")

    assert kept_prompt_uids == ["different"]


def test_soft_reward_group_filter_uses_sample_standard_deviation():
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.config = OmegaConf.create(
        {
            "visharness": {
                "filter_groups": {
                    "min_reward_std": 0.05,
                }
            }
        }
    )
    batch = make_filter_batch(
        uids=["boundary"] * 4,
        metric_vals=[0.0, 0.0, 0.0, 0.1],
    )

    kept_prompt_uids, metrics = trainer._select_filter_group_uids(batch, "trajectory_reward")

    assert kept_prompt_uids == ["boundary"]
    assert metrics == {}


@pytest.mark.parametrize("min_reward_std", [-0.01, float("nan"), float("inf")])
def test_trajectory_reward_group_filter_rejects_invalid_std_threshold(min_reward_std):
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.config = OmegaConf.create(
        {"visharness": {"filter_groups": {"min_reward_std": min_reward_std}}}
    )

    with pytest.raises(ValueError, match="min_reward_std must be non-negative"):
        trainer._select_filter_group_uids(
            make_filter_batch(["prompt-a"] * 4, [0.0, 0.1, 0.2, 0.3]),
            "trajectory_reward",
        )


def test_ordinary_group_filter_rejects_non_trajectory_reward_metric():
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"filter_groups": {"metric": "seq_reward"}},
            "visharness": {},
        }
    )

    with pytest.raises(ValueError, match="metric=trajectory_reward"):
        trainer._apply_filter_groups(
            new_batch=make_filter_batch(["prompt-a"] * 4, [0.0, 0.1, 0.2, 0.3]),
            batch=None,
            metrics={},
            num_prompt_in_batch=0,
        )


def test_trajectory_reward_filter_drops_zero_variance_group_despite_local_errors():
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.config = OmegaConf.create(
        {
            "visharness": {
                "filter_groups": {
                    "min_reward_std": 0.05,
                },
            }
        }
    )
    turn_records = np.empty(4, dtype=object)
    turn_records[0] = [
        {
            "response_ids": [1],
            "output_format_success": False,
            "tool_args_success": False,
        }
    ]
    turn_records[1] = [
        {
            "response_ids": [1],
            "output_format_success": True,
            "tool_args_success": False,
        }
    ]
    turn_records[2] = [
        {
            "response_ids": [1],
            "output_format_success": True,
            "tool_args_success": True,
            "turn_truncated_by_length": True,
        }
    ]
    turn_records[3] = [
        {
            "response_ids": [1],
            "output_format_success": True,
            "tool_args_success": True,
        }
    ]
    batch = DataProto.from_dict(
        tensors={"dummy": torch.zeros(4, dtype=torch.long)},
        non_tensors={
            "uid": np.array(["local"] * 4, dtype=object),
            "task_reward": np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float64),
            "trajectory_reward": np.array([4.8, 4.8, 4.8, 4.8], dtype=np.float64),
            "turn_records": turn_records,
        },
    )

    kept_prompt_uids, metrics = trainer._select_filter_group_uids(batch, "trajectory_reward")

    assert kept_prompt_uids == []
    assert batch.non_tensor_batch["trajectory_advantage_suppressed"].tolist() == [False] * 4
    assert metrics == {}


def test_trajectory_reward_filter_includes_enabled_step_cost_in_std():
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.config = OmegaConf.create(
        {
            "visharness": {
                "filter_groups": {"min_reward_std": 0.05},
                "per_turn": {"max_response_length": 8},
                "trajectory_reward": {
                    "enable_step_cost": True,
                    "step_cost": 0.2,
                },
            }
        }
    )
    turn_records = np.empty(4, dtype=object)
    for trajectory_index in range(4):
        turn_records[trajectory_index] = [
            {
                "response_ids": [turn_index + 1],
                "output_format_success": True,
                "tool_args_success": True,
            }
            for turn_index in range(trajectory_index + 1)
        ]

    def make_batch():
        return DataProto.from_dict(
            tensors={"dummy": torch.zeros(4, dtype=torch.long)},
            non_tensors={
                "uid": np.array(["same-task-reward"] * 4, dtype=object),
                "task_reward": np.array([5.0] * 4, dtype=np.float64),
                "turn_records": turn_records,
            },
        )

    step_cost_batch = make_batch()
    trainer._prepare_filter_group_metric(step_cost_batch, "trajectory_reward")
    assert step_cost_batch.non_tensor_batch["trajectory_reward"].tolist() == pytest.approx(
        [4.8, 4.6, 4.4, 4.2]
    )
    kept_prompt_uids, _ = trainer._select_filter_group_uids(
        step_cost_batch,
        "trajectory_reward",
    )
    assert kept_prompt_uids == ["same-task-reward"]

    trainer.config.visharness.trajectory_reward.enable_step_cost = False
    no_step_cost_batch = make_batch()
    trainer._prepare_filter_group_metric(no_step_cost_batch, "trajectory_reward")
    assert no_step_cost_batch.non_tensor_batch["trajectory_reward"].tolist() == pytest.approx(
        [5.0] * 4
    )
    kept_prompt_uids, _ = trainer._select_filter_group_uids(
        no_step_cost_batch,
        "trajectory_reward",
    )
    assert kept_prompt_uids == []


def test_per_turn_batch_honors_legacy_trajectory_advantage_suppression_flag():
    turn_records = np.empty(4, dtype=object)
    for index in range(4):
        turn_records[index] = [
            {
                "turn_index": 1,
                "prompt_ids": [10],
                "response_ids": [20 + index],
                "output_format_success": index != 0,
                "tool_args_success": index != 0,
                "turn_response_length": 1,
                "turn_truncated_by_length": False,
                "multi_modal_data": {"images": []},
                "mm_processor_kwargs": {},
            }
        ]

    batch = DataProto.from_dict(
        tensors={"dummy": torch.zeros(4, dtype=torch.long)},
        non_tensors={
            "uid": np.array(["prompt-a"] * 4, dtype=object),
            "trajectory_uid": np.array([f"trajectory-{index}" for index in range(4)], dtype=object),
            "task_reward": np.array([5.00, 5.01, 5.02, 5.03], dtype=np.float64),
            "trajectory_reward": np.array([4.80, 4.81, 4.82, 4.83], dtype=np.float64),
            "trajectory_advantage_suppressed": np.array([True] * 4, dtype=bool),
            "turn_records": turn_records,
        },
    )

    result, metrics = build_per_turn_grpo_batch(
        batch,
        tokenizer=FakeTokenizer(),
        processor=FakeProcessor(),
        max_prompt_length=4,
        max_response_length=4,
        trajectory_step_cost=0.2,
        output_format_error_cost=0.5,
        tool_args_error_cost=0.5,
        truncation_error_cost=0.75,
        overlong_buffer_length=0,
        overlong_cost_coef=0.0,
        norm_adv_by_std=True,
    )

    assert result.non_tensor_batch["trajectory_reward"].tolist() == pytest.approx([4.80, 4.81, 4.82, 4.83])
    assert result.non_tensor_batch["trajectory_advantage_suppressed"].tolist() == [True] * 4
    assert result.non_tensor_batch["trajectory_advantage"].tolist() == pytest.approx([0.0] * 4)
    assert result.non_tensor_batch["turn_advantage"].tolist() == pytest.approx([-0.5, 0.0, 0.0, 0.0])
    assert result.batch["advantages"][:, 0].tolist() == pytest.approx([-0.5, 0.0, 0.0, 0.0])
    assert metrics["visharness/per_turn/trajectory_advantage_suppressed_group_count"] == 1.0
    assert metrics["visharness/per_turn/trajectory_advantage_suppressed_trajectory_count"] == 4.0
    assert metrics["visharness/per_turn/trajectory_advantage_suppressed_trajectory_ratio"] == 1.0


def make_trajectory_metrics_batch():
    turn_records = np.empty(3, dtype=object)
    turn_records[0] = [
        {
            "response_ids": [1, 2, 3],
            "turn_response_length": 3,
            "output_format_success": True,
            "tool_args_success": True,
        },
        {
            "response_ids": list(range(9)),
            "turn_response_length": 9,
            "output_format_success": False,
            "tool_args_success": False,
        },
    ]
    turn_records[1] = [
        {
            "response_ids": [1, 2],
            "turn_response_length": 2,
            "output_format_success": True,
            "tool_args_success": True,
        }
    ]
    turn_records[2] = [
        {
            "response_ids": list(range(10)),
            "turn_response_length": 10,
            "turn_truncated_by_length": True,
            "output_format_success": False,
            "tool_args_success": False,
        }
    ]
    return DataProto.from_dict(
        tensors={"dummy": torch.zeros(3, dtype=torch.long)},
        non_tensors={
            "uid": np.array(["prompt-a", "prompt-a", "prompt-b"], dtype=object),
            "turn_records": turn_records,
            "task_reward": np.array([5.0, 5.0, 1.0], dtype=np.float64),
        },
    )


def test_trajectory_metrics_separate_trajectory_reward_from_local_advantage_costs():
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.config = OmegaConf.create(
        {
            "data": {"max_response_length": 10},
            "visharness": {
                "per_turn": {"max_response_length": 10},
                "trajectory_reward": {"step_cost": 0.2},
                "per_turn_advantage": {
                    "output_format_error_cost": 0.5,
                    "tool_args_error_cost": 0.5,
                    "truncation_error_cost": 0.75,
                    "overlong_buffer_length": 2,
                    "overlong_cost_coef": 0.5,
                },
            },
        }
    )
    batch = make_trajectory_metrics_batch()
    accumulator = trainer._new_trajectory_metrics_accumulator()

    trainer._accumulate_trajectory_metrics(accumulator, batch)
    metrics = trainer._summarize_trajectory_metrics(accumulator, "rollout/eligible")

    assert batch.non_tensor_batch["trajectory_reward"].tolist() == pytest.approx([4.6, 4.8, 0.8])
    assert metrics["rollout/eligible/trajectory_reward_mean"] == pytest.approx(
        (4.6 + 4.8 + 0.8) / 3
    )
    assert metrics["rollout/eligible/group_trajectory_reward_std_mean"] == pytest.approx(
        np.mean([np.std([4.6, 4.8], ddof=1), 0.0])
    )
    assert metrics["rollout/eligible/interaction_turns_mean"] == pytest.approx(4 / 3)
    assert metrics["rollout/eligible/assistant_response_length_mean"] == pytest.approx(6.0)
    assert metrics["rollout/eligible/trajectory_step_cost_mean"] == pytest.approx(0.8 / 3)
    assert metrics["rollout/eligible/local_advantage_cost_mean"] == pytest.approx(1.25 / 3)
    assert metrics["rollout/eligible/output_format_error_rate"] == pytest.approx(0.25)
    assert metrics["rollout/eligible/raw_output_format_failure_rate"] == pytest.approx(0.5)
    assert metrics["rollout/eligible/tool_args_error_rate"] == 0.0
    assert metrics["rollout/eligible/truncation_error_rate"] == pytest.approx(0.25)
    assert metrics["rollout/eligible/hard_error_turn_rate"] == pytest.approx(0.5)
    assert (
        metrics["rollout/eligible/output_format_error_rate"]
        + metrics["rollout/eligible/tool_args_error_rate"]
        + metrics["rollout/eligible/truncation_error_rate"]
    ) == pytest.approx(metrics["rollout/eligible/hard_error_turn_rate"])
    assert "rollout/eligible/trajectory_reward_min" not in metrics
    assert "rollout/eligible/format_error_cost_mean" not in metrics


def test_trainer_disables_trajectory_step_cost_without_discarding_configured_coefficient():
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.config = OmegaConf.create(
        {
            "visharness": {
                "trajectory_reward": {
                    "enable_step_cost": False,
                    "step_cost": 0.2,
                }
            }
        }
    )

    settings = trainer._get_trajectory_reward_settings()

    assert settings["enable_step_cost"] is False
    assert settings["configured_step_cost"] == pytest.approx(0.2)
    assert settings["step_cost"] == 0.0


def test_disabled_trajectory_step_cost_keeps_task_reward_for_filtering_and_grpo():
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.config = OmegaConf.create(
        {
            "data": {"max_response_length": 10},
            "visharness": {
                "per_turn": {"max_response_length": 10},
                "trajectory_reward": {
                    "enable_step_cost": False,
                    "step_cost": 0.2,
                },
                "per_turn_advantage": {
                    "output_format_error_cost": 0.5,
                    "tool_args_error_cost": 0.5,
                    "truncation_error_cost": 0.75,
                    "overlong_buffer_length": 2,
                    "overlong_cost_coef": 0.5,
                },
            },
        }
    )
    batch = make_trajectory_metrics_batch()
    accumulator = trainer._new_trajectory_metrics_accumulator()

    trainer._accumulate_trajectory_metrics(accumulator, batch)

    assert batch.non_tensor_batch["trajectory_reward"].tolist() == pytest.approx([5.0, 5.0, 1.0])
    assert accumulator["trajectory_step_costs"] == pytest.approx([0.0, 0.0, 0.0])


def test_concise_logged_metrics_keep_only_main_dashboard():
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.config = OmegaConf.create(
        {
            "visharness": {
                "logged_metrics": {
                    "mode": "concise",
                    "drop_prefixes": ["critic/"],
                    "include_exact": ["actor/loss"],
                    "include_prefixes": ["rollout/eligible/", "train/selected/", "filter/"],
                }
            }
        }
    )

    filtered = trainer._filter_logged_metrics(
        {
            "actor/loss": 1.0,
            "actor/perf/max_memory_allocated_gb": 2.0,
            "critic/score/mean": 3.0,
            "timing_s/reward": 4.0,
            "rollout/eligible/trajectory_reward_mean": 5.0,
            "train/selected/interaction_turns_mean": 6.0,
            "filter/reward_group_keep_rate": 0.5,
        }
    )

    assert filtered == {
        "actor/loss": 1.0,
        "rollout/eligible/trajectory_reward_mean": 5.0,
        "train/selected/interaction_turns_mean": 6.0,
        "filter/reward_group_keep_rate": 0.5,
    }


def test_validation_trajectory_reward_mean_uses_fallback_for_missing_values():
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)

    metrics = trainer._val_metrics_update(
        data_sources=np.array(["source"] * 4, dtype=object),
        sample_uids=["a", "b", "c", "d"],
        reward_extra_infos_dict={
            "trajectory_reward": [1.0, None, float("nan"), 4.0],
            "reward": [10.0, 2.0, 3.0, 40.0],
        },
        sample_turns=[],
    )

    assert metrics["val-core/trajectory_reward_mean"] == pytest.approx(2.5)
    assert metrics["val-core/samples"] == 4


def test_actor_update_sample_metrics_ignore_dp_padding_rows():
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    metrics = {}
    batch = DataProto.from_dict(
        tensors={
            "prompts": torch.zeros((3, 4), dtype=torch.long),
            "response_mask": torch.tensor([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.long),
            "attention_mask": torch.tensor(
                [
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0],
                ],
                dtype=torch.long,
            ),
        },
        non_tensors={
            "data_source": np.asarray(
                ["visharness/rec8k", "visharness/gres", "visharness/reasonseg"],
                dtype=object,
            ),
            "turn_advantage": np.asarray([1.0, -2.0, 100.0], dtype=np.float64),
        },
    )

    trainer._add_actor_update_sample_metrics(metrics, batch)

    assert metrics["train/update/per_turn_samples"] == 2
    assert metrics["train/update/assistant_response_length_mean"] == pytest.approx(1.5)
    assert metrics["train/update/prompt_length_mean"] == pytest.approx(2.5)
    assert metrics["train/update/by_data_source/rec8k/turn_samples"] == 1
    assert metrics["train/update/by_data_source/gres/turn_sample_share"] == pytest.approx(0.5)
    assert metrics["train/update/by_data_source/gres/turn_advantage_abs_mean"] == pytest.approx(2.0)
    assert "train/update/by_data_source/reasonseg/turn_samples" not in metrics


def test_reference_kl_metrics_are_source_specific_and_ignore_padding_rows():
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "kl_loss_type": "low_var_kl+",
                    "loss_agg_mode": "seq-mean-token-mean",
                }
            }
        }
    )
    batch = DataProto.from_dict(
        tensors={
            "old_log_probs": torch.tensor(
                [[0.0, 0.0], [-1.0, 7.0], [20.0, 20.0]],
                dtype=torch.float32,
            ),
            "ref_log_prob": torch.zeros((3, 2), dtype=torch.float32),
            "response_mask": torch.tensor(
                [[1, 1], [1, 0], [0, 0]],
                dtype=torch.long,
            ),
        },
        non_tensors={
            "data_source": np.asarray(
                ["visharness/rec8k", "visharness/gres", "visharness/reasonseg"],
                dtype=object,
            )
        },
    )

    metrics = trainer._reference_kl_metrics_by_source(batch)

    assert metrics[
        "train/update/by_data_source/rec8k/pre_update_reference_kl_mean"
    ] == pytest.approx(0.0)
    assert metrics[
        "train/update/by_data_source/gres/pre_update_reference_kl_mean"
    ] == pytest.approx(
        np.e - 2.0
    )
    assert (
        "train/update/by_data_source/reasonseg/pre_update_reference_kl_mean"
        not in metrics
    )


def test_archive_current_hf_checkpoint_copies_inference_model(tmp_path):
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.global_steps = 10
    trainer.config = OmegaConf.create(
        {
            "trainer": {"default_local_dir": str(tmp_path / "checkpoints")},
            "visharness": {
                "archive_checkpoints": {
                    "enable": True,
                    "dir": str(tmp_path / "checkpoints" / "archived"),
                }
            },
        }
    )
    source_dir = tmp_path / "checkpoints" / "global_step_10" / "actor" / "huggingface"
    source_dir.mkdir(parents=True)
    (source_dir / "config.json").write_text("{}", encoding="utf-8")
    (source_dir / "model.safetensors").write_text("weights-v1", encoding="utf-8")

    trainer._archive_current_hf_checkpoint()

    target_dir = tmp_path / "checkpoints" / "archived" / "global_step_10"
    assert (target_dir / "config.json").read_text(encoding="utf-8") == "{}"
    assert (target_dir / "model.safetensors").read_text(encoding="utf-8") == "weights-v1"

    (source_dir / "model.safetensors").write_text("weights-v2", encoding="utf-8")
    trainer._archive_current_hf_checkpoint()

    assert (target_dir / "model.safetensors").read_text(encoding="utf-8") == "weights-v2"


def test_trainer_update_actor_recomputes_per_turn_old_log_probs():
    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.tokenizer = FakeTokenizer()
    trainer.processor = FakeProcessor()
    trainer.use_reference_policy = False
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": 2},
            "visharness": {
                "per_turn": {
                    "num_mini_batches": 2,
                    "trajectory_equal_weight": True,
                },
            },
            "actor_rollout_ref": {
                "actor": {
                    "calculate_entropy": False,
                    "entropy_coeff": 0.0,
                    "ppo_mini_batch_size": 1,
                    "ppo_epochs": 1,
                    "data_loader_seed": 42,
                    "shuffle": False,
                },
                "rollout": {
                    "n": 2,
                    "temperature": 1.0,
                    "multi_turn": {"enable": True},
                },
            }
        }
    )
    trainer._get_dp_size = lambda worker_group, role: 2

    observed = {}

    def compute_old_log_prob(per_turn_batch):
        observed["num_per_turn_samples"] = len(per_turn_batch)
        shape = per_turn_batch.batch["responses"].shape
        return (
            DataProto.from_dict(
                tensors={
                    "old_log_probs": torch.zeros(shape),
                    "entropys": torch.zeros(shape),
                }
            ),
            0.25,
        )

    trainer._compute_old_log_prob = compute_old_log_prob

    class FakeActorRolloutWorkerGroup:
        def update_actor(self, data):
            observed["mini_batch_size"] = tu.get(data, "mini_batch_size")
            observed["loss_mask_rows"] = data["loss_mask"].shape[0]
            observed["effective_global_batch_sizes"] = data[EFFECTIVE_GLOBAL_BATCH_SIZE_KEY].tolist()
            observed["trajectory_loss_weights"] = data[TRAJECTORY_LOSS_WEIGHT_KEY][:, 0].tolist()
            return tu.get_tensordict(
                tensor_dict={},
                non_tensor_dict={"metrics": {"mfu": [0.5], "pg_loss": [1.0]}},
            )

    trainer.actor_rollout_wg = FakeActorRolloutWorkerGroup()

    per_turn_batch, _ = build_per_turn_grpo_batch(
        _make_trajectory_weighting_batch(),
        tokenizer=FakeTokenizer(),
        processor=FakeProcessor(),
        max_prompt_length=6,
        max_response_length=4,
        trajectory_step_cost=0.0,
        output_format_error_cost=0.5,
        tool_args_error_cost=0.5,
        truncation_error_cost=0.75,
        overlong_buffer_length=0,
        overlong_cost_coef=0.0,
        norm_adv_by_std=False,
        trajectory_equal_weight=True,
        local_cost_mode="per_event_floor",
    )
    output = trainer._update_actor(per_turn_batch)

    assert observed["num_per_turn_samples"] == 8
    assert observed["mini_batch_size"] == 4
    assert observed["loss_mask_rows"] == 8
    assert observed["effective_global_batch_sizes"] == [3, 3, 2, 2] * 2
    assert all(weight > 0 for weight in observed["trajectory_loss_weights"][:5])
    assert observed["trajectory_loss_weights"][5:] == [0.0, 0.0, 0.0]
    assert output.meta_info["metrics"]["actor/pg_loss"] == [1.0]
    assert output.meta_info["metrics"]["visharness/per_turn_samples"] == [5]
    assert output.meta_info["metrics"]["visharness/per_turn_padding"] == [3]
    assert output.meta_info["metrics"]["visharness/per_turn_num_mini_batches"] == [2]
    assert output.meta_info["metrics"]["visharness/per_turn_optimizer_steps"] == [2]
    assert output.meta_info["metrics"]["train/update/optimizer_minibatches"] == [2]
    assert output.meta_info["metrics"]["train/update/global_mini_batch_size"] == [4]
    assert output.meta_info["metrics"]["train/update/actual_optimizer_steps"] == [2]
    assert output.meta_info["metrics"]["train/update/per_turn_padding_ratio"] == [0.375]
    assert "train/update/trajectory_minibatch_weight_correction_min" in output.meta_info["metrics"]
    assert "train/update/trajectory_minibatch_weight_correction_max" in output.meta_info["metrics"]
