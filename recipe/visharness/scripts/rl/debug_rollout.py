"""Run one real VisHarness trajectory through verl's asynchronous AgentLoop."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import ray
import requests
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.reward_loop import RewardLoopManager
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.main_ppo import create_rl_dataset
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.utils import hf_processor, hf_tokenizer, omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.device import get_device_name
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.rollout.llm_server import LLMServerManager

from visharness.agent_loop.dynamic_validation_manager import (
    VisHarnessAgentLoopManager,
)

DEFAULT_MODEL_PATH = Path("/vepfs-dev/metro/hantao/nwp_bench/fyw/code/fyw/code/VisionAgent/output/Qwen3-VL-8B-Thinking/sft/v0-20260518-001941/checkpoint-103")
DEFAULT_DATA_PATH = PROJECT_ROOT / "training_data/GRPO/verl_visharness/val.parquet"
DEFAULT_CONTROLLER_LOCATION = (
    PROJECT_ROOT / "tool_server/tool_workers/online_workers/controller_addr/controller_addr.json"
)
REQUIRED_TOOLS = {
    "PhraseToBoxMask",
    "PhraseToPoint",
    "PointToBoxMask",
    "SplitImageIntoPatches",
    "SuperResolution",
    "MergeBoxMask",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path(os.getenv("VISHARNESS_MODEL_PATH", DEFAULT_MODEL_PATH)))
    parser.add_argument("--data-path", type=Path, default=Path(os.getenv("VISHARNESS_VAL_DATA", DEFAULT_DATA_PATH)))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--per-turn-max-response-length", type=int, default=4096)
    parser.add_argument("--data-max-prompt-length", type=int, default=None)
    parser.add_argument("--per-turn-max-prompt-length", type=int, default=None)
    parser.add_argument("--rollout-max-prompt-length", type=int, default=None)
    parser.add_argument("--rollout-placeholder-prompt-length", type=int, default=8)
    parser.add_argument("--rollout-placeholder-response-length", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-trajectory-length", type=int, default=None, help="Deprecated alias for older debug commands")
    parser.add_argument("--max-agent-turns", type=int, default=6)
    parser.add_argument("--prompt-length", type=int, default=None)
    parser.add_argument("--response-length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--controller-location", type=Path, default=DEFAULT_CONTROLLER_LOCATION)
    args = parser.parse_args()
    old_length = args.max_trajectory_length
    if args.per_turn_max_prompt_length is None:
        args.per_turn_max_prompt_length = old_length or 24576
    if args.data_max_prompt_length is None:
        args.data_max_prompt_length = args.per_turn_max_prompt_length
    if args.rollout_max_prompt_length is None:
        args.rollout_max_prompt_length = args.prompt_length or args.data_max_prompt_length
    if args.rollout_placeholder_response_length is None:
        args.rollout_placeholder_response_length = old_length or 65536
    if args.max_model_len is None:
        args.max_model_len = old_length or 65536
    return args


def check_inputs(args: argparse.Namespace) -> str:
    if not args.model_path.joinpath("config.json").is_file():
        raise FileNotFoundError(f"Model config does not exist: {args.model_path / 'config.json'}")
    if not args.data_path.is_file():
        raise FileNotFoundError(f"Rollout parquet does not exist: {args.data_path}")
    if args.sample_index < 0:
        raise ValueError(f"sample-index must be non-negative, got {args.sample_index}")
    if args.tensor_parallel_size > args.num_gpus or args.num_gpus % args.tensor_parallel_size != 0:
        raise ValueError(
            f"num-gpus ({args.num_gpus}) must be divisible by tensor-parallel-size "
            f"({args.tensor_parallel_size})"
        )

    if args.controller_location.is_file():
        controller_addr = json.loads(args.controller_location.read_text())["controller_addr"]
    else:
        controller_addr = str(args.controller_location)

    with requests.Session() as session:
        session.trust_env = False
        models_response = session.post(f"{controller_addr}/list_models", timeout=5)
        models_response.raise_for_status()
        workers_response = session.post(f"{controller_addr}/list_workers", timeout=5)
        workers_response.raise_for_status()

    models = set(models_response.json()["models"])
    missing = REQUIRED_TOOLS - models
    if missing:
        raise RuntimeError(f"Controller {controller_addr} is missing required tools: {sorted(missing)}")

    workers = workers_response.json()["workers"]
    print(f"[ready] controller: {controller_addr}")
    print(f"[ready] tools: {sorted(models)}")
    print(f"[ready] workers ({len(workers)}): {workers}")
    return controller_addr


def set_config_value(config: DictConfig, key: str, value: Any) -> None:
    OmegaConf.update(config, key, value, merge=False, force_add=True)


def build_config(args: argparse.Namespace) -> DictConfig:
    config_dir = PROJECT_ROOT / "recipe/visharness/configs"
    os.environ["VISHARNESS_MODEL_PATH"] = str(args.model_path)
    os.environ["VISHARNESS_TRAIN_DATA"] = str(args.data_path)
    os.environ["VISHARNESS_VAL_DATA"] = str(args.data_path)
    os.environ["VISHARNESS_PER_TURN_MAX_PROMPT_LENGTH"] = str(args.per_turn_max_prompt_length)
    os.environ["VISHARNESS_PER_TURN_MAX_RESPONSE_LENGTH"] = str(args.per_turn_max_response_length)
    os.environ["VISHARNESS_ROLLOUT_MAX_PROMPT_LENGTH"] = str(args.rollout_max_prompt_length)
    os.environ["VISHARNESS_MAX_AGENT_TURNS"] = str(args.max_agent_turns)

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="visharness_grpo")

    data_prompt_length = args.prompt_length or args.data_max_prompt_length
    response_length = args.response_length or args.rollout_placeholder_response_length
    debug_values = {
        "data.train_files": str(args.data_path),
        "data.val_files": str(args.data_path),
        "data.max_prompt_length": data_prompt_length,
        "data.max_response_length": response_length,
        "data.filter_overlong_prompts_workers": 1,
        "data.shuffle": False,
        "actor_rollout_ref.model.path": str(args.model_path),
        "actor_rollout_ref.model.enable_gradient_checkpointing": False,
        "actor_rollout_ref.actor.use_dynamic_bsz": True,
        "actor_rollout_ref.actor.fsdp_config.param_offload": True,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.mode": "async",
        "actor_rollout_ref.rollout.n": 1,
        "actor_rollout_ref.rollout.prompt_length": args.rollout_placeholder_prompt_length,
        "actor_rollout_ref.rollout.response_length": response_length,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": args.tensor_parallel_size,
        "actor_rollout_ref.rollout.data_parallel_size": args.num_gpus // args.tensor_parallel_size,
        "actor_rollout_ref.rollout.gpu_memory_utilization": args.gpu_memory_utilization,
        "actor_rollout_ref.rollout.enforce_eager": True,
        "actor_rollout_ref.rollout.max_model_len": args.max_model_len,
        "actor_rollout_ref.rollout.max_num_batched_tokens": args.max_model_len,
        "actor_rollout_ref.rollout.max_num_seqs": 8,
        "actor_rollout_ref.rollout.calculate_log_probs": True,
        "actor_rollout_ref.rollout.agent.num_workers": 1,
        "actor_rollout_ref.rollout.multi_turn.max_assistant_turns": args.max_agent_turns,
        "actor_rollout_ref.rollout.multi_turn.max_user_turns": args.max_agent_turns,
        "actor_rollout_ref.rollout.temperature": args.temperature,
        "actor_rollout_ref.rollout.top_p": args.top_p,
        "visharness.per_turn.max_prompt_length": args.per_turn_max_prompt_length,
        "visharness.per_turn.max_response_length": args.per_turn_max_response_length,
        "trainer.n_gpus_per_node": args.num_gpus,
        "trainer.nnodes": 1,
        "reward.num_workers": 1,
    }
    for key, value in debug_values.items():
        set_config_value(config, key, value)
    OmegaConf.resolve(config)
    return config


def init_agent_loop_manager(config: DictConfig) -> VisHarnessAgentLoopManager:
    role_worker_mapping = {Role.ActorRollout: ray.remote(ActorRolloutRefWorker)}
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec={"global_pool": [config.trainer.n_gpus_per_node]},
        mapping={Role.ActorRollout: "global_pool"},
    )
    resource_pool_manager.create_resource_pool()

    resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
    actor_rollout_cls = RayClassWithInitArgs(
        cls=role_worker_mapping[Role.ActorRollout],
        config=config.actor_rollout_ref,
        role="actor_rollout",
    )
    worker_dict_cls = create_colocated_worker_cls(class_dict={"actor_rollout": actor_rollout_cls})
    worker_group = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=worker_dict_cls,
        device_name=get_device_name(),
    )
    actor_rollout_wg = worker_group.spawn(prefix_set={"actor_rollout"})["actor_rollout"]
    actor_rollout_wg.init_model()

    reward_loop_manager = RewardLoopManager(config=config, rm_resource_pool=None)
    llm_server_manager = LLMServerManager.create(config=config, worker_group=actor_rollout_wg)
    agent_loop_manager = VisHarnessAgentLoopManager.create(
        config=config,
        llm_client=llm_server_manager.get_client(),
        reward_loop_worker_handles=reward_loop_manager.reward_loop_workers,
    )
    checkpoint_manager = CheckpointEngineManager(
        config=omega_conf_to_dataclass(config.actor_rollout_ref.rollout.checkpoint_engine),
        trainer=actor_rollout_wg,
        replicas=llm_server_manager.get_replicas(),
    )
    checkpoint_manager.sleep_replicas()
    checkpoint_manager.update_weights()
    return agent_loop_manager


def build_one_sample_batch(config: DictConfig, sample_index: int) -> tuple[DataProto, Any]:
    tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path, trust_remote_code=True)
    processor = hf_processor(config.actor_rollout_ref.model.path, trust_remote_code=True, use_fast=True)
    dataset = create_rl_dataset(
        config.data.val_files,
        config.data,
        tokenizer,
        processor,
        is_train=False,
        max_samples=sample_index + 1,
    )
    if sample_index >= len(dataset):
        raise IndexError(f"sample-index {sample_index} is outside the filtered dataset of size {len(dataset)}")

    batch_dict = collate_fn([dataset[sample_index]])
    batch = DataProto.from_single_dict(batch_dict)
    batch.meta_info.update({"validate": False, "global_steps": 0})
    return batch, tokenizer


def json_summary(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_summary(item) for key, item in value.items() if key != "multi_modal_data"}
    if isinstance(value, (list, tuple)):
        return [json_summary(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return f"<bytes length={len(value)}>"
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "size") and hasattr(value, "mode"):
        return f"<PIL.Image mode={value.mode} size={value.size}>"
    return value


def decode_ids(tokenizer: Any, token_ids: Any) -> str:
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def clear_console() -> None:
    """Clear the terminal while keeping subsequent rollout output together."""
    print("\033[2J\033[H", end="", flush=True)


def print_result(result: DataProto, input_batch: DataProto, tokenizer: Any) -> None:
    raw_prompt = input_batch.non_tensor_batch["raw_prompt"][0]

    prompts = result.batch["prompts"][0]
    responses = result.batch["responses"][0]
    prompt_attention = result.batch["attention_mask"][0, : prompts.shape[-1]].bool()
    response_mask = result.batch["response_mask"][0].bool()
    response_attention = result.batch["attention_mask"][0, -responses.shape[-1] :].bool()

    initial_prompt_ids = prompts[prompt_attention].tolist()
    full_response_ids = responses[response_attention].tolist()
    full_trajectory = decode_ids(tokenizer, initial_prompt_ids + full_response_ids)
    trainable_response = decode_ids(tokenizer, responses[response_mask])

    clear_console()
    # print("\n================ FULL TRAJECTORY TOKEN STREAM ================\n")
    # print(full_trajectory)
    # print("\n================ ORIGINAL RAW MESSAGES ================\n")
    # print(json.dumps(json_summary(raw_prompt), ensure_ascii=False, indent=2))
    # print("\n================ TRAINABLE TOKENS ================\n")
    # print(trainable_response)

    turn_records = result.non_tensor_batch["turn_records"][0]
    print("\n================ EXACT PER-TURN MODEL INPUTS ================\n")
    for record in turn_records:
        print(
            f"\n---------------- TURN {record['turn_index']} "
            f"(visible_images={record['visible_image_names']}) ----------------\n"
        )
        prompt_ids = list(record["prompt_ids"])
        response_ids = list(record["response_ids"])
        print("[FULL TURN: prompt context + current assistant response]\n")
        print(decode_ids(tokenizer, prompt_ids + response_ids))
        print(
            f"\n[LOSS TOKENS: current assistant response only; "
            f"prompt_tokens={len(prompt_ids)} excluded, response_tokens={len(response_ids)} included]\n"
        )
        print(decode_ids(tokenizer, response_ids))

    print("\n================ TURN RECORDS ================\n")
    for record in turn_records:
        summary = {
            "turn_index": record["turn_index"],
            "action_type": record["action_type"],
            "tool_name": record["tool_name"],
            "stop_reason": record.get("stop_reason"),
            "turn_max_tokens": record.get("turn_max_tokens"),
            "turn_response_length": record.get("turn_response_length"),
            "generation_context_length_before_turn": record.get("generation_context_length_before_turn"),
            "generation_context_length_after_turn": record.get("generation_context_length_after_turn"),
            "trajectory_prompt_length": record.get("trajectory_prompt_length"),
            "trajectory_token_length_before_turn": record.get("trajectory_token_length_before_turn"),
            "trajectory_token_length_after_turn": record.get("trajectory_token_length_after_turn"),
            "observation_token_length": record.get("observation_token_length"),
            "raw_observation_token_length": record.get("raw_observation_token_length"),
            "observation_truncated_by_trajectory_limit": record.get("observation_truncated_by_trajectory_limit"),
            "output_format_success": record["output_format_success"],
            "tool_args_success": record["tool_args_success"],
            "action_error": record.get("action_error"),
            "tool_args_error": record.get("tool_args_error"),
            "visible_image_names": record["visible_image_names"],
            "completion": record["completion"],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    metrics = {
        "num_turns": int(result.non_tensor_batch["__num_turns__"][0]),
        "trajectory_finished": result.non_tensor_batch["trajectory_finished"][0],
        "tool_call_count": result.non_tensor_batch["tool_call_count"][0],
        "per_turn_max_response_length": result.non_tensor_batch.get("per_turn_max_response_length", [None])[0],
        "rollout_placeholder_response_length": result.non_tensor_batch.get(
            "rollout_placeholder_response_length", [None]
        )[0],
        "max_agent_turns": result.non_tensor_batch.get("max_agent_turns", [None])[0],
        "max_agent_turns_reached": result.non_tensor_batch.get("max_agent_turns_reached", [None])[0],
        "trajectory_response_length": result.non_tensor_batch.get("trajectory_response_length", [None])[0],
        "trajectory_token_length": result.non_tensor_batch.get("trajectory_token_length", [None])[0],
        "final_results": result.non_tensor_batch["final_results"][0],
    }
    for key in (
        "score",
        "task_reward",
        "step_penalty",
        "output_format_penalty",
        "tool_args_penalty",
        "output_format_success_rate",
        "tool_args_success_rate",
        "task_metric",
        "iou",
        "gt_count",
        "pred_count",
    ):
        if key in result.non_tensor_batch:
            metrics[key] = result.non_tensor_batch[key][0]
    print("\n================ RESULT SUMMARY ================\n")
    print(json.dumps(json_summary(metrics), ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    check_inputs(args)
    config = build_config(args)

    np.random.seed(args.seed)
    ray.shutdown()
    ray.init(
        runtime_env={
            "env_vars": {
                "PYTHONPATH": f"{PROJECT_ROOT}:{PROJECT_ROOT / 'verl'}",
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
                "VISHARNESS_PER_TURN_MAX_PROMPT_LENGTH": str(args.per_turn_max_prompt_length),
                "VISHARNESS_PER_TURN_MAX_RESPONSE_LENGTH": str(args.per_turn_max_response_length),
                "VISHARNESS_ROLLOUT_MAX_PROMPT_LENGTH": str(args.rollout_max_prompt_length),
                "VISHARNESS_MAX_AGENT_TURNS": str(args.max_agent_turns),
                "VISHARNESS_DEBUG_AGENT_LOOP": os.getenv("VISHARNESS_DEBUG_AGENT_LOOP", "0"),
                "VISHARNESS_DEBUG_AGENT_LOOP_HOST": os.getenv("VISHARNESS_DEBUG_AGENT_LOOP_HOST", "0.0.0.0"),
                "VISHARNESS_DEBUG_AGENT_LOOP_PORT": os.getenv("VISHARNESS_DEBUG_AGENT_LOOP_PORT", "5682"),
                "VISHARNESS_DEBUG_AGENT_LOOP_WAIT": os.getenv("VISHARNESS_DEBUG_AGENT_LOOP_WAIT", "1"),
                "VISHARNESS_DEBUG_AGENT_LOOP_ONCE": os.getenv("VISHARNESS_DEBUG_AGENT_LOOP_ONCE", "1"),
            },
        },
        ignore_reinit_error=True,
    )

    try:
        batch, tokenizer = build_one_sample_batch(config, args.sample_index)
        print(
            f"[run] sample={args.sample_index}, source={batch.non_tensor_batch['data_source'][0]}, "
            f"item={batch.non_tensor_batch['extra_info'][0].get('item_id')}"
        )
        manager = init_agent_loop_manager(config)
        result = manager.generate_sequences(prompts=batch)
        print_result(result, batch, tokenizer)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
