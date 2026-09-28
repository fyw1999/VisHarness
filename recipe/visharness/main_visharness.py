"""VisHarness training entry point using trajectory-level GRPO and per-turn actor updates."""

import os
import socket
from copy import deepcopy

import hydra
import ray

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler, run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device


def _maybe_wait_for_debugger() -> None:
    """Optionally pause the Ray TaskRunner process for VSCode attach debugging."""
    if os.getenv("VISHARNESS_DEBUG_TASKRUNNER", "0") != "1":
        return

    host = os.getenv("VISHARNESS_DEBUG_HOST", "0.0.0.0")
    port = int(os.getenv("VISHARNESS_DEBUG_PORT", "5681"))
    wait = os.getenv("VISHARNESS_DEBUG_WAIT", "1") != "0"

    try:
        import debugpy
    except ImportError:
        print("[debug] debugpy is not installed in the Ray TaskRunner process; continuing without debugger.")
        return

    try:
        debugpy.listen((host, port))
        print(f"[debug] VisHarness TaskRunner waiting for VSCode attach on {host}:{port}, pid={os.getpid()}")
    except RuntimeError as exc:
        print(f"[debug] debugpy.listen({host}:{port}) failed: {exc}")

    if wait:
        debugpy.wait_for_client()
    debugpy.breakpoint()


def _propagate_runtime_env_to_ray(config) -> None:
    """Make optional runtime settings visible inside Ray worker processes."""
    from omegaconf import OmegaConf

    if config.get("visharness", {}).get("debug_taskrunner", False):
        os.environ["VISHARNESS_DEBUG_TASKRUNNER"] = "1"
    if config.get("visharness", {}).get("debug_agent_loop", False):
        os.environ["VISHARNESS_DEBUG_AGENT_LOOP"] = "1"
    if config.get("visharness", {}).get("debug_reward", False):
        os.environ["VISHARNESS_DEBUG_REWARD"] = "1"

    runtime_env_keys = (
        "VISHARNESS_DEBUG_TASKRUNNER",
        "VISHARNESS_DEBUG_HOST",
        "VISHARNESS_DEBUG_PORT",
        "VISHARNESS_DEBUG_WAIT",
        "VISHARNESS_DEBUG_AGENT_LOOP",
        "VISHARNESS_DEBUG_AGENT_LOOP_HOST",
        "VISHARNESS_DEBUG_AGENT_LOOP_PORT",
        "VISHARNESS_DEBUG_AGENT_LOOP_WAIT",
        "VISHARNESS_DEBUG_AGENT_LOOP_ONCE",
        "VISHARNESS_DEBUG_REWARD",
        "VISHARNESS_DEBUG_REWARD_HOST",
        "VISHARNESS_DEBUG_REWARD_PORT",
        "VISHARNESS_DEBUG_REWARD_WAIT",
        "VISHARNESS_DEBUG_REWARD_ONCE",
        "VISHARNESS_PER_TURN_MAX_PROMPT_LENGTH",
        "VISHARNESS_PER_TURN_MAX_RESPONSE_LENGTH",
        "VISHARNESS_ROLLOUT_MAX_PROMPT_LENGTH",
        "VISHARNESS_VALIDATION_MAX_PROMPT_LENGTH",
        "VISHARNESS_MAX_AGENT_TURNS",
        "VISHARNESS_TRAJECTORY_STEP_COST_ENABLE",
        "VISHARNESS_TRAJECTORY_STEP_COST",
    )
    for key in runtime_env_keys:
        if key in os.environ:
            OmegaConf.update(
                config,
                f"ray_kwargs.ray_init.runtime_env.env_vars.{key}",
                os.environ[key],
                merge=False,
                force_add=True,
            )


class VisHarnessTaskRunner(TaskRunner):
    """Build standard verl workers, then run the custom VisHarness trainer."""

    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        if config.get("visharness", {}).get("debug_taskrunner", False):
            os.environ["VISHARNESS_DEBUG_TASKRUNNER"] = "1"
        if config.get("visharness", {}).get("debug_agent_loop", False):
            os.environ["VISHARNESS_DEBUG_AGENT_LOOP"] = "1"
        if config.get("visharness", {}).get("debug_reward", False):
            os.environ["VISHARNESS_DEBUG_REWARD"] = "1"
        _maybe_wait_for_debugger()
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_teacher_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_data_config = deepcopy(config.data)
        val_max_prompt_length = val_data_config.get("val_max_prompt_length", None)
        if val_max_prompt_length is None:
            val_max_prompt_length = int(config.actor_rollout_ref.rollout.max_model_len) - 1
        val_data_config.max_prompt_length = int(val_max_prompt_length)
        val_dataset = create_rl_dataset(
            val_data_config.val_files,
            val_data_config,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        task_sampling_config = config.get("visharness", {}).get("task_sampling", {})
        use_task_balanced_sampling = bool(task_sampling_config.get("enable", False))
        if use_task_balanced_sampling:
            from visharness.data.task_balanced_sampler import create_temperature_balanced_sampler

            configured_gen_batch_size = config.data.get("gen_batch_size", None)
            sampler_batch_size = int(
                configured_gen_batch_size
                if configured_gen_batch_size is not None
                else config.data.train_batch_size
            )
            train_sampler = create_temperature_balanced_sampler(
                train_dataset,
                alpha=float(task_sampling_config.get("alpha", 0.5)),
                batch_size=sampler_batch_size,
                seed=config.data.get("seed"),
                epoch_size_mode="all_sources_once",
            )
            plan = train_sampler.plan
            distribution = ", ".join(
                f"{source}={plan.source_counts[source]}->{plan.probabilities[source]:.2%}"
                for source in plan.sources
            )
            print(
                "Task-balanced prompt sampling enabled: "
                f"alpha={plan.alpha}, coverage-based source cycling, "
                f"initial_step_estimate={plan.num_samples // plan.batch_size}, {distribution}"
            )
        else:
            train_sampler = create_rl_sampler(config.data, train_dataset)

        from visharness.rl_trainer import VisHarnessTrainer

        trainer = VisHarnessTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="configs", config_name="visharness_grpo", version_base=None)
def main(config):
    auto_set_device(config)
    _propagate_runtime_env_to_ray(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(VisHarnessTaskRunner))


if __name__ == "__main__":
    main()
