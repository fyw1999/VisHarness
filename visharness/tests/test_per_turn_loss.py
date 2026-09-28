import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer
from verl.utils import tensordict_utils as tu
from visharness.rl_trainer import per_turn_loss
from visharness.rl_trainer.per_turn_loss import (
    EFFECTIVE_GLOBAL_BATCH_SIZE_KEY,
    TRAJECTORY_LOSS_WEIGHT_KEY,
    annotate_effective_global_batch_size,
    rescale_trajectory_loss_weights_for_minibatches,
    visharness_ppo_loss,
)
from visharness.rl_trainer.visharness_trainer import VisHarnessTrainer


def _make_actor_batch(valid_rows: list[bool]) -> TensorDict:
    response_mask = torch.tensor([[1, 0] if is_valid else [0, 0] for is_valid in valid_rows])
    return TensorDict({"response_mask": response_mask}, batch_size=[len(valid_rows)])


def test_effective_batch_size_matches_verl_dp_dispatch_and_local_mini_batches():
    batch = _make_actor_batch([True] * 10 + [False, False])

    annotate_effective_global_batch_size(batch, global_mini_batch_size=4, dp_size=4)

    # Verl dispatches contiguous chunks of three rows to each DP rank. Each
    # rank then consumes one local row per optimizer mini-batch. The final two
    # dummy rows therefore affect optimizer mini-batches 1 and 2.
    assert batch[EFFECTIVE_GLOBAL_BATCH_SIZE_KEY].tolist() == [4, 3, 3] * 4

    local_batch_size = 3
    for mini_batch_index, expected_size in enumerate([4, 3, 3]):
        group_values = []
        for dp_rank in range(4):
            row_index = dp_rank * local_batch_size + mini_batch_index
            group_values.append(int(batch[EFFECTIVE_GLOBAL_BATCH_SIZE_KEY][row_index]))
        assert group_values == [expected_size] * 4


def test_effective_batch_size_is_unchanged_without_dummy_rows():
    batch = _make_actor_batch([True] * 8)

    annotate_effective_global_batch_size(batch, global_mini_batch_size=4, dp_size=2)

    assert batch[EFFECTIVE_GLOBAL_BATCH_SIZE_KEY].tolist() == [4] * 8


def test_nonuniform_trajectory_weights_are_exact_with_padding_and_multiple_mini_batches():
    batch = _make_actor_batch([True] * 7 + [False])
    base_weights = torch.tensor([1.75, 1.75, 0.7, 0.7, 0.7, 0.7, 0.7, 0.0]).unsqueeze(-1)
    batch[TRAJECTORY_LOSS_WEIGHT_KEY] = batch["response_mask"].float() * base_weights
    annotate_effective_global_batch_size(batch, global_mini_batch_size=4, dp_size=2)

    corrections = rescale_trajectory_loss_weights_for_minibatches(
        batch,
        num_mini_batches=2,
        num_real_sequences=7,
    )

    assert corrections.tolist() == pytest.approx(
        [8.0 / 7.0, 8.0 / 7.0, 6.0 / 7.0, 6.0 / 7.0] * 2
    )
    assert batch[TRAJECTORY_LOSS_WEIGHT_KEY][:, 0].tolist() == pytest.approx(
        [2.0, 2.0, 0.6, 0.6, 0.8, 0.8, 0.6, 0.0]
    )

    # DP ranks receive contiguous four-row shards. The two global optimizer
    # mini-batches are therefore rows [0, 1, 4, 5] and [2, 3, 6, 7].
    turn_losses = torch.arange(1.0, 9.0)
    optimizer_groups = ([0, 1, 4, 5], [2, 3, 6, 7])
    mini_batch_objectives = []
    for indices in optimizer_groups:
        row_indices = torch.tensor(indices)
        valid = batch["response_mask"][row_indices].any(dim=-1)
        weighted_losses = batch[TRAJECTORY_LOSS_WEIGHT_KEY][row_indices, 0] * turn_losses[row_indices]
        mini_batch_objectives.append(weighted_losses[valid].sum() / valid.sum())

    actual_epoch_objective = torch.stack(mini_batch_objectives).mean()
    expected_trajectory_objective = 0.5 * (turn_losses[:2].mean() + turn_losses[2:7].mean())
    torch.testing.assert_close(actual_epoch_objective, expected_trajectory_objective)


def test_trajectory_weight_rescaling_rejects_stale_real_sequence_metadata():
    batch = _make_actor_batch([True, True, False, False])
    batch[TRAJECTORY_LOSS_WEIGHT_KEY] = batch["response_mask"].float()
    annotate_effective_global_batch_size(batch, global_mini_batch_size=4, dp_size=1)

    with pytest.raises(ValueError, match="metadata disagrees"):
        rescale_trajectory_loss_weights_for_minibatches(
            batch,
            num_mini_batches=1,
            num_real_sequences=3,
        )


def test_effective_batch_size_rejects_invalid_dp_or_mini_batch_shapes():
    batch = _make_actor_batch([True] * 6)

    with pytest.raises(ValueError, match="divisible by DP size"):
        annotate_effective_global_batch_size(batch, global_mini_batch_size=4, dp_size=4)

    with pytest.raises(ValueError, match="Global mini-batch size"):
        annotate_effective_global_batch_size(batch, global_mini_batch_size=3, dp_size=2)


def test_visharness_ppo_loss_overrides_only_the_sequence_denominator(monkeypatch):
    data = TensorDict(
        {EFFECTIVE_GLOBAL_BATCH_SIZE_KEY: torch.tensor([3, 3], dtype=torch.long)},
        batch_size=[2],
    )
    tu.assign_non_tensor_data(data, "global_batch_size", 4)
    observed = {}

    def fake_ppo_loss(*, config, model_output, data, dp_group):
        observed["config"] = config
        observed["model_output"] = model_output
        observed["global_batch_size"] = tu.get_non_tensor_data(data, "global_batch_size", None)
        observed["dp_group"] = dp_group
        return torch.tensor(1.25), {"metric": 2.0}

    monkeypatch.setattr(per_turn_loss, "ppo_loss", fake_ppo_loss)

    result = visharness_ppo_loss(config="config", model_output="output", data=data, dp_group="dp")

    assert result[0].item() == pytest.approx(1.25)
    assert result[1] == {"metric": 2.0}
    assert observed == {
        "config": "config",
        "model_output": "output",
        "global_batch_size": 3,
        "dp_group": "dp",
    }


def test_visharness_ppo_loss_weights_policy_advantages_and_restores_them(monkeypatch):
    original_advantages = torch.tensor([[2.0, 0.0], [3.0, 3.0]])
    data = TensorDict(
        {
            EFFECTIVE_GLOBAL_BATCH_SIZE_KEY: torch.tensor([2, 2], dtype=torch.long),
            TRAJECTORY_LOSS_WEIGHT_KEY: torch.tensor([[1.25, 0.0], [0.75, 0.75]]),
            "advantages": original_advantages.clone(),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor_data(data, "global_batch_size", 2)
    observed = {}

    def fake_ppo_loss(**kwargs):
        observed["advantages"] = kwargs["data"]["advantages"].clone()
        return torch.tensor(0.5), {}

    monkeypatch.setattr(per_turn_loss, "ppo_loss", fake_ppo_loss)

    visharness_ppo_loss(config=None, model_output=None, data=data)

    torch.testing.assert_close(
        observed["advantages"],
        torch.tensor([[2.5, 0.0], [2.25, 2.25]]),
    )
    torch.testing.assert_close(data["advantages"], original_advantages)


def test_visharness_ppo_loss_restores_advantages_when_upstream_loss_fails(monkeypatch):
    original_advantages = torch.tensor([[1.0, 1.0]])
    data = TensorDict(
        {
            EFFECTIVE_GLOBAL_BATCH_SIZE_KEY: torch.tensor([1], dtype=torch.long),
            TRAJECTORY_LOSS_WEIGHT_KEY: torch.tensor([[2.0, 2.0]]),
            "advantages": original_advantages.clone(),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor_data(data, "global_batch_size", 1)

    def failing_ppo_loss(**kwargs):
        raise RuntimeError("upstream failure")

    monkeypatch.setattr(per_turn_loss, "ppo_loss", failing_ppo_loss)

    with pytest.raises(RuntimeError, match="upstream failure"):
        visharness_ppo_loss(config=None, model_output=None, data=data)

    torch.testing.assert_close(data["advantages"], original_advantages)


def test_visharness_ppo_loss_rejects_mixed_optimizer_mini_batch_annotations(monkeypatch):
    data = TensorDict(
        {EFFECTIVE_GLOBAL_BATCH_SIZE_KEY: torch.tensor([3, 4], dtype=torch.long)},
        batch_size=[2],
    )
    tu.assign_non_tensor_data(data, "global_batch_size", 4)
    monkeypatch.setattr(per_turn_loss, "ppo_loss", lambda **kwargs: None)

    with pytest.raises(ValueError, match="inconsistent effective global batch sizes"):
        visharness_ppo_loss(config=None, model_output=None, data=data)


def test_trainer_installs_visharness_loss_after_worker_initialization(monkeypatch):
    import verl.utils.config as config_utils

    installed = {}

    class FakeActorWorkerGroup:
        def set_loss_fn(self, loss_fn):
            installed["loss_fn"] = loss_fn

    def fake_parent_init_workers(self):
        self.actor_rollout_wg = FakeActorWorkerGroup()

    monkeypatch.setattr(RayDAPOTrainer, "init_workers", fake_parent_init_workers)
    monkeypatch.setattr(config_utils, "omega_conf_to_dataclass", lambda config: "actor-config")

    trainer = VisHarnessTrainer.__new__(VisHarnessTrainer)
    trainer.config = OmegaConf.create({"actor_rollout_ref": {"actor": {"loss_agg_mode": "token-mean"}}})
    trainer.init_workers()

    installed_loss = installed["loss_fn"]
    assert installed_loss.func is visharness_ppo_loss
    assert installed_loss.keywords == {"config": "actor-config"}
