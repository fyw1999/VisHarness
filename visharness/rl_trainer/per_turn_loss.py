"""Loss normalization helpers for data-parallel per-turn actor updates."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.utils.losses import ppo_loss


EFFECTIVE_GLOBAL_BATCH_SIZE_KEY = "visharness_effective_global_batch_size"
TRAJECTORY_LOSS_WEIGHT_KEY = "visharness_trajectory_loss_weights"


def annotate_effective_global_batch_size(
    batch: TensorDict,
    *,
    global_mini_batch_size: int,
    dp_size: int,
) -> TensorDict:
    """Annotate rows with the number of real sequences in their optimizer mini-batch.

    The unified Verl worker first splits the global batch into contiguous DP
    shards. Every DP rank then iterates over equally sized local mini-batches.
    A global optimizer mini-batch is therefore the union of the same local
    mini-batch position across all DP shards. Fully masked rows are VisHarness
    dispatch padding and must not contribute to sequence-mean denominators.

    Actor mini-batch shuffling must be disabled so these precomputed groups are
    preserved until the loss function runs.
    """

    if "response_mask" not in batch:
        raise KeyError("Cannot annotate sequence normalization without response_mask")
    if global_mini_batch_size <= 0 or dp_size <= 0:
        raise ValueError(
            "global_mini_batch_size and dp_size must be positive, got "
            f"{global_mini_batch_size=} {dp_size=}"
        )

    response_mask = batch["response_mask"]
    if response_mask.ndim < 2:
        raise ValueError(f"response_mask must have at least two dimensions, got {response_mask.shape}")

    batch_size = int(response_mask.shape[0])
    if batch_size <= 0:
        raise ValueError("Cannot annotate an empty actor batch")
    if batch_size % dp_size != 0:
        raise ValueError(f"Actor batch size {batch_size} must be divisible by DP size {dp_size}")
    if global_mini_batch_size % dp_size != 0:
        raise ValueError(
            f"Global mini-batch size {global_mini_batch_size} must be divisible by DP size {dp_size}"
        )
    if batch_size % global_mini_batch_size != 0:
        raise ValueError(
            f"Actor batch size {batch_size} must be divisible by global mini-batch size {global_mini_batch_size}"
        )

    local_batch_size = batch_size // dp_size
    local_mini_batch_size = global_mini_batch_size // dp_size
    num_mini_batches = batch_size // global_mini_batch_size
    valid_rows = response_mask.reshape(batch_size, -1).to(torch.bool).any(dim=-1)
    effective_sizes = torch.empty(batch_size, dtype=torch.long, device=response_mask.device)

    for mini_batch_index in range(num_mini_batches):
        row_ranges: list[tuple[int, int]] = []
        valid_sequence_count = 0
        for dp_rank in range(dp_size):
            start = dp_rank * local_batch_size + mini_batch_index * local_mini_batch_size
            end = start + local_mini_batch_size
            row_ranges.append((start, end))
            valid_sequence_count += int(valid_rows[start:end].sum().item())

        if valid_sequence_count <= 0:
            raise ValueError(f"Actor optimizer mini-batch {mini_batch_index} contains no trainable sequences")
        for start, end in row_ranges:
            effective_sizes[start:end] = valid_sequence_count

    batch[EFFECTIVE_GLOBAL_BATCH_SIZE_KEY] = effective_sizes
    return batch


def rescale_trajectory_loss_weights_for_minibatches(
    batch: TensorDict,
    *,
    num_mini_batches: int,
    num_real_sequences: int,
) -> torch.Tensor:
    """Correct trajectory weights when padded optimizer mini-batches differ in size.

    Let optimizer mini-batch ``j`` contain ``m_j`` real sequences, let ``M`` be
    the number of real sequences in the complete actor batch, and let ``J`` be
    the number of optimizer mini-batches.  Verl independently sequence-averages
    every optimizer mini-batch by ``m_j``.  Multiplying its row weights by

    ``J * m_j / M``

    makes the average of the ``J`` mini-batch objectives equal the intended
    full-batch trajectory-weighted objective.  The factor is one when there is
    one optimizer mini-batch, or when every mini-batch has the same real size.

    Returns the per-row correction factors for logging and tests.
    """

    if TRAJECTORY_LOSS_WEIGHT_KEY not in batch:
        raise KeyError(f"Actor batch is missing {TRAJECTORY_LOSS_WEIGHT_KEY!r}")
    if EFFECTIVE_GLOBAL_BATCH_SIZE_KEY not in batch:
        raise KeyError(f"Actor batch is missing {EFFECTIVE_GLOBAL_BATCH_SIZE_KEY!r}")
    if num_mini_batches <= 0 or num_real_sequences <= 0:
        raise ValueError(
            "num_mini_batches and num_real_sequences must be positive, got "
            f"{num_mini_batches=} {num_real_sequences=}"
        )

    weights = batch[TRAJECTORY_LOSS_WEIGHT_KEY]
    effective_sizes = batch[EFFECTIVE_GLOBAL_BATCH_SIZE_KEY].reshape(-1)
    if weights.shape[0] != effective_sizes.numel():
        raise ValueError(
            "Trajectory weights and effective batch-size annotations disagree: "
            f"{weights.shape[0]} rows versus {effective_sizes.numel()} annotations"
        )
    valid_rows = batch["response_mask"].reshape(weights.shape[0], -1).to(torch.bool).any(dim=-1)
    actual_real_sequences = int(valid_rows.sum().item())
    if actual_real_sequences != num_real_sequences:
        raise ValueError(
            "Trajectory-weight metadata disagrees with the actor response mask: "
            f"{num_real_sequences=} {actual_real_sequences=}"
        )
    corrections = effective_sizes.to(dtype=weights.dtype, device=weights.device)
    corrections = corrections * (float(num_mini_batches) / float(num_real_sequences))
    correction_shape = (weights.shape[0],) + (1,) * (weights.ndim - 1)
    batch[TRAJECTORY_LOSS_WEIGHT_KEY] = weights * corrections.reshape(correction_shape)
    return corrections


def visharness_ppo_loss(config, model_output, data: TensorDict, dp_group=None):
    """Run Verl's PPO loss with VisHarness sequence and trajectory weights.

    ``TRAJECTORY_LOSS_WEIGHT_KEY`` contains the non-negative per-token policy
    row weight: trajectory weight ``q`` times the optimizer-mini-batch
    correction ``r`` (and zero on padding). It scales only the policy-gradient
    term through ``advantages``; entropy and reference-KL regularizers keep
    their original aggregation. The original advantages are restored after
    the call so repeated PPO epochs cannot apply the weight more than once.
    """

    if EFFECTIVE_GLOBAL_BATCH_SIZE_KEY not in data:
        raise KeyError(
            f"Actor loss input is missing {EFFECTIVE_GLOBAL_BATCH_SIZE_KEY!r}; "
            "the per-turn batch was not annotated before DP dispatch"
        )

    effective_sizes = data[EFFECTIVE_GLOBAL_BATCH_SIZE_KEY].reshape(-1)
    if effective_sizes.numel() == 0:
        raise ValueError("Actor loss received an empty effective batch-size annotation")

    effective_global_batch_size = int(effective_sizes[0].item())
    if not bool(torch.all(effective_sizes == effective_global_batch_size).item()):
        raise ValueError(
            "One actor micro-batch contains inconsistent effective global batch sizes; "
            "actor mini-batch shuffling or DP grouping changed after annotation"
        )
    if effective_global_batch_size <= 0:
        raise ValueError("Effective global batch size must be positive")

    padded_global_batch_size = tu.get_non_tensor_data(data=data, key="global_batch_size", default=None)
    if padded_global_batch_size is not None and effective_global_batch_size > int(padded_global_batch_size):
        raise ValueError(
            "Effective global batch size cannot exceed the dispatched mini-batch size: "
            f"{effective_global_batch_size} > {padded_global_batch_size}"
        )

    tu.assign_non_tensor_data(
        tensor_dict=data,
        key="global_batch_size",
        val=effective_global_batch_size,
    )

    trajectory_loss_weights = data.get(TRAJECTORY_LOSS_WEIGHT_KEY, None)
    if trajectory_loss_weights is None:
        return ppo_loss(config=config, model_output=model_output, data=data, dp_group=dp_group)
    if "advantages" not in data:
        raise KeyError("Actor loss input is missing 'advantages' for trajectory weighting")
    if trajectory_loss_weights.shape != data["advantages"].shape:
        raise ValueError(
            "Trajectory loss weights must match advantages, got "
            f"{trajectory_loss_weights.shape} and {data['advantages'].shape}"
        )
    if not bool(torch.isfinite(trajectory_loss_weights).all().item()):
        raise ValueError("Trajectory loss weights must be finite")
    if bool((trajectory_loss_weights < 0).any().item()):
        raise ValueError("Trajectory loss weights must be non-negative")

    original_advantages = data["advantages"]
    data["advantages"] = original_advantages * trajectory_loss_weights.to(
        dtype=original_advantages.dtype,
        device=original_advantages.device,
    )
    try:
        return ppo_loss(config=config, model_output=model_output, data=data, dp_group=dp_group)
    finally:
        data["advantages"] = original_advantages
