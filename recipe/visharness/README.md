# VisHarness Recipe

This directory contains reproducible VisHarness experiment configurations and
launch scripts. Importable implementation code belongs in `visharness/`.

## RL training

`scripts/rl/train_visharness.sh` computes reward and GRPO advantage for each
complete trajectory, then expands accepted trajectories into the exact
per-turn prompts/images used during rollout before updating the actor.

`scripts/rl/debug_rollout.py` is an optional single-trajectory integration
debugger for inspecting exact model inputs, responses, loss tokens, turn
records, and rewards.

Run commands from the repository root after activating the AgentRL runtime
environment.

## Per-turn training

```bash
bash recipe/visharness/scripts/rl/train_visharness.sh
```

The script contains runnable defaults for the current VisionAgent workspace,
including the model, parquet files, AgentRL Python, eight-GPU training
configuration, caches, and checkpoint directory. Override a script variable
when needed, for example:

```bash
NUM_GPUS=2 ROLLOUT_TP_SIZE=2 TRAIN_BATCH_SIZE=1 GROUP_SIZE=2 \
DRY_RUN=1 bash recipe/visharness/scripts/rl/train_visharness.sh
```

Arguments after the script are forwarded as Hydra overrides and take
precedence over the script defaults.

The custom trainer currently requires GRPO outcome advantages,
`algorithm.use_kl_in_reward=false`, and `actor.use_prefix_grouper=false`. It
supports `actor.loss_agg_mode=token-mean` and `seq-mean-token-mean`. Sequence
mean uses the real sequence count after excluding fully masked per-turn DP
padding and requires `actor.shuffle=false`. Actor KL loss is computed from
reference log probabilities recomputed on the per-turn samples.

Ordinary GRPO keeps a fixed number of actor optimizer mini-batches per outer
rollout update. By default this is derived as
`TRAIN_BATCH_SIZE / PPO_MINI_BATCH_SIZE` (currently `8 / 4 = 2`). The expanded
turn batch is loss-masked and padded only to `actor_dp_size *
PER_TURN_NUM_MINI_BATCHES`; the global turn mini-batch size therefore adapts to
the number of accepted turns without changing the number of optimizer steps.
Dynamic token micro-batches only accumulate gradients inside those optimizer
steps. Override `PER_TURN_NUM_MINI_BATCHES` explicitly when an experiment needs
a different update count. The dashboard reports the executed plan as
`train/update/optimizer_minibatches`, `train/update/global_mini_batch_size`,
`train/update/actual_optimizer_steps`, and
`train/update/per_turn_padding_ratio`; the padding ratio is the number of
fully loss-masked turn rows divided by the final padded actor batch size.
