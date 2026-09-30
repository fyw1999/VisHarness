#!/usr/bin/env bash

set -euo pipefail

# Run from anywhere. Every value below can be overridden with an environment
# variable, while additional Hydra overrides can still be passed as arguments.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
CACHE_ROOT="${CACHE_ROOT:-/vepfs-dev/metro/hantao/nwp_bench/fyw/cache}"
RAY_TMP_ROOT="${RAY_TMP_ROOT:-${CACHE_ROOT}/ray-tmp}"
RAY_TMP_LINK="${RAY_TMP_LINK:-/tmp/fyw-ray}"
RAY_TMP_DIR="${RAY_TMP_DIR:-${RAY_TMP_LINK}/visharness}"

MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/sft/Qwen3-VL-8B-Thinking/v0-20260928-211731/checkpoint-105}"
MODEL_FAMILY="${MODEL_FAMILY:-auto}"
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/training_data/GRPO/verl_visharness/train.parquet}"
VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/training_data/GRPO/verl_visharness_official_val/val.parquet}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-VisHarness-grpo}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints/VisHarness/${EXPERIMENT_NAME}}"
SWANLAB_RESUME_ENABLED="${SWANLAB_RESUME_ENABLED:-false}"
SWANLAB_RESUME_MODE="${SWANLAB_RESUME_MODE:-must}"
SWANLAB_RUN_ID="${SWANLAB_RUN_ID:-}"

NUM_GPUS="${NUM_GPUS:-8}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-2}"
SEED="${SEED:-42}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
ROLLOUT_DP_SIZE="${ROLLOUT_DP_SIZE:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
GROUP_SIZE="${GROUP_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
PER_TURN_NUM_MINI_BATCHES="${PER_TURN_NUM_MINI_BATCHES:-auto}"
PER_TURN_MAX_PROMPT_LENGTH="${PER_TURN_MAX_PROMPT_LENGTH:-23800}"
DATA_MAX_PROMPT_LENGTH="${DATA_MAX_PROMPT_LENGTH:-${PER_TURN_MAX_PROMPT_LENGTH}}"
ROLLOUT_MAX_PROMPT_LENGTH="${ROLLOUT_MAX_PROMPT_LENGTH:-49152}"
PER_TURN_MAX_RESPONSE_LENGTH="${PER_TURN_MAX_RESPONSE_LENGTH:-2048}"
TRAJECTORY_EQUAL_WEIGHT="${TRAJECTORY_EQUAL_WEIGHT:-false}"
MAX_AGENT_TURNS="${MAX_AGENT_TURNS:-12}"
ROLLOUT_PLACEHOLDER_PROMPT_LENGTH="${ROLLOUT_PLACEHOLDER_PROMPT_LENGTH:-8}"
ROLLOUT_PLACEHOLDER_RESPONSE_LENGTH="${ROLLOUT_PLACEHOLDER_RESPONSE_LENGTH:-8}"
PER_TURN_MAX_TOKEN_LENGTH=$((PER_TURN_MAX_PROMPT_LENGTH + PER_TURN_MAX_RESPONSE_LENGTH))
ROLLOUT_MAX_TOKEN_LENGTH=$((ROLLOUT_MAX_PROMPT_LENGTH + PER_TURN_MAX_RESPONSE_LENGTH))
DEFAULT_MAX_MODEL_LEN=32768
if (( PER_TURN_MAX_TOKEN_LENGTH > DEFAULT_MAX_MODEL_LEN )); then
    DEFAULT_MAX_MODEL_LEN="${PER_TURN_MAX_TOKEN_LENGTH}"
fi
if (( ROLLOUT_MAX_TOKEN_LENGTH > DEFAULT_MAX_MODEL_LEN )); then
    DEFAULT_MAX_MODEL_LEN="${ROLLOUT_MAX_TOKEN_LENGTH}"
fi
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${DEFAULT_MAX_MODEL_LEN}}"
VAL_MAX_PROMPT_LENGTH="${VAL_MAX_PROMPT_LENGTH:-$((MAX_MODEL_LEN - 1))}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65535}"
# Keep the current 14k dynamic-batch budget when sequence parallelism can
# accommodate the longest turn. If SP is reduced (notably SP=1), raise the
# default automatically so one full training sequence still fits.
MIN_TRAIN_TOKEN_LEN_PER_GPU=$(((PER_TURN_MAX_TOKEN_LENGTH + ULYSSES_SEQUENCE_PARALLEL_SIZE - 1) / ULYSSES_SEQUENCE_PARALLEL_SIZE))
DEFAULT_PPO_MAX_TOKEN_LEN_PER_GPU=14000
if (( DEFAULT_PPO_MAX_TOKEN_LEN_PER_GPU < MIN_TRAIN_TOKEN_LEN_PER_GPU )); then
    DEFAULT_PPO_MAX_TOKEN_LEN_PER_GPU="${MIN_TRAIN_TOKEN_LEN_PER_GPU}"
fi
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-${DEFAULT_PPO_MAX_TOKEN_LEN_PER_GPU}}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${PPO_MAX_TOKEN_LEN_PER_GPU}}"
REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}}"
ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}}"
TURN_OVERLONG_BUFFER_LENGTH="${TURN_OVERLONG_BUFFER_LENGTH:-1024}"
TURN_OVERLONG_COST_COEF="${TURN_OVERLONG_COST_COEF:-0.75}"
TURN_SOFT_OVERLONG_ENABLE="${TURN_SOFT_OVERLONG_ENABLE:-false}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
VLLM_MM_PROCESSOR_CACHE_GB="${VLLM_MM_PROCESSOR_CACHE_GB:-0}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-true}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-true}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-false}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:--1}"
ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-true}"
AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-8}"
TOOL_MAX_CONSECUTIVE_OOM="${TOOL_MAX_CONSECUTIVE_OOM:-5}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
LR_SCHEDULER_TYPE="constant"
CLIP_RATIO="${CLIP_RATIO:-0.2}"
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
USE_KL_LOSS="${USE_KL_LOSS:-true}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.04}"
KL_LOSS_TYPE="${KL_LOSS_TYPE:-low_var_kl+}"
LOSS_AGG_MODE="${LOSS_AGG_MODE:-seq-mean-token-mean}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[console,swanlab]}"
TRAIN_LOG_GENERATIONS_FREQ="${TRAIN_LOG_GENERATIONS_FREQ:-10}"
TRAIN_LOG_GENERATIONS_NUM="${TRAIN_LOG_GENERATIONS_NUM:-4}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-${SAVE_FREQ}}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-0}"
VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-2048}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.0}"
VAL_TOP_P="${VAL_TOP_P:-1.0}"
VAL_TOP_K="${VAL_TOP_K:--1}"
VAL_PRESENCE_PENALTY="${VAL_PRESENCE_PENALTY:-0.0}"
VAL_MIN_P="${VAL_MIN_P:-0.0}"
VAL_REPETITION_PENALTY="${VAL_REPETITION_PENALTY:-1.0}"
VAL_NUM_GENERATIONS="${VAL_NUM_GENERATIONS:-1}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-false}"
VAL_SHUFFLE="${VAL_SHUFFLE:-false}"
VAL_DYNAMIC_SCHEDULING="${VAL_DYNAMIC_SCHEDULING:-true}"
VAL_MAX_IN_FLIGHT="${VAL_MAX_IN_FLIGHT:-24}"
VAL_PROGRESS_INTERVAL="${VAL_PROGRESS_INTERVAL:-10}"
VAL_SAVE_DETAILED_RESULTS="${VAL_SAVE_DETAILED_RESULTS:-true}"
VAL_RESULTS_DIR="${VAL_RESULTS_DIR:-${CHECKPOINT_DIR}/validation}"
ROLLOUT_PROGRESS="${ROLLOUT_PROGRESS:-true}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
MAX_RESUME_CKPTS="${MAX_RESUME_CKPTS:-3}"
ARCHIVE_HF_CHECKPOINTS="${ARCHIVE_HF_CHECKPOINTS:-true}"
ARCHIVED_CHECKPOINT_DIR="${ARCHIVED_CHECKPOINT_DIR:-${CHECKPOINT_DIR}/archived}"
FILTER_GROUPS_ENABLE="${FILTER_GROUPS_ENABLE:-false}"
FILTER_GROUPS_METRIC="${FILTER_GROUPS_METRIC:-trajectory_reward}"
FILTER_GROUPS_MAX_NUM_GEN_BATCHES="${FILTER_GROUPS_MAX_NUM_GEN_BATCHES:-0}"
MIN_TRAJECTORIES_AFTER_ROLLOUT_FILTER="${MIN_TRAJECTORIES_AFTER_ROLLOUT_FILTER:-4}"
MIN_REWARD_STD="${MIN_REWARD_STD:-0.2}"
TASK_BALANCED_SAMPLING="${TASK_BALANCED_SAMPLING:-true}"
TASK_SAMPLING_ALPHA="${TASK_SAMPLING_ALPHA:-0.2}"
TASK_MAX_NO_PROGRESS_CYCLES="${TASK_MAX_NO_PROGRESS_CYCLES:-3}"
TRAJECTORY_STEP_COST_ENABLE="${TRAJECTORY_STEP_COST_ENABLE:-false}"
TRAJECTORY_STEP_COST="${TRAJECTORY_STEP_COST:-0.2}"
TURN_OUTPUT_FORMAT_ERROR_COST="${TURN_OUTPUT_FORMAT_ERROR_COST:-0.2}"
TURN_TOOL_ARGS_ERROR_COST="${TURN_TOOL_ARGS_ERROR_COST:-0.2}"
TURN_TRUNCATION_ERROR_COST="${TURN_TRUNCATION_ERROR_COST:-0.5}"
TURN_LOCAL_COST_MODE="${TURN_LOCAL_COST_MODE:-per_event_floor}"
DRY_RUN="${DRY_RUN:-0}"

case "${TRAJECTORY_STEP_COST_ENABLE}" in
    true|True|TRUE|1|yes|Yes|YES)
        TRAJECTORY_STEP_COST_ENABLE=true
        ;;
    false|False|FALSE|0|no|No|NO)
        TRAJECTORY_STEP_COST_ENABLE=false
        ;;
    *)
        echo "TRAJECTORY_STEP_COST_ENABLE must be a boolean; got ${TRAJECTORY_STEP_COST_ENABLE}." >&2
        exit 1
        ;;
esac

case "${TRAJECTORY_EQUAL_WEIGHT}" in
    true|True|TRUE|1|yes|Yes|YES)
        TRAJECTORY_EQUAL_WEIGHT=true
        ;;
    false|False|FALSE|0|no|No|NO)
        TRAJECTORY_EQUAL_WEIGHT=false
        ;;
    *)
        echo "TRAJECTORY_EQUAL_WEIGHT must be a boolean; got ${TRAJECTORY_EQUAL_WEIGHT}." >&2
        exit 1
        ;;
esac

case "${ROLLOUT_DO_SAMPLE}" in
    true|True|TRUE|1|yes|Yes|YES)
        ROLLOUT_DO_SAMPLE=true
        ;;
    false|False|FALSE|0|no|No|NO)
        ROLLOUT_DO_SAMPLE=false
        ;;
    *)
        echo "ROLLOUT_DO_SAMPLE must be a boolean; got ${ROLLOUT_DO_SAMPLE}." >&2
        exit 1
        ;;
esac

case "${FILTER_GROUPS_ENABLE}" in
    true|True|TRUE|1|yes|Yes|YES)
        FILTER_GROUPS_ENABLE=true
        ;;
    false|False|FALSE|0|no|No|NO)
        FILTER_GROUPS_ENABLE=false
        ;;
    *)
        echo "FILTER_GROUPS_ENABLE must be a boolean; got ${FILTER_GROUPS_ENABLE}." >&2
        exit 1
        ;;
esac

case "${TURN_SOFT_OVERLONG_ENABLE}" in
    true|True|TRUE|1|yes|Yes|YES)
        TURN_SOFT_OVERLONG_ENABLE=true
        ;;
    false|False|FALSE|0|no|No|NO)
        TURN_SOFT_OVERLONG_ENABLE=false
        ;;
    *)
        echo "TURN_SOFT_OVERLONG_ENABLE must be a boolean; got ${TURN_SOFT_OVERLONG_ENABLE}." >&2
        exit 1
        ;;
esac

case "${TURN_LOCAL_COST_MODE}" in
    per_event_floor|trajectory_weighted_additive)
        ;;
    *)
        echo "TURN_LOCAL_COST_MODE must be per_event_floor or trajectory_weighted_additive; got ${TURN_LOCAL_COST_MODE}." >&2
        exit 1
        ;;
esac
if [[ "${TRAJECTORY_EQUAL_WEIGHT}" == "true" \
    && "${TURN_LOCAL_COST_MODE}" != "per_event_floor" ]]; then
    echo "TRAJECTORY_EQUAL_WEIGHT=true requires TURN_LOCAL_COST_MODE=per_event_floor so trajectory weights do not scale local costs." >&2
    exit 1
fi

case "${TASK_BALANCED_SAMPLING}" in
    true|True|TRUE|1|yes|Yes|YES)
        TASK_BALANCED_SAMPLING=true
        ;;
    false|False|FALSE|0|no|No|NO)
        TASK_BALANCED_SAMPLING=false
        ;;
    *)
        echo "TASK_BALANCED_SAMPLING must be a boolean; got ${TASK_BALANCED_SAMPLING}." >&2
        exit 1
        ;;
esac

case "${VAL_DYNAMIC_SCHEDULING}" in
    true|True|TRUE|1|yes|Yes|YES)
        VAL_DYNAMIC_SCHEDULING=true
        ;;
    false|False|FALSE|0|no|No|NO)
        VAL_DYNAMIC_SCHEDULING=false
        ;;
    *)
        echo "VAL_DYNAMIC_SCHEDULING must be a boolean; got ${VAL_DYNAMIC_SCHEDULING}." >&2
        exit 1
        ;;
esac

if (( ROLLOUT_TP_SIZE <= 0 || ROLLOUT_DP_SIZE <= 0 )); then
    echo "ROLLOUT_TP_SIZE (${ROLLOUT_TP_SIZE}) and ROLLOUT_DP_SIZE (${ROLLOUT_DP_SIZE}) must be positive." >&2
    exit 1
fi
if (( TRAIN_BATCH_SIZE <= 0 || VAL_BATCH_SIZE <= 0 || PPO_MINI_BATCH_SIZE <= 0 || PPO_EPOCHS <= 0 )); then
    echo "TRAIN_BATCH_SIZE (${TRAIN_BATCH_SIZE}), VAL_BATCH_SIZE (${VAL_BATCH_SIZE}), PPO_MINI_BATCH_SIZE (${PPO_MINI_BATCH_SIZE}), and PPO_EPOCHS (${PPO_EPOCHS}) must be positive." >&2
    exit 1
fi
if [[ "${PER_TURN_NUM_MINI_BATCHES}" == "auto" ]]; then
    if (( TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE != 0 )); then
        echo "Cannot derive PER_TURN_NUM_MINI_BATCHES: TRAIN_BATCH_SIZE (${TRAIN_BATCH_SIZE}) must be divisible by PPO_MINI_BATCH_SIZE (${PPO_MINI_BATCH_SIZE}). Set PER_TURN_NUM_MINI_BATCHES explicitly." >&2
        exit 1
    fi
    PER_TURN_NUM_MINI_BATCHES=$((TRAIN_BATCH_SIZE / PPO_MINI_BATCH_SIZE))
elif [[ ! "${PER_TURN_NUM_MINI_BATCHES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PER_TURN_NUM_MINI_BATCHES must be 'auto' or a positive integer; got ${PER_TURN_NUM_MINI_BATCHES}." >&2
    exit 1
fi
case "${LOSS_AGG_MODE}" in
    token-mean|seq-mean-token-mean)
        ;;
    *)
        echo "VisHarness LOSS_AGG_MODE must be token-mean or seq-mean-token-mean; got: ${LOSS_AGG_MODE}" >&2
        exit 1
        ;;
esac
if [[ "${TRAJECTORY_EQUAL_WEIGHT}" == "true" || "${TURN_LOCAL_COST_MODE}" == "per_event_floor" ]] \
    && [[ "${LOSS_AGG_MODE}" != "seq-mean-token-mean" ]]; then
    echo "Trajectory-equal weighting and per-event local costs require LOSS_AGG_MODE=seq-mean-token-mean." >&2
    exit 1
fi
if (( ULYSSES_SEQUENCE_PARALLEL_SIZE <= 0 )); then
    echo "ULYSSES_SEQUENCE_PARALLEL_SIZE (${ULYSSES_SEQUENCE_PARALLEL_SIZE}) must be positive." >&2
    exit 1
fi
if (( NUM_GPUS % ULYSSES_SEQUENCE_PARALLEL_SIZE != 0 )); then
    echo "NUM_GPUS (${NUM_GPUS}) must be divisible by ULYSSES_SEQUENCE_PARALLEL_SIZE (${ULYSSES_SEQUENCE_PARALLEL_SIZE})." >&2
    exit 1
fi
if (( PPO_MAX_TOKEN_LEN_PER_GPU <= 0 || REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU <= 0 || ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU <= 0 )); then
    echo "PPO_MAX_TOKEN_LEN_PER_GPU, REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU, and ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU must all be positive." >&2
    exit 1
fi
if (( PPO_MAX_TOKEN_LEN_PER_GPU * ULYSSES_SEQUENCE_PARALLEL_SIZE < PER_TURN_MAX_TOKEN_LENGTH )); then
    echo "Actor token budget is too small: PPO_MAX_TOKEN_LEN_PER_GPU (${PPO_MAX_TOKEN_LEN_PER_GPU}) * ULYSSES_SEQUENCE_PARALLEL_SIZE (${ULYSSES_SEQUENCE_PARALLEL_SIZE}) must be >= PER_TURN_MAX_TOKEN_LENGTH (${PER_TURN_MAX_TOKEN_LENGTH})." >&2
    exit 1
fi
if (( REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU * ULYSSES_SEQUENCE_PARALLEL_SIZE < PER_TURN_MAX_TOKEN_LENGTH )); then
    echo "Reference log-prob token budget is too small: REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU (${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU}) * ULYSSES_SEQUENCE_PARALLEL_SIZE (${ULYSSES_SEQUENCE_PARALLEL_SIZE}) must be >= PER_TURN_MAX_TOKEN_LENGTH (${PER_TURN_MAX_TOKEN_LENGTH})." >&2
    exit 1
fi
if (( ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU * ULYSSES_SEQUENCE_PARALLEL_SIZE < PER_TURN_MAX_TOKEN_LENGTH )); then
    echo "Actor old-log-prob token budget is too small: ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU (${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU}) * ULYSSES_SEQUENCE_PARALLEL_SIZE (${ULYSSES_SEQUENCE_PARALLEL_SIZE}) must be >= PER_TURN_MAX_TOKEN_LENGTH (${PER_TURN_MAX_TOKEN_LENGTH})." >&2
    exit 1
fi
ACTOR_DP_SIZE=$((NUM_GPUS / ULYSSES_SEQUENCE_PARALLEL_SIZE))
if (( TOOL_MAX_CONSECUTIVE_OOM <= 0 )); then
    echo "TOOL_MAX_CONSECUTIVE_OOM (${TOOL_MAX_CONSECUTIVE_OOM}) must be positive." >&2
    exit 1
fi
ROLLOUT_REPLICA_WORLD_SIZE=$((ROLLOUT_TP_SIZE * ROLLOUT_DP_SIZE))
if (( NUM_GPUS % ROLLOUT_REPLICA_WORLD_SIZE != 0 )); then
    echo "NUM_GPUS (${NUM_GPUS}) must be divisible by ROLLOUT_TP_SIZE * ROLLOUT_DP_SIZE (${ROLLOUT_TP_SIZE} * ${ROLLOUT_DP_SIZE} = ${ROLLOUT_REPLICA_WORLD_SIZE})." >&2
    exit 1
fi
if (( DATA_MAX_PROMPT_LENGTH <= 0 || PER_TURN_MAX_PROMPT_LENGTH <= 0 || ROLLOUT_MAX_PROMPT_LENGTH <= 0 || VAL_MAX_PROMPT_LENGTH <= 0 || ROLLOUT_PLACEHOLDER_PROMPT_LENGTH <= 0 || PER_TURN_MAX_RESPONSE_LENGTH <= 0 || ROLLOUT_PLACEHOLDER_RESPONSE_LENGTH <= 0 || VAL_MAX_TOKENS <= 0 || VAL_NUM_GENERATIONS <= 0 || MAX_AGENT_TURNS <= 0 )); then
    echo "DATA_MAX_PROMPT_LENGTH, PER_TURN_MAX_PROMPT_LENGTH, ROLLOUT_MAX_PROMPT_LENGTH, VAL_MAX_PROMPT_LENGTH, ROLLOUT_PLACEHOLDER_PROMPT_LENGTH, PER_TURN_MAX_RESPONSE_LENGTH, ROLLOUT_PLACEHOLDER_RESPONSE_LENGTH, VAL_MAX_TOKENS, VAL_NUM_GENERATIONS, and MAX_AGENT_TURNS must all be positive." >&2
    exit 1
fi
if (( VAL_MAX_PROMPT_LENGTH >= MAX_MODEL_LEN )); then
    echo "VAL_MAX_PROMPT_LENGTH (${VAL_MAX_PROMPT_LENGTH}) must be smaller than MAX_MODEL_LEN (${MAX_MODEL_LEN}) so validation can generate at least one token." >&2
    exit 1
fi
if (( VAL_MAX_IN_FLIGHT <= 0 || VAL_PROGRESS_INTERVAL <= 0 )); then
    echo "VAL_MAX_IN_FLIGHT (${VAL_MAX_IN_FLIGHT}) and VAL_PROGRESS_INTERVAL (${VAL_PROGRESS_INTERVAL}) must be positive." >&2
    exit 1
fi
if [[ "${FILTER_GROUPS_ENABLE}" == "true" || "${TASK_BALANCED_SAMPLING}" == "true" ]] && ((
    MIN_TRAJECTORIES_AFTER_ROLLOUT_FILTER <= 0
    || MIN_TRAJECTORIES_AFTER_ROLLOUT_FILTER > GROUP_SIZE
)); then
    echo "MIN_TRAJECTORIES_AFTER_ROLLOUT_FILTER (${MIN_TRAJECTORIES_AFTER_ROLLOUT_FILTER}) must be in [1, GROUP_SIZE=${GROUP_SIZE}] when reward group filtering or task-balanced sampling is enabled for GRPO." >&2
    exit 1
fi
if [[ "${FILTER_GROUPS_ENABLE}" == "true" && "${FILTER_GROUPS_METRIC}" != "trajectory_reward" ]]; then
    echo "FILTER_GROUPS_METRIC must be trajectory_reward for ordinary GRPO; got ${FILTER_GROUPS_METRIC}." >&2
    exit 1
fi
if ! [[ "${TASK_SAMPLING_ALPHA}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
    echo "TASK_SAMPLING_ALPHA must be a non-negative number; got ${TASK_SAMPLING_ALPHA}." >&2
    exit 1
fi
if ! [[ "${TASK_MAX_NO_PROGRESS_CYCLES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TASK_MAX_NO_PROGRESS_CYCLES must be a positive integer; got ${TASK_MAX_NO_PROGRESS_CYCLES}." >&2
    exit 1
fi
if ! [[ "${WARMUP_STEPS}" =~ ^[0-9]+$ ]]; then
    echo "WARMUP_STEPS must be a non-negative integer; got ${WARMUP_STEPS}." >&2
    exit 1
fi
if [[ "${TASK_BALANCED_SAMPLING}" == "true" && "${FILTER_GROUPS_MAX_NUM_GEN_BATCHES}" != "0" ]]; then
    echo "Task-balanced sampling requires FILTER_GROUPS_MAX_NUM_GEN_BATCHES=0; TASK_MAX_NO_PROGRESS_CYCLES is the dead-loop guard." >&2
    exit 1
fi
ROLLOUT_NUM_REPLICAS=$((NUM_GPUS / ROLLOUT_REPLICA_WORLD_SIZE))
MAX_NUM_SEQS=$((TRAIN_BATCH_SIZE * GROUP_SIZE))

VALIDATION_REQUIRED=false
case "${VAL_BEFORE_TRAIN}" in
    true|True|TRUE|1|yes|Yes|YES)
        VALIDATION_REQUIRED=true
        ;;
esac
if [[ "${TEST_FREQ}" =~ ^[0-9]+$ ]] && (( TEST_FREQ > 0 )); then
    VALIDATION_REQUIRED=true
fi
if [[ ! -e "${VAL_DATA}" && "${VALIDATION_REQUIRED}" != "true" ]]; then
    echo "Validation data not found and validation is disabled; using train data for unused val dataset: ${TRAIN_DATA}"
    VAL_DATA="${TRAIN_DATA}"
fi

for required_path in "${PYTHON_BIN}" "${MODEL_PATH}/config.json" "${TRAIN_DATA}" "${VAL_DATA}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path does not exist: ${required_path}" >&2
        exit 1
    fi
done

# Select the output-protocol parser from the model architecture metadata. Keep
# MODEL_FAMILY as an explicit override for checkpoints trained with a custom
# protocol that differs from their base architecture.
if [[ "${MODEL_FAMILY}" == "auto" ]]; then
    MODEL_TYPE="$(
        "${PYTHON_BIN}" - "${MODEL_PATH}/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as config_file:
    config = json.load(config_file)

print(config.get("model_type", ""))
PY
    )"
    case "${MODEL_TYPE}" in
        qwen3_vl)
            MODEL_FAMILY="qwen3_vl"
            ;;
        qwen3_5|qwen3_5_moe)
            MODEL_FAMILY="qwen3_5"
            ;;
        *)
            echo "Unsupported model_type '${MODEL_TYPE}' in ${MODEL_PATH}/config.json; set MODEL_FAMILY explicitly." >&2
            exit 1
            ;;
    esac
else
    case "${MODEL_FAMILY}" in
        qwen3_vl|qwen3-vl)
            MODEL_FAMILY="qwen3_vl"
            ;;
        qwen3_5|qwen3.5|qwen3-5|qwen3_5_moe)
            MODEL_FAMILY="qwen3_5"
            ;;
        *)
            echo "Unsupported MODEL_FAMILY '${MODEL_FAMILY}'; expected auto, qwen3_vl, or qwen3_5." >&2
            exit 1
            ;;
    esac
fi

mkdir -p \
    "${CACHE_ROOT}/tmp" \
    "${CACHE_ROOT}/pip" \
    "${CACHE_ROOT}/torch" \
    "${CACHE_ROOT}/torchinductor" \
    "${CACHE_ROOT}/triton" \
    "${CACHE_ROOT}/xdg" \
    "${RAY_TMP_ROOT}" \
    "${CHECKPOINT_DIR}" \
    "${ARCHIVED_CHECKPOINT_DIR}"

# Keep Ray's socket path short while storing its session data on VEPFS.
if [[ -L "${RAY_TMP_LINK}" ]]; then
    if [[ "$(readlink -f "${RAY_TMP_LINK}")" != "$(readlink -f "${RAY_TMP_ROOT}")" ]]; then
        echo "Ray temp symlink points elsewhere: ${RAY_TMP_LINK}" >&2
        exit 1
    fi
elif [[ -e "${RAY_TMP_LINK}" ]]; then
    echo "Ray temp path exists and is not a symlink: ${RAY_TMP_LINK}" >&2
    exit 1
else
    ln -s "${RAY_TMP_ROOT}" "${RAY_TMP_LINK}"
fi
mkdir -p "${RAY_TMP_DIR}"

# Hydra/Ray workers read these values from recipe/visharness/configs/*.yaml
# and reward code via os.getenv.
export VISHARNESS_MODEL_PATH="${MODEL_PATH}"
export VISHARNESS_MODEL_FAMILY="${MODEL_FAMILY}"
export VISHARNESS_TRAIN_DATA="${TRAIN_DATA}"
export VISHARNESS_VAL_DATA="${VAL_DATA}"
export VISHARNESS_PER_TURN_MAX_PROMPT_LENGTH="${PER_TURN_MAX_PROMPT_LENGTH}"
export VISHARNESS_PER_TURN_MAX_RESPONSE_LENGTH="${PER_TURN_MAX_RESPONSE_LENGTH}"
export VISHARNESS_ROLLOUT_MAX_PROMPT_LENGTH="${ROLLOUT_MAX_PROMPT_LENGTH}"
export VISHARNESS_VALIDATION_MAX_PROMPT_LENGTH="${VAL_MAX_PROMPT_LENGTH}"
export VISHARNESS_MAX_AGENT_TURNS="${MAX_AGENT_TURNS}"
export VISHARNESS_VALIDATION_MAX_TOKENS="${VAL_MAX_TOKENS}"
export VISHARNESS_VALIDATION_TEMPERATURE="${VAL_TEMPERATURE}"
export VISHARNESS_VALIDATION_TOP_P="${VAL_TOP_P}"
export VISHARNESS_VALIDATION_TOP_K="${VAL_TOP_K}"
export VISHARNESS_VALIDATION_PRESENCE_PENALTY="${VAL_PRESENCE_PENALTY}"
export VISHARNESS_VALIDATION_MIN_P="${VAL_MIN_P}"
export VISHARNESS_VALIDATION_REPETITION_PENALTY="${VAL_REPETITION_PENALTY}"
export VISHARNESS_VALIDATION_RESULTS_DIR="${VAL_RESULTS_DIR}"
export VISHARNESS_TRAJECTORY_STEP_COST_ENABLE="${TRAJECTORY_STEP_COST_ENABLE}"
export VISHARNESS_TRAJECTORY_STEP_COST="${TRAJECTORY_STEP_COST}"

# Keep downloaded/generated caches away from the small container root filesystem.
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_ROOT}/huggingface/datasets}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_ROOT}/pip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-${SEED}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# vLLM's V1 sleep/free-cache memory pool uses CuMemAllocator, which currently
# asserts if PyTorch expandable segments are enabled. Keep this unset by default
# so colocated vLLM rollout can start and later release cache for actor training.
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    echo "Warning: unsetting PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF} because vLLM CuMemAllocator is incompatible with expandable_segments:True." >&2
    unset PYTORCH_CUDA_ALLOC_CONF
fi
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-${PROJECT_ROOT}/swanlog}"
case "${SWANLAB_RESUME_ENABLED}" in
    true|True|TRUE|1|yes|Yes|YES)
        SWANLAB_RESUME="${SWANLAB_RESUME:-${SWANLAB_RESUME_MODE}}"
        if [[ -z "${SWANLAB_RUN_ID}" ]]; then
            echo "SWANLAB_RESUME_ENABLED=true requires SWANLAB_RUN_ID to be set." >&2
            exit 1
        fi
        export SWANLAB_RESUME
        export SWANLAB_RUN_ID
        ;;
    false|False|FALSE|0|no|No|NO)
        unset SWANLAB_RESUME
        unset SWANLAB_RUN_ID
        ;;
    *)
        echo "SWANLAB_RESUME_ENABLED must be true or false, got: ${SWANLAB_RESUME_ENABLED}" >&2
        exit 1
        ;;
esac

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS - 1)))"
    export CUDA_VISIBLE_DEVICES
fi
export VISHARNESS_TOOL_MAX_CONSECUTIVE_OOM="${TOOL_MAX_CONSECUTIVE_OOM}"

cd "${PROJECT_ROOT}"

command=(
    "${PYTHON_BIN}" -m recipe.visharness.main_visharness
    "data.seed=${SEED}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.val_batch_size=${VAL_BATCH_SIZE}"
    "data.validation_shuffle=${VAL_SHUFFLE}"
    # data.max_prompt_length only filters original dataset prompts.
    # Do not override data.max_response_length here: real per-turn response
    # length is visharness.per_turn.max_response_length, and rollout placeholder
    # width is actor_rollout_ref.rollout.response_length.
    "data.max_prompt_length=${DATA_MAX_PROMPT_LENGTH}"
    "data.dataloader_num_workers=4"
    "+actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPLEMENTATION}"
    "actor_rollout_ref.model.enable_gradient_checkpointing=true"
    "actor_rollout_ref.model.use_remove_padding=true"
    "actor_rollout_ref.actor.optim.lr=${LEARNING_RATE}"
    "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0"
    "actor_rollout_ref.actor.optim.lr_warmup_steps=${WARMUP_STEPS}"
    "actor_rollout_ref.actor.optim.lr_scheduler_type=${LR_SCHEDULER_TYPE}"
    "actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO}"
    "actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}"
    "actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}"
    "actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}"
    "actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}"
    "actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS}"
    "actor_rollout_ref.actor.use_dynamic_bsz=true"
    "actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ULYSSES_SEQUENCE_PARALLEL_SIZE}"
    "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${ULYSSES_SEQUENCE_PARALLEL_SIZE}"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
    "actor_rollout_ref.actor.data_loader_seed=${SEED}"
    "actor_rollout_ref.actor.shuffle=false"
    "actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE}"
    "actor_rollout_ref.actor.fsdp_config.seed=${SEED}"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD}"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD}"
    "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]"
    "actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]"
    "actor_rollout_ref.ref.ulysses_sequence_parallel_size=${ULYSSES_SEQUENCE_PARALLEL_SIZE}"
    "actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${ULYSSES_SEQUENCE_PARALLEL_SIZE}"
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU}"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.n=${GROUP_SIZE}"
    "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}"
    "actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P}"
    "actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K}"
    "actor_rollout_ref.rollout.do_sample=${ROLLOUT_DO_SAMPLE}"
    "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}"
    "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}"
    "actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K}"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE}"
    "actor_rollout_ref.rollout.val_kwargs.n=${VAL_NUM_GENERATIONS}"
    "actor_rollout_ref.rollout.prompt_length=${ROLLOUT_PLACEHOLDER_PROMPT_LENGTH}"
    "actor_rollout_ref.rollout.response_length=${ROLLOUT_PLACEHOLDER_RESPONSE_LENGTH}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE}"
    "actor_rollout_ref.rollout.data_parallel_size=${ROLLOUT_DP_SIZE}"
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=${VLLM_MM_PROCESSOR_CACHE_GB}"
    "actor_rollout_ref.rollout.free_cache_engine=true"
    "actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER}"
    "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
    "actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS}"
    "actor_rollout_ref.rollout.agent.num_workers=${AGENT_LOOP_WORKERS}"
    "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_AGENT_TURNS}"
    "actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_AGENT_TURNS}"
    "algorithm.filter_groups.enable=${FILTER_GROUPS_ENABLE}"
    "algorithm.filter_groups.metric=${FILTER_GROUPS_METRIC}"
    "algorithm.filter_groups.max_num_gen_batches=${FILTER_GROUPS_MAX_NUM_GEN_BATCHES}"
    "visharness.filter_groups.min_trajectories_after_rollout_filter=${MIN_TRAJECTORIES_AFTER_ROLLOUT_FILTER}"
    "visharness.filter_groups.min_reward_std=${MIN_REWARD_STD}"
    "visharness.task_sampling.enable=${TASK_BALANCED_SAMPLING}"
    "visharness.task_sampling.alpha=${TASK_SAMPLING_ALPHA}"
    "visharness.task_sampling.max_no_progress_cycles=${TASK_MAX_NO_PROGRESS_CYCLES}"
    "visharness.rollout_progress=${ROLLOUT_PROGRESS}"
    "visharness.tool_call.max_consecutive_oom=${TOOL_MAX_CONSECUTIVE_OOM}"
    "visharness.per_turn.max_prompt_length=${PER_TURN_MAX_PROMPT_LENGTH}"
    "visharness.per_turn.max_response_length=${PER_TURN_MAX_RESPONSE_LENGTH}"
    "visharness.per_turn.trajectory_equal_weight=${TRAJECTORY_EQUAL_WEIGHT}"
    "visharness.per_turn.num_mini_batches=${PER_TURN_NUM_MINI_BATCHES}"
    "visharness.trajectory_reward.enable_step_cost=${TRAJECTORY_STEP_COST_ENABLE}"
    "visharness.trajectory_reward.step_cost=${TRAJECTORY_STEP_COST}"
    "visharness.per_turn_advantage.output_format_error_cost=${TURN_OUTPUT_FORMAT_ERROR_COST}"
    "visharness.per_turn_advantage.tool_args_error_cost=${TURN_TOOL_ARGS_ERROR_COST}"
    "visharness.per_turn_advantage.truncation_error_cost=${TURN_TRUNCATION_ERROR_COST}"
    "visharness.per_turn_advantage.local_cost_mode=${TURN_LOCAL_COST_MODE}"
    "visharness.per_turn_advantage.soft_overlong_enabled=${TURN_SOFT_OVERLONG_ENABLE}"
    "visharness.per_turn_advantage.overlong_buffer_length=${TURN_OVERLONG_BUFFER_LENGTH}"
    "visharness.per_turn_advantage.overlong_cost_coef=${TURN_OVERLONG_COST_COEF}"
    "visharness.train_generations.log_freq=${TRAIN_LOG_GENERATIONS_FREQ}"
    "visharness.train_generations.num_samples=${TRAIN_LOG_GENERATIONS_NUM}"
    "visharness.validation.dynamic_scheduling=${VAL_DYNAMIC_SCHEDULING}"
    "visharness.validation.max_in_flight=${VAL_MAX_IN_FLIGHT}"
    "visharness.validation.progress_interval=${VAL_PROGRESS_INTERVAL}"
    "visharness.validation.save_detailed_results=${VAL_SAVE_DETAILED_RESULTS}"
    "visharness.validation.results_dir=${VAL_RESULTS_DIR}"
    "visharness.archive_checkpoints.enable=${ARCHIVE_HF_CHECKPOINTS}"
    "visharness.archive_checkpoints.dir=${ARCHIVED_CHECKPOINT_DIR}"
    "trainer.n_gpus_per_node=${NUM_GPUS}"
    "trainer.nnodes=1"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.logger=${TRAINER_LOGGER}"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    "trainer.default_local_dir=${CHECKPOINT_DIR}"
    "trainer.resume_mode=auto"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.max_actor_ckpt_to_keep=${MAX_RESUME_CKPTS}"
    "trainer.max_critic_ckpt_to_keep=${MAX_RESUME_CKPTS}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.val_before_train=${VAL_BEFORE_TRAIN}"
    "trainer.log_val_generations=${LOG_VAL_GENERATIONS}"
    "+ray_kwargs.ray_init._temp_dir=${RAY_TMP_DIR}"
    "$@"
)

echo "=== VisHarness training ==="
echo "Python:       ${PYTHON_BIN}"
echo "Model:        ${MODEL_PATH}"
echo "Model family: ${MODEL_FAMILY}"
echo "Train data:   ${TRAIN_DATA}"
echo "Val data:     ${VAL_DATA}"
echo "GPUs:         ${CUDA_VISIBLE_DEVICES}"
echo "Seed:         ${SEED} (PYTHONHASHSEED=${PYTHONHASHSEED})"
echo "Train parallel: Ulysses SP=${ULYSSES_SEQUENCE_PARALLEL_SIZE}, actor DP=${ACTOR_DP_SIZE}"
echo "Train tokens: actor=${PPO_MAX_TOKEN_LEN_PER_GPU}/GPU, ref_logprob=${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU}/GPU, old_logprob=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU}/GPU, max_sequence=${PER_TURN_MAX_TOKEN_LENGTH}"
echo "Rollout:      TP=${ROLLOUT_TP_SIZE}, DP=${ROLLOUT_DP_SIZE}, replicas=${ROLLOUT_NUM_REPLICAS}, max_num_seqs=${MAX_NUM_SEQS}"
echo "Prompt batch: ${TRAIN_BATCH_SIZE}"
echo "Group size:   ${GROUP_SIZE}"
echo "Per-turn PPO: fixed_mini_batches=${PER_TURN_NUM_MINI_BATCHES}, optimizer_steps=$((PER_TURN_NUM_MINI_BATCHES * PPO_EPOCHS)) per outer step (ppo_epochs=${PPO_EPOCHS})"
echo "Per-turn policy weighting: $([[ "${TRAJECTORY_EQUAL_WEIGHT}" == "true" ]] && echo trajectory-equal || echo turn-equal) (trajectory_equal_weight=${TRAJECTORY_EQUAL_WEIGHT})"
echo "Data prompt filter: ${DATA_MAX_PROMPT_LENGTH}"
echo "Per-turn train prompt: ${PER_TURN_MAX_PROMPT_LENGTH}"
echo "Rollout max prompt: ${ROLLOUT_MAX_PROMPT_LENGTH}"
echo "Validation max prompt: ${VAL_MAX_PROMPT_LENGTH} (model-context limit; independent of training)"
echo "Rollout placeholder prompt: ${ROLLOUT_PLACEHOLDER_PROMPT_LENGTH}"
echo "vLLM tokens:   max_model_len=${MAX_MODEL_LEN}, max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
echo "Turn max:     ${PER_TURN_MAX_RESPONSE_LENGTH}"
echo "Placeholder response: ${ROLLOUT_PLACEHOLDER_RESPONSE_LENGTH}"
echo "Agent turns:  ${MAX_AGENT_TURNS}"
echo "Tool OOM:     invalidate trajectory after ${TOOL_MAX_CONSECUTIVE_OOM} consecutive OOM responses"
echo "Sampling:     do_sample=${ROLLOUT_DO_SAMPLE}, temperature=${ROLLOUT_TEMPERATURE}, top_p=${ROLLOUT_TOP_P}, top_k=${ROLLOUT_TOP_K}"
if [[ "${TRAJECTORY_STEP_COST_ENABLE}" == "true" ]]; then
    echo "Trajectory reward: task_reward - ${TRAJECTORY_STEP_COST} * num_turns (turn penalty enabled)"
else
    echo "Trajectory reward: task_reward (turn penalty disabled; configured step cost=${TRAJECTORY_STEP_COST})"
fi
echo "Turn advantage costs: mode=${TURN_LOCAL_COST_MODE}, format=${TURN_OUTPUT_FORMAT_ERROR_COST}, args=${TURN_TOOL_ARGS_ERROR_COST}, truncation=${TURN_TRUNCATION_ERROR_COST}, soft_overlong_enabled=${TURN_SOFT_OVERLONG_ENABLE}, soft_overlong_max=${TURN_OVERLONG_COST_COEF}, buffer=${TURN_OVERLONG_BUFFER_LENGTH}"
echo "LR schedule:  type=${LR_SCHEDULER_TYPE}, lr=${LEARNING_RATE}, warmup_steps=${WARMUP_STEPS}"
echo "Clip ratio:   low=${CLIP_RATIO_LOW}, high=${CLIP_RATIO_HIGH}"
echo "KL loss:      enabled=${USE_KL_LOSS}, beta=${KL_LOSS_COEF}, type=${KL_LOSS_TYPE}"
echo "Loss aggregate: ${LOSS_AGG_MODE}"
echo "Memory:       gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}, actor_param_offload=${ACTOR_PARAM_OFFLOAD}, actor_optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD}"
echo "vLLM MM cache: mm_processor_cache_gb=${VLLM_MM_PROCESSOR_CACHE_GB}"
echo "vLLM eager:   ${ROLLOUT_ENFORCE_EAGER}"
echo "Validation:   before_train=${VAL_BEFORE_TRAIN}, test_freq=${TEST_FREQ}, loader_batch=${VAL_BATCH_SIZE}, n=${VAL_NUM_GENERATIONS}, shuffle=${VAL_SHUFFLE}, dynamic=${VAL_DYNAMIC_SCHEDULING}, max_in_flight=${VAL_MAX_IN_FLIGHT}, log_generations=${LOG_VAL_GENERATIONS}"
echo "Val sampling: max_tokens=${VAL_MAX_TOKENS}, temperature=${VAL_TEMPERATURE}, top_p=${VAL_TOP_P}, top_k=${VAL_TOP_K}, presence_penalty=${VAL_PRESENCE_PENALTY}, min_p=${VAL_MIN_P}, repetition_penalty=${VAL_REPETITION_PENALTY}"
echo "Val results:  save_details=${VAL_SAVE_DETAILED_RESULTS}, dir=${VAL_RESULTS_DIR}"
echo "Attention:    ${ATTN_IMPLEMENTATION}"
echo "Logger:       ${TRAINER_LOGGER} (train generations every ${TRAIN_LOG_GENERATIONS_FREQ} step(s), ${TRAIN_LOG_GENERATIONS_NUM} sample(s))"
echo "SwanLab:      mode=${SWANLAB_MODE}, resume_enabled=${SWANLAB_RESUME_ENABLED}, resume=${SWANLAB_RESUME:-}, run_id=${SWANLAB_RUN_ID:-}"
echo "Progress:     rollout=${ROLLOUT_PROGRESS}"
if [[ "${FILTER_GROUPS_ENABLE}" == "true" ]]; then
    echo "Group filter: true (${FILTER_GROUPS_METRIC}, surviving trajectories>=${MIN_TRAJECTORIES_AFTER_ROLLOUT_FILTER}, sample std>=${MIN_REWARD_STD})"
elif [[ "${TASK_BALANCED_SAMPLING}" == "true" ]]; then
    echo "Group filter: false (reward-std filtering disabled; invalid trajectories and prompt groups with <${MIN_TRAJECTORIES_AFTER_ROLLOUT_FILTER} survivors are dropped before exact-quota top-up)"
else
    echo "Group filter: false (all surviving valid trajectories are used; no group-size/reward-variance filtering)"
fi
echo "Task sampling: enabled=${TASK_BALANCED_SAMPLING}, alpha=${TASK_SAMPLING_ALPHA}, independent source cycling, exact post-filter quotas, no-progress cycles=${TASK_MAX_NO_PROGRESS_CYCLES}"
echo "Checkpoints:  ${CHECKPOINT_DIR}"
echo "Resume keep:  ${MAX_RESUME_CKPTS}"
echo "Archived HF:  ${ARCHIVE_HF_CHECKPOINTS} (${ARCHIVED_CHECKPOINT_DIR})"

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'Command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    exit 0
fi

exec "${command[@]}"
