#!/usr/bin/env bash
set -euo pipefail

# Cross-model generalization: Moonlight-16B-A3B Base (DeepseekV3 MoE: 64 routed
# experts top-6 + 2 shared, sigmoid routing with expert bias, layer 0 dense) trained
# on GSM8K train, validated on GSM8K test + MATH500. Reuses the Qwen3 length-bucket
# RDC + token-level TIS off-policy recipe and only overrides what differs for Moonlight.
# trainer.val_before_train=True gives the step-0 validation.
#
# Usage: OFF_POLICY_MINI_STEPS=4 EXPERIMENT_NAME_BASE=moonlight-gsm8k-off-4-RDC-TIS \
#            bash low_precision/moonlight_gsm8k_off-policy-RDC-TIS.sh [extra hydra overrides]

RAY_DATA_HOME=${RAY_DATA_HOME:-"/inspire/hdd/project/qianghuaxuexi/public"}
WORKING_DIR=${WORKING_DIR:-"/inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence"}

# Moonlight's tiktoken tokenizer reads its local vocab through blobfile (tiktoken>=0.11);
# compute nodes are offline, so install from the shared wheel dir on every node.
python3 -c "import blobfile" 2>/dev/null || pip install --quiet --no-index \
    --find-links "${RAY_DATA_HOME}/wheels" blobfile || \
    echo "WARNING: blobfile install failed; Moonlight tokenizer will not load"

# Base model (user decision 2026-11-12). The "-stop-imend" dir symlinks the Base weights and
# only changes generation_config.json to stop on <|im_end|> as well as [EOS]: Base's own config
# stops only on [EOS], so chat-templated rollouts would otherwise run to the length limit.
export MODEL_PATH=${MODEL_PATH:-"/inspire/hdd3/project/qianghuaxuexi/public/models/moonshotai/Moonlight-16B-A3B-stop-imend"}
export TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/datasets/gsm8k/train.parquet"}
# 1319 GSM8K test + 500 MATH500; validation metrics are split per data_source.
export TEST_FILE=${TEST_FILE:-"${WORKING_DIR}/data/gsm8k_math500/test.parquet"}

# Moonlight max_position_embeddings=8192. GSM8K prompts are short; MATH500 gets a
# larger validation budget than the 1K training budget.
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
export VAL_MAX_RESPONSE_LENGTH=${VAL_MAX_RESPONSE_LENGTH:-3072}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-4096}

# RDC knobs rescaled for 1K responses. Length buckets cover the 1K budget. The
# expert-usage score is 0.5*|dc|_1/(N+tau) with N = response tokens * top-k slots;
# tau=4096 was sized for Qwen's ~10K-token responses and would flatten every score
# at ~300-token GSM8K lengths, so the default shrinks it proportionally.
export ROUTER_MISMATCH_RS_LENGTH_BUCKET_EDGES=${ROUTER_MISMATCH_RS_LENGTH_BUCKET_EDGES:-"[256,512,768]"}
export ROUTER_EXPERT_USAGE_SMOOTHING_TAU=${ROUTER_EXPERT_USAGE_SMOOTHING_TAU:-256.0}
export ROUTER_EXPERT_USAGE_NUM_EXPERTS=${ROUTER_EXPERT_USAGE_NUM_EXPERTS:-64}

# Training TP must be 1. MLA's linear_kv_down_proj (kv_a_proj_with_mqa, [576, 2048]) is
# duplicated on every TP rank, but the Megatron->vLLM weight export all-gathers and
# concatenates it like a column-parallel weight, so TP=4 sends [2304, 2048] and the first
# weight sync crashes. At TP=1 the export takes no TP-merge path. Experts stay sharded by EP.
export TRAIN_TP=${TRAIN_TP:-1}
export TRAIN_EP=${TRAIN_EP:-8}
export TRAIN_ETP=${TRAIN_ETP:-1}

# Rollout TP must be 1: verl's FP8 rollout uses 128x128 weight blocks, and vLLM requires
# each TP shard of the routed experts' intermediate size (1408) to be a multiple of 128.
# TP=2 gives 704 and fails at engine init; TP=1 keeps 1408 = 11 x 128.
export GEN_TP=${GEN_TP:-1}

export ACTOR_LR=${ACTOR_LR:-3e-6}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}
# Checkpoint every 20 steps; the base recipe keeps only the latest one (max_actor_ckpt_to_keep=1).
export SAVE_FREQ=${SAVE_FREQ:-20}
export VAL_BEFORE_TRAIN=True

export PROJECT_NAME=${PROJECT_NAME:-"router drift control"}
off_policy_mini_steps=${OFF_POLICY_MINI_STEPS:-2}
export EXPERIMENT_NAME_BASE=${EXPERIMENT_NAME_BASE:-moonlight-gsm8k-off-${off_policy_mini_steps}-top8RDC-TIS}

# METHOD_SCRIPT selects the arm (RDC+TIS by default); every arm sees the same overrides.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
method_script=${METHOD_SCRIPT:-"${script_dir}/off-policy-lengthbucket-RDC-TIS.sh"}

# Resolve launch-critical overrides in order, matching Hydra's last-value precedence.
# The default off-policy arm enables both RS switches; R3 requires them explicitly off.
model_path=${MODEL_PATH}
tokenizer_path=
replay_mode=disabled
mismatch_rs=True
current_aware_rs=True
for override in "$@"; do
    key=${override%%=*}
    key=${key#++}
    key=${key#+}
    value=${override#*=}
    # Hydra accepts quoted string values as well as shell-quoted arguments.
    value=${value#\"}; value=${value%\"}
    value=${value#\'}; value=${value%\'}
    case "${key}" in
        actor_rollout_ref.model.path) model_path=${value} ;;
        actor_rollout_ref.model.tokenizer_path) tokenizer_path=${value} ;;
        actor_rollout_ref.actor.megatron.router_replay.mode) replay_mode=${value} ;;
        router.enable_mismatch_rs) mismatch_rs=${value} ;;
        router.enable_current_aware_mismatch_rs) current_aware_rs=${value} ;;
    esac
done
r3_defaults=()
if [[ "${replay_mode}" == R3 ]]; then
    if [[ "${mismatch_rs,,}" != false || "${current_aware_rs,,}" != false ]]; then
        echo "ERROR: R3 requires router.enable_mismatch_rs=False and router.enable_current_aware_mismatch_rs=False; refusing to silently disable RS." >&2
        exit 2
    fi
    # R3 does not return actor routes. Explicit user metrics overrides still win.
    r3_defaults+=(router.enable_mismatch_metrics=False)
fi

# Isolate dynamic modules per invocation on each node, before the runner starts Ray.
# /tmp is node-local; preserve an explicitly supplied cache and never delete caches.
# This mitigates suspected concurrent cache loading, not a confirmed root-cause fix.
if [[ -z "${HF_MODULES_CACHE:-}" ]]; then
    HF_MODULES_CACHE=$(mktemp -d /tmp/moonlight-hf-modules.XXXXXXXX)
fi
export HF_MODULES_CACHE
mkdir -p -- "${HF_MODULES_CACHE}"
if [[ -z "${tokenizer_path}" || "${tokenizer_path,,}" == null ]]; then
    tokenizer_path=${model_path}
fi
# A single process primes and validates the exact tokenizer before any workers start.
python3 - "${tokenizer_path}" <<'PY'
import os
import sys
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    sys.argv[1], trust_remote_code=True, local_files_only=True
)
print(f"[Moonlight] tokenizer preflight OK: {type(tokenizer).__name__}; "
      f"HF_MODULES_CACHE={os.environ['HF_MODULES_CACHE']}", flush=True)
PY

# Moonlight ships a custom tiktoken tokenizer, so both loaders need trust_remote_code.
# mbridge's DeepseekV3 default moe_router_bias_update_rate=0.001 lets Megatron update
# expert_bias from expert load every step: a non-gradient routing shift outside RDC's
# drift model. Freeze it so routing only moves through gradient updates.
exec bash "${method_script}" \
    actor_rollout_ref.model.trust_remote_code=True \
    data.trust_remote_code=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_bias_update_rate=0.0 \
    "${r3_defaults[@]}" \
    "$@"
