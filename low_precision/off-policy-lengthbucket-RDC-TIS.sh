#!/usr/bin/env bash
set -euo pipefail

# Current-aware off-policy experiment built on the established
# length-bucket RDC + token-level TIS recipe. OFF_POLICY_MINI_STEPS controls
# how many non-overlapping PPO updates reuse one fixed rollout batch.
off_policy_mini_steps=${OFF_POLICY_MINI_STEPS:-2}
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-256}

if ! [[ "${off_policy_mini_steps}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: OFF_POLICY_MINI_STEPS must be a positive integer; got ${off_policy_mini_steps}." >&2
    exit 2
fi
if ! [[ "${train_prompt_bsz}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: TRAIN_PROMPT_BSZ must be a positive integer; got ${train_prompt_bsz}." >&2
    exit 2
fi
if (( train_prompt_bsz % off_policy_mini_steps != 0 )); then
    echo "ERROR: TRAIN_PROMPT_BSZ=${train_prompt_bsz} is not divisible by " \
        "OFF_POLICY_MINI_STEPS=${off_policy_mini_steps}." >&2
    exit 2
fi

expected_prompt_mini_bsz=$((train_prompt_bsz / off_policy_mini_steps))
if [[ -n "${TRAIN_PROMPT_MINI_BSZ:-}" ]] && ! [[ "${TRAIN_PROMPT_MINI_BSZ}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: TRAIN_PROMPT_MINI_BSZ must be a positive integer; got ${TRAIN_PROMPT_MINI_BSZ}." >&2
    exit 2
fi
if [[ -n "${TRAIN_PROMPT_MINI_BSZ:-}" ]] && (( TRAIN_PROMPT_MINI_BSZ != expected_prompt_mini_bsz )); then
    echo "ERROR: TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ} conflicts with " \
        "OFF_POLICY_MINI_STEPS=${off_policy_mini_steps}; expected ${expected_prompt_mini_bsz}." >&2
    exit 2
fi

run_timestamp=${RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}
experiment_name_base=${EXPERIMENT_NAME_BASE:-off-${off_policy_mini_steps}-top8RDC-TIS}

export RUN_TIMESTAMP=${run_timestamp}
export EXPERIMENT_NAME_BASE=${experiment_name_base}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-${experiment_name_base}-${run_timestamp}}
export TRAIN_PROMPT_BSZ=${train_prompt_bsz}
export TRAIN_PROMPT_MINI_BSZ=${expected_prompt_mini_bsz}
export ENABLE_ROUTER_MISMATCH_RS=True
export ROUTER_MISMATCH_RS_MODE=length_bucket_top_fraction
export ROUTER_MISMATCH_RS_FRACTION=0.08
export ROUTER_MISMATCH_ALIGNMENT_WARMUP_STEPS=1
export ENABLE_ROLLOUT_ROUTING_REPLAY=False

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "${script_dir}/run_grpo_qwen3_30b_a3b_expert_distribution_l1_lengthbucket_top8RS_TIS.sh" \
    router.enable_current_aware_mismatch_rs=True \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.megatron.router_replay.mode=disabled \
    actor_rollout_ref.rollout.enable_rollout_routing_replay=False \
    "$@"
