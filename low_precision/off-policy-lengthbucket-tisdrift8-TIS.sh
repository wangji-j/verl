#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Off-policy PROBABILITY-DRIFT ablation: second control arm for RDC.
#
# Same off-policy length-bucket recipe as off-policy-lengthbucket-RDC-TIS.sh
# (OFF_POLICY_MINI_STEPS non-overlapping PPO updates reuse one rollout batch),
# and the *same* current-aware, per-mini-step, length-bucket top-8% drop -- but
# the ranking signal is the rollout-vs-current PROBABILITY drift (the log-ratio
# TIS truncates) instead of RDC's router drift. TIS still corrects every
# surviving token; the drop only zeroes response_mask rows.
#
#   RDC arm       : rank by router (expert-routing) drift.
#   THIS arm      : rank by probability drift.
#   random arm    : no ranking (uniform random drop).
#
# Score is switchable (test A first, then B):
#   TIS_DRIFT_SCORE=trunc_frac         (A) fraction of a sequence's tokens whose
#                                          TIS weight is truncated (exp(logr) > C).
#   TIS_DRIFT_SCORE=mean_abs_logratio  (B) mean |log-ratio| over valid tokens.
#
# RDC and R3 are OFF (no router routes returned; router.enable_mismatch_metrics
# is forced False, overriding the base recipe default). LR stays 3e-6 and every
# batch/parallelism knob matches the RDC arm, so the ONLY difference vs the RDC
# arm is the ranking signal.
#
# off-16: OFF_POLICY_MINI_STEPS=16 -> TRAIN_PROMPT_MINI_BSZ=16 (with the default
# TRAIN_PROMPT_BSZ=256). Override OFF_POLICY_MINI_STEPS for other off-k.
# =============================================================================

off_policy_mini_steps=${OFF_POLICY_MINI_STEPS:-16}
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

# --- Probability-drift ablation knobs ----------------------------------------
tis_drift_score=${TIS_DRIFT_SCORE:-trunc_frac}          # trunc_frac | mean_abs_logratio
tis_drift_clip_c=${TIS_DRIFT_CLIP_C:-2.0}               # matches rollout_is_threshold
tis_drift_fraction=${TIS_DRIFT_FRACTION:-0.08}
tis_drift_length_bucket_edges=${TIS_DRIFT_LENGTH_BUCKET_EDGES:-"[2048,4096,8192,12288]"}
if [[ "${tis_drift_score}" != "trunc_frac" && "${tis_drift_score}" != "mean_abs_logratio" ]]; then
    echo "ERROR: TIS_DRIFT_SCORE must be 'trunc_frac' or 'mean_abs_logratio'; got ${tis_drift_score}." >&2
    exit 2
fi

run_timestamp=${RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}
experiment_name_base=${EXPERIMENT_NAME_BASE:-off-${off_policy_mini_steps}-tisdrift8-${tis_drift_score}-TIS}

export RUN_TIMESTAMP=${run_timestamp}
export EXPERIMENT_NAME_BASE=${experiment_name_base}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-${experiment_name_base}-${run_timestamp}}
export TRAIN_PROMPT_BSZ=${train_prompt_bsz}
export TRAIN_PROMPT_MINI_BSZ=${expected_prompt_mini_bsz}
# RDC fully off. R3 off.
export ENABLE_ROUTER_MISMATCH_RS=False
export ENABLE_ROLLOUT_ROUTING_REPLAY=False

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "${script_dir}/run_grpo_qwen3_30b_a3b_expert_distribution_l1_lengthbucket_top8RS_TIS.sh" \
    router.enable_mismatch_metrics=False \
    router.enable_mismatch_rs=False \
    router.enable_current_aware_mismatch_rs=False \
    actor_rollout_ref.rollout.enable_return_routed_experts=False \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.megatron.router_replay.mode=disabled \
    actor_rollout_ref.rollout.enable_rollout_routing_replay=False \
    +tis_drift.enable=True \
    +tis_drift.off_policy_mini_step=True \
    +tis_drift.score="${tis_drift_score}" \
    +tis_drift.clip_c="${tis_drift_clip_c}" \
    +tis_drift.fraction="${tis_drift_fraction}" \
    +tis_drift.length_bucket_edges="${tis_drift_length_bucket_edges}" \
    "$@"
