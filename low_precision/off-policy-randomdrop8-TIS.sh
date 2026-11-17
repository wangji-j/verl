#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# off-k ABLATION: TIS + uniform random 8% sequence drop. No RDC, no R3.
#
# Control arm for off-policy-lengthbucket-RDC-TIS.sh. That arm drops the highest
# router-drift 8% inside each response-length bucket, re-probed at theta_(j-1)
# before every mini-step (current-aware RDC). This arm drops a uniformly random
# 8%, no buckets, no drift signal. Everything else is held fixed, so the only
# difference is WHICH responses are removed.
#
#   RDC arm    > this arm  => the drift ranking is doing the work.
#   RDC arm   == this arm  => the gain was just training on 8% fewer sequences.
#
# Mirrors off-policy-lengthbucket-RDC-TIS.sh: a thin wrapper that derives the
# off-k batch split and delegates to the shared sync recipe, overriding only the
# entry point (dapo.main_dapo_random_drop) and the RDC/R3 switches.
#
# off-k staleness needs no RDC machinery. It comes from the batch split alone:
# ppo_mini_batch_size = TRAIN_PROMPT_BSZ / OFF_POLICY_MINI_STEPS, so the actor
# worker runs K sequential optimizer steps over one fixed rollout batch and
# mini-step j trains at theta_(j-1) against routes captured at theta_0. The mask
# is applied once per rollout batch before compute_advantage (the same point
# static RDC masks); every mini-step inherits it. See dapo/main_dapo_random_drop.py.
#
# Defaults to off-16 (OFF_POLICY_MINI_STEPS=16, MBS=16 prompts) at lr 3e-6.
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

random_drop_fraction=${RANDOM_DROP_FRACTION:-0.08}
random_drop_rounding=${RANDOM_DROP_ROUNDING:-ceil}
random_drop_count=${RANDOM_DROP_COUNT:-null}
random_drop_seed=${RANDOM_DROP_SEED:-1234}

run_timestamp=${RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}
experiment_name_base=${EXPERIMENT_NAME_BASE:-off-${off_policy_mini_steps}-randomdrop8-TIS}

export RUN_TIMESTAMP=${run_timestamp}
export EXPERIMENT_NAME_BASE=${experiment_name_base}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-${experiment_name_base}-${run_timestamp}}
export TRAIN_PROMPT_BSZ=${train_prompt_bsz}
export TRAIN_PROMPT_MINI_BSZ=${expected_prompt_mini_bsz}

# off-16 at 3e-6 = the sync recipe's default LR, i.e. the same LR the RDC arm
# runs at. Keeps the ablation controlled. Note this is a FIXED-LR comparison, not
# the PR2-aligned LR-scaled-with-kappa ladder -- label it as such in the paper.
export ACTOR_LR=${ACTOR_LR:-3e-6}

# --- RDC off, R3 off ---------------------------------------------------------
# enable_mismatch_metrics is left ON by the base recipe and is harmless (routes
# are compared and logged, nothing is masked), which gives a free drift readout
# to compare the random-dropped set against. Set ROUTER_DRIFT_METRICS=off below
# to save the routed_experts transfer instead.
export ENABLE_ROUTER_MISMATCH_RS=False
export ENABLE_ROLLOUT_ROUTING_REPLAY=False

router_metrics_overrides=()
if [[ "${ROUTER_DRIFT_METRICS:-on}" == "off" ]]; then
    router_metrics_overrides+=(router.enable_mismatch_metrics=False)
fi

echo "[OFF-RANDOMDROP] off_k=${off_policy_mini_steps} train_bsz=${train_prompt_bsz} mini_bsz=${expected_prompt_mini_bsz} lr=${ACTOR_LR}"
echo "[OFF-RANDOMDROP] random_drop: fraction=${random_drop_fraction} rounding=${random_drop_rounding} count=${random_drop_count} seed=${random_drop_seed}"
echo "[OFF-RANDOMDROP] RDC=OFF R3=OFF TIS=token(C=2) drift_metrics=${ROUTER_DRIFT_METRICS:-on}"
echo "[OFF-RANDOMDROP] experiment=${EXPERIMENT_NAME}"

# Swap the entry point so the trainer subclass is used. The base recipe reads
# ENTRY_POINT and defaults to dapo.main_dapo, so no other caller is affected.
export ENTRY_POINT=dapo.main_dapo_random_drop

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "${script_dir}/run_grpo_qwen3_30b_a3b_expert_distribution_l1_lengthbucket_top8RS_TIS.sh" \
    +random_drop.enable=True \
    +random_drop.fraction="${random_drop_fraction}" \
    +random_drop.rounding="${random_drop_rounding}" \
    +random_drop.count="${random_drop_count}" \
    +random_drop.seed="${random_drop_seed}" \
    router.enable_current_aware_mismatch_rs=False \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.megatron.router_replay.mode=disabled \
    actor_rollout_ref.rollout.enable_rollout_routing_replay=False \
    "${router_metrics_overrides[@]}" \
    "$@"
