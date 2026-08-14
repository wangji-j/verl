#!/usr/bin/env bash

export EXPERIMENT_NAME_BASE=${EXPERIMENT_NAME_BASE:-madz-top8RS-nocap-TIS-base}
export ROUTER_MISMATCH_RS_MODE=${ROUTER_MISMATCH_RS_MODE:-length_bucket_mad_zscore_top_fraction}
export ROUTER_MISMATCH_RS_FRACTION=${ROUTER_MISMATCH_RS_FRACTION:-0.08}
export ROUTER_MISMATCH_RS_MAD_EPSILON=${ROUTER_MISMATCH_RS_MAD_EPSILON:-3e-6}

exec "$(dirname "$0")/run_grpo_qwen3_30b_a3b_expert_distribution_l1_lengthbucket_top8RS_TIS.sh" \
    router.mismatch_rs_mad_epsilon="${ROUTER_MISMATCH_RS_MAD_EPSILON}" \
    "$@"
