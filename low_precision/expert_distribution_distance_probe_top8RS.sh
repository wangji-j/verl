#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export EXPERIMENT_NAME_BASE=${EXPERIMENT_NAME_BASE:-GRPO-DEEPSCALER-Qwen3-30B-A3B-base-MEGATRON-VLLM-FP8-expert-distance-probe-top8RS-TIS-C2-16K}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-50}
export SAVE_FREQ=${SAVE_FREQ:-5}
export VERL_REWARD_DEBUG_STEPS=${VERL_REWARD_DEBUG_STEPS:-0}
export VERL_REWARD_DEBUG_SAMPLES=${VERL_REWARD_DEBUG_SAMPLES:-0}

# Save compact per-response/layer expert histograms without the much larger
# token/logprob tensors. Candidate distances reuse the existing histogram pass.
export VERL_ROUTER_ANALYSIS_DUMP_MODE=${VERL_ROUTER_ANALYSIS_DUMP_MODE:-expert_counts}
export VERL_ROUTER_ANALYSIS_DUMP_EVERY_N=${VERL_ROUTER_ANALYSIS_DUMP_EVERY_N:-5}
export VERL_ROUTER_ANALYSIS_DUMP_STEPS=${VERL_ROUTER_ANALYSIS_DUMP_STEPS:-150}
export VERL_ROUTER_ANALYSIS_DUMP_TOPK_TOKENS=${VERL_ROUTER_ANALYSIS_DUMP_TOPK_TOKENS:-0}

exec bash "${SCRIPT_DIR}/run_grpo_qwen3_30b_a3b_expert_distribution_l1_top8RS.sh" \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_is_batch_normalize=false \
    algorithm.rollout_correction.rollout_rs=null \
    algorithm.rollout_correction.rollout_rs_threshold=null \
    "$@"
