#!/usr/bin/env bash
# TIS + overlap_fraction metric + GLOBAL top-8% RS (no length bucketing).
# Purpose: dump token-level flip data (token_mismatch) for offline evaluation of
# length structure, split-half noise, IACT, and EB-shrinkage design on the
# BASE model (all prior overlap dumps were Instruct-2507).
# Offline evaluator: experiment_analysis/eval_overlap_dump.py

export EXPERIMENT_NAME_BASE=${EXPERIMENT_NAME_BASE:-overlap-global-top8RS-TIS-base}
export ROUTER_MISMATCH_METRIC_MODE=${ROUTER_MISMATCH_METRIC_MODE:-overlap_fraction}
export ROUTER_MISMATCH_RS_MODE=${ROUTER_MISMATCH_RS_MODE:-top_fraction}
export ROUTER_MISMATCH_RS_FRACTION=${ROUTER_MISMATCH_RS_FRACTION:-0.08}
# Dynamic Sampling OFF for this dump run: keeps every step a plain 256x8 batch
# (fixed wall-clock, no resampling), so dumps cover the natural length/difficulty
# distribution rather than the filter-survivor distribution.
export ENABLE_FILTER_GROUPS=${ENABLE_FILTER_GROUPS:-False}
# dump every step for the analysis window; token mode already stores
# token_mismatch (per-token flip rate) needed by the evaluator.
export VERL_ROUTER_ANALYSIS_DUMP_EVERY_N=${VERL_ROUTER_ANALYSIS_DUMP_EVERY_N:-1}
export VERL_ROUTER_ANALYSIS_DUMP_MODE=${VERL_ROUTER_ANALYSIS_DUMP_MODE:-tokens}

exec "$(dirname "$0")/run_grpo_qwen3_30b_a3b_expert_distribution_l1_lengthbucket_top8RS_TIS.sh" "$@"
