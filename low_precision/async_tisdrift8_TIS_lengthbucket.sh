#!/usr/bin/env bash
set -xeuo pipefail

# =============================================================================
# Fully-async (disaggregated) ABLATION: TIS + probability-drift length-bucket 8%
# drop. Second CONTROL ARM for async_RDC_TIS_lengthbucket.sh.
#
# The RDC arm drops the highest-router-drift 8% inside each response-length
# bucket. This arm drops the highest ROLLOUT-vs-TRAIN PROBABILITY-drift 8% inside
# each bucket -- the log-ratio TIS itself truncates -- so the only difference vs
# the RDC arm is the ranking signal (router drift -> probability drift). The
# random-drop arm (async_randomdrop8_TIS.sh) drops a uniformly random 8% and is
# the third arm. Everything else -- model, data, TIS (token-level, C=2), LR,
# staleness, parallelism, node split -- is held fixed across all three.
#
#   RDC arm       : rank by router drift.
#   THIS arm      : rank by probability drift (A trunc_frac / B mean_abs_logratio).
#   random arm    : uniform random.
#
# Score switch (test A first, then B):
#   TIS_DRIFT_SCORE=trunc_frac         (A) fraction of a sequence's tokens whose
#                                          TIS weight is truncated (exp(logr) > C).
#   TIS_DRIFT_SCORE=mean_abs_logratio  (B) mean |log-ratio| over valid tokens.
#
# Async anchor: there is no per-mini-step loop, so the drift is scored against the
# train-side old_log_probs recompute already on the batch -- exactly the anchor
# the static async RDC arm compares against. The mask is applied inside
# _fit_compute_log_prob, where RDC applies its own, and TIS still corrects every
# surviving token (bypass_mode stays False -- the same 3-policy setup as RDC).
#
# RDC is fully off here: no router mismatch RS/metrics, rollout does not return
# routed_experts (so this arm is also cheaper per step). The drop hook lives in
#   verl/experimental/fully_async_policy/tis_drift_ablation.py
# wired in by the trainer subclass in
#   verl/experimental/fully_async_policy/fully_async_tis_drift_main.py
# Neither touches the shared async trainer, so concurrent runs are unaffected.
#
# Cluster: 4 nodes x 8 GPUs = 32 GPUs, split 3 nodes rollout + 1 node trainer.
# KEEP THE SAME SPLIT as the RDC / random arms (NNODES_ROLLOUT=3 NNODES_TRAIN=1).
# =============================================================================

ulimit -c 0 || true

# math-verify is required by the MATH-500 validation scorer on every node;
# verl's wrapper silently returns 0.0 if the import fails, so ensure it here.
python3 -c "import math_verify" 2>/dev/null || pip install --quiet --no-index \
    --find-links /inspire/hdd/project/qianghuaxuexi/public/wheels math-verify || \
    echo "WARNING: math-verify install failed; MATH-500 validation scores will be 0"

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://wandb1.sii.edu.cn/}
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_MODE=${WANDB_MODE:-online}
export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=${NVTE_FP8_BLOCK_SCALING_FP32_SCALES:-1}

run_timestamp=${RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}
project_name=${PROJECT_NAME:-router drift control}
exp_name_base=${EXPERIMENT_NAME_BASE:-async-tisdrift8-TIS}
exp_name=${EXPERIMENT_NAME:-${exp_name_base}_${run_timestamp}}
trainer_logger=${TRAINER_LOGGER:-'["console","tensorboard","wandb"]'}

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=${USE_KL_LOSS:-False}
kl_loss_coef=${KL_LOSS_COEF:-0.0}
kl_loss_type=${KL_LOSS_TYPE:-low_var_kl}

clip_ratio_low=0.2
clip_ratio_high=0.27

# --- Token-level truncated importance sampling (TIS C=2), decoupled mode ------
# Identical to the RDC arm. bypass_mode stays False so old_log_prob is recomputed
# as the proximal anchor; that keeps the policy setup (pi_rollout, pi_old, pi_theta)
# the same as the arm being compared against, and is also the anchor the
# probability-drift score is measured against.
rollout_is=token
rollout_is_threshold=2.0
rollout_is_batch_normalize=false
bypass_mode=False
rollout_rs=null
rollout_rs_threshold=null

# --- Probability-drift ablation knobs ----------------------------------------
# fraction/bucketing match the RDC arm's length_bucket_top_fraction; clip_c
# matches rollout_is_threshold so trunc_frac counts exactly TIS's truncations.
tis_drift_score=${TIS_DRIFT_SCORE:-trunc_frac}          # trunc_frac | mean_abs_logratio
tis_drift_clip_c=${TIS_DRIFT_CLIP_C:-2.0}
tis_drift_fraction=${TIS_DRIFT_FRACTION:-0.08}
tis_drift_length_bucket_edges=${TIS_DRIFT_LENGTH_BUCKET_EDGES:-"[2048,4096,8192,12288]"}
if [[ "${tis_drift_score}" != "trunc_frac" && "${tis_drift_score}" != "mean_abs_logratio" ]]; then
    echo "ERROR: TIS_DRIFT_SCORE must be 'trunc_frac' or 'mean_abs_logratio'; got ${tis_drift_score}." >&2
    exit 2
fi

max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-$((20 * 1024))}
val_max_response_length=${VAL_MAX_RESPONSE_LENGTH:-30720}
rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN:-32768}
enable_overlong_buffer=False
overlong_buffer_len=512
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

# --- Async batching -----------------------------------------------------------
n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}
train_prompt_bsz=0
gen_prompt_bsz=1
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-32}

# --- Async / staleness knobs (keep identical to the RDC arm) ------------------
staleness_threshold=${STALENESS_THRESHOLD:-0.5}
trigger_parameter_sync_step=${TRIGGER_PARAMETER_SYNC_STEP:-4}
require_batches=${REQUIRE_BATCHES:-1}
partial_rollout=${PARTIAL_ROLLOUT:-True}

total_training_steps=${TOTAL_TRAINING_STEPS:-500}
samples_per_train_step=$(( train_prompt_mini_bsz * require_batches * trigger_parameter_sync_step ))
total_rollout_steps=${TOTAL_ROLLOUT_STEPS:-$(( total_training_steps * samples_per_train_step ))}

WORKING_DIR=${WORKING_DIR:-"/inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence"}
RECIPE_DIR=${RECIPE_DIR:-"${WORKING_DIR}"}

RAY_DATA_HOME=${RAY_DATA_HOME:-"/inspire/hdd/project/qianghuaxuexi/public"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3-30B-A3B"}
CKPTS_DIR=${CKPTS_DIR:-"/inspire/hdd3/project/qianghuaxuexi/hujiarui-25046/ckpts/${project_name}/${exp_name}"}
export VERL_REWARD_DEBUG_DIR=${VERL_REWARD_DEBUG_DIR:-"${CKPTS_DIR}/reward_debug"}
export VERL_REWARD_DEBUG_STEPS=${VERL_REWARD_DEBUG_STEPS:-40}
export VERL_REWARD_DEBUG_SAMPLES=${VERL_REWARD_DEBUG_SAMPLES:-16}
export VERL_PERF_DEBUG_DIR=${VERL_PERF_DEBUG_DIR:-"${CKPTS_DIR}/perf_debug"}
DEEPSCALER_DIR=${DEEPSCALER_DIR:-"${RAY_DATA_HOME}/datasets/deepscaler"}
DAPO_MATH_17K_FILE=${DAPO_MATH_17K_FILE:-"${RAY_DATA_HOME}/datasets/DAPO-Math-17k/dapo-math-17k.parquet"}
TRAIN_FILE=${TRAIN_FILE:-"${DAPO_MATH_17K_FILE}"}
AIME24_25_FILE=${AIME24_25_FILE:-"${RAY_DATA_HOME}/datasets/aime_2024/aime24_aime25_x32.parquet"}
MATH500_FILE=${MATH500_FILE:-"${WORKING_DIR}/data/math500/test.parquet"}
MATH_BENCHMARK_FILE=${MATH_BENCHMARK_FILE:-"${WORKING_DIR}/data/math500/aime24_aime25_x32_math500.parquet"}
ZEBRALOGIC_FILE=${ZEBRALOGIC_FILE:-"${WORKING_DIR}/data/zebra_logic/test.parquet"}
FULL_BENCHMARK_FILE=${FULL_BENCHMARK_FILE:-"${WORKING_DIR}/data/zebra_logic/math_and_zebralogic.parquet"}
TEST_FILE=${TEST_FILE:-"${FULL_BENCHMARK_FILE}"}

# --- Node split: 4 nodes x 8 GPUs = 32 GPUs ----------------------------------
# KEEP MATCHED ACROSS ARMS (see async_randomdrop8_TIS.sh for the full rationale).
NNODES_ROLLOUT=${NNODES_ROLLOUT:-3}
NNODES_TRAIN=${NNODES_TRAIN:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
NNODES=${PET_NNODES:-${NNODES:-$((NNODES_ROLLOUT + NNODES_TRAIN))}}
NODE_RANK=${PET_NODE_RANK:-${NODE_RANK:-0}}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
RAY_MASTER_ADDR=${RAY_MASTER_ADDR:-${MASTER_ADDR}}
RAY_MASTER_PORT=${RAY_MASTER_PORT:-6379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}

temperature=1.0
top_p=1.0
top_k=-1
val_top_p=${VAL_TOP_P:-0.95}
val_top_k=${VAL_TOP_K:-20}

use_dynamic_bsz=False
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
actor_lr=${ACTOR_LR:-1e-6}
actor_lr_warmup_init=${ACTOR_LR_WARMUP_INIT:-1e-7}
offload=${OFFLOAD:-true}

train_tp=${TRAIN_TP:-4}
train_pp=${TRAIN_PP:-1}
train_ep=${TRAIN_EP:-4}
train_etp=${TRAIN_ETP:-2}

gen_tp=${GEN_TP:-2}
rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.85}
rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}
rollout_enforce_eager=${ROLLOUT_ENFORCE_EAGER:-True}

val_before_train=${VAL_BEFORE_TRAIN:-True}
test_freq=${TEST_FREQ:-5}
save_freq=${SAVE_FREQ:-30}
total_epochs=${TOTAL_EPOCHS:-100}

export PYTHONPATH="${WORKING_DIR}:${RECIPE_DIR}:${PYTHONPATH:-}"
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
export VLLM_CONFIGURE_LOGGING=${VLLM_CONFIGURE_LOGGING:-1}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}
export VLLM_USE_DEEP_GEMM_E8M0=${VLLM_USE_DEEP_GEMM_E8M0:-0}
export VERL_DISABLE_BROKEN_DEEP_EP=${VERL_DISABLE_BROKEN_DEEP_EP:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}

cd "${WORKING_DIR}"
mkdir -p "${CKPTS_DIR}" "${CKPTS_DIR}/run_logs"

TRAINING_CMD=(
    python3 -m verl.experimental.fully_async_policy.fully_async_tis_drift_main
    --config-path=config
    --config-name=fully_async_ppo_megatron_trainer.yaml
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.prompt_key=prompt
    data.truncation=left
    data.return_raw_chat=True
    +data.apply_chat_template_kwargs.enable_thinking=false
    data.filter_overlong_prompts=True
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.train_batch_size=${train_prompt_bsz}
    data.gen_batch_size=${gen_prompt_bsz}
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=${use_kl_in_reward}
    algorithm.kl_ctrl.kl_coef=${kl_coef}
    algorithm.rollout_correction.bypass_mode=${bypass_mode}
    algorithm.rollout_correction.rollout_is=${rollout_is}
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold}
    algorithm.rollout_correction.rollout_is_batch_normalize=${rollout_is_batch_normalize}
    algorithm.rollout_correction.rollout_rs=${rollout_rs}
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold}
    # --- Probability-drift ablation (new top-level group, hence the + prefixes) -
    +tis_drift.enable=True
    +tis_drift.score=${tis_drift_score}
    +tis_drift.clip_c=${tis_drift_clip_c}
    +tis_drift.fraction=${tis_drift_fraction}
    +tis_drift.length_bucket_edges=${tis_drift_length_bucket_edges}
    # --- RDC fully OFF: this is the router-signal-free control arm ------------
    router.enable_mismatch_metrics=False
    router.enable_mismatch_rs=False
    router.enable_current_aware_mismatch_rs=False
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type}
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.optim.min_lr=${actor_lr}
    actor_rollout_ref.actor.optim.lr_warmup_steps=10
    actor_rollout_ref.actor.optim.lr_warmup_init=${actor_lr_warmup_init}
    actor_rollout_ref.actor.optim.lr_decay_style=constant
    actor_rollout_ref.actor.optim.lr_decay_steps=${total_rollout_steps}
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    actor_rollout_ref.actor.megatron.param_offload=${offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload}
    actor_rollout_ref.actor.megatron.grad_offload=${offload}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${train_etp}
    actor_rollout_ref.actor.megatron.use_remove_padding=True
    actor_rollout_ref.actor.megatron.dtype=bfloat16
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.expert_parallel_size=1
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens}
    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_p=${top_p}
    actor_rollout_ref.rollout.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.max_response_length=${val_max_response_length}
    actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len}
    +actor_rollout_ref.rollout.quantization=fp8
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.enable_return_routed_experts=False
    actor_rollout_ref.rollout.enforce_eager=${rollout_enforce_eager}
    actor_rollout_ref.hybrid_engine=False
    actor_rollout_ref.ref.megatron.param_offload=${offload}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp}
    actor_rollout_ref.ref.megatron.use_remove_padding=True
    actor_rollout_ref.ref.megatron.dtype=bfloat16
    reward.reward_manager.name=dapo
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer}
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len}
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor}
    +reward.reward_kwargs.overlong_buffer_cfg.log=False
    +reward.reward_kwargs.max_resp_len=${max_response_length}
    trainer.logger=${trainer_logger}
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES_TRAIN}
    trainer.val_before_train=${val_before_train}
    trainer.test_freq=${test_freq}
    trainer.save_freq=${save_freq}
    trainer.total_epochs=${total_epochs}
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.resume_mode=auto
    trainer.log_val_generations=1
    trainer.max_actor_ckpt_to_keep=1
    trainer.max_critic_ckpt_to_keep=1
    rollout.nnodes=${NNODES_ROLLOUT}
    rollout.n_gpus_per_node=${NGPUS_PER_NODE}
    rollout.total_rollout_steps=${total_rollout_steps}
    async_training.staleness_threshold=${staleness_threshold}
    async_training.trigger_parameter_sync_step=${trigger_parameter_sync_step}
    async_training.require_batches=${require_batches}
    async_training.partial_rollout=${partial_rollout}
    +ray_kwargs.ray_init.address=auto
)

start_ray_cluster() {
    local ray_head_wait_timeout=${RAY_HEAD_WAIT_TIMEOUT:-600}

    if [ -n "${INTERFACE_NAME:-}" ]; then
        export RAY_RAYLET_NODE_MANAGER_CONFIG_NIC_NAME=${INTERFACE_NAME}
        export RAY_GCS_SERVER_CONFIG_NIC_NAME=${INTERFACE_NAME}
    fi
    export RAY_RUNTIME_ENV_AGENT_CREATION_TIMEOUT_S=${RAY_RUNTIME_ENV_AGENT_CREATION_TIMEOUT_S:-1200}
    export RAY_GCS_RPC_CLIENT_CONNECT_TIMEOUT_S=${RAY_GCS_RPC_CLIENT_CONNECT_TIMEOUT_S:-120}

    local ray_start_common_opts=(
        --num-gpus "${NGPUS_PER_NODE}"
        --temp-dir "${RAY_TMPDIR}"
    )

    if [ -n "${RAY_OBJECT_STORE_MEMORY:-}" ]; then
        ray_start_common_opts+=(--object-store-memory "${RAY_OBJECT_STORE_MEMORY}")
    fi
    if [ -n "${RAY_MEMORY:-}" ]; then
        ray_start_common_opts+=(--memory "${RAY_MEMORY}")
    fi

    if [ "${NNODES}" -gt 1 ]; then
        if [ "${NODE_RANK}" = "0" ]; then
            export RAY_ADDRESS="${RAY_MASTER_ADDR}:${RAY_MASTER_PORT}"
            echo "INFO: Starting Ray head on $(hostname), address=${RAY_ADDRESS}"
            ray start \
                --head \
                --port="${RAY_MASTER_PORT}" \
                --dashboard-port="${RAY_DASHBOARD_PORT}" \
                "${ray_start_common_opts[@]}" \
                --system-config='{"gcs_server_request_timeout_seconds": 60, "gcs_rpc_server_reconnect_timeout_s": 60}'

            local start_time
            start_time=$(date +%s)
            while ! ray health-check --address "${RAY_ADDRESS}" >/dev/null 2>&1; do
                if [ "$(( $(date +%s) - start_time ))" -ge "${ray_head_wait_timeout}" ]; then
                    echo "ERROR: Timed out waiting for Ray head at ${RAY_ADDRESS}" >&2
                    ray stop --force >/dev/null 2>&1 || true
                    exit 1
                fi
                echo "INFO: Ray head not healthy yet, retrying in 5s..."
                sleep 5
            done
            echo "INFO: Ray head is healthy."
        else
            local head_node_address="${RAY_MASTER_ADDR}:${RAY_MASTER_PORT}"
            export RAY_ADDRESS="${head_node_address}"
            echo "INFO: Worker rank ${NODE_RANK} waiting for Ray head at ${head_node_address}"
            local start_time
            start_time=$(date +%s)
            while ! ray health-check --address "${head_node_address}" >/dev/null 2>&1; do
                if [ "$(( $(date +%s) - start_time ))" -ge "${ray_head_wait_timeout}" ]; then
                    echo "ERROR: Timed out waiting for Ray head at ${head_node_address}" >&2
                    exit 1
                fi
                echo "INFO: Ray head not healthy yet, retrying in 5s..."
                sleep 5
            done
            echo "INFO: Joining Ray cluster at ${head_node_address}"
            ray start --address="${head_node_address}" "${ray_start_common_opts[@]}"
        fi
    else
        export RAY_ADDRESS="127.0.0.1:${RAY_MASTER_PORT}"
        echo "INFO: Starting single-node Ray head on $(hostname), address=${RAY_ADDRESS}"
        ray start \
            --head \
            --port="${RAY_MASTER_PORT}" \
            --dashboard-port="${RAY_DASHBOARD_PORT}" \
            "${ray_start_common_opts[@]}" \
            --system-config='{"gcs_server_request_timeout_seconds": 60, "gcs_rpc_server_reconnect_timeout_s": 60}'
    fi
}

wait_for_ray_nodes() {
    if [ "${NNODES}" -le 1 ] || [ "${NODE_RANK}" != "0" ]; then
        return 0
    fi

    echo "INFO: Waiting for all ${NNODES} Ray nodes to join..."
    local timeout=${RAY_NODES_WAIT_TIMEOUT:-600}
    local start_time
    start_time=$(date +%s)
    while true; do
        if [ "$(( $(date +%s) - start_time ))" -ge "${timeout}" ]; then
            echo "ERROR: Timeout waiting for Ray nodes." >&2
            ray status || true
            exit 1
        fi

        local ready_nodes
        ready_nodes=$(python3 - <<'PY' 2>/dev/null || true
import ray

try:
    ray.init(address="auto", ignore_reinit_error=True)
    print(len([node for node in ray.nodes() if node.get("Alive")]))
except Exception:
    print(0)
PY
)
        ready_nodes=${ready_nodes:-0}
        if [ "${ready_nodes}" -ge "${NNODES}" ]; then
            echo "INFO: All ${NNODES} Ray nodes have joined."
            break
        fi
        echo "INFO: Waiting for Ray nodes... (${ready_nodes}/${NNODES})"
        sleep 5
    done
}

main() {
    export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/verl_ray_${NODE_RANK}_${run_timestamp}}
    mkdir -p "${RAY_TMPDIR}"

    export GLOO_SOCKET_TIMEOUT=${GLOO_SOCKET_TIMEOUT:-600}
    export GLOO_TCP_TIMEOUT=${GLOO_TCP_TIMEOUT:-600}

    echo "[ASYNC-TISDRIFT] model=${MODEL_PATH}"
    echo "[ASYNC-TISDRIFT] train=${TRAIN_FILE}"
    echo "[ASYNC-TISDRIFT] val=${TEST_FILE}"
    echo "[ASYNC-TISDRIFT] project=${project_name} experiment=${exp_name}"
    echo "[ASYNC-TISDRIFT] nnodes=${NNODES} (rollout=${NNODES_ROLLOUT} train=${NNODES_TRAIN}) node_rank=${NODE_RANK} gpus_per_node=${NGPUS_PER_NODE}"
    echo "[ASYNC-TISDRIFT] RDC=OFF  TIS=${rollout_is}(C=${rollout_is_threshold})"
    echo "[ASYNC-TISDRIFT] tis_drift: score=${tis_drift_score} clip_c=${tis_drift_clip_c} fraction=${tis_drift_fraction} buckets=${tis_drift_length_bucket_edges}"
    echo "[ASYNC-TISDRIFT] staleness=${staleness_threshold} trigger_sync=${trigger_parameter_sync_step}"
    echo "INFO: Ray temp dir: ${RAY_TMPDIR}"

    ray stop --force >/dev/null 2>&1 || true
    start_ray_cluster
    wait_for_ray_nodes

    if [ "${NODE_RANK}" = "0" ]; then
        echo "INFO: Rank 0 launching fully-async probability-drift + TIS ablation with Ray address ${RAY_ADDRESS}"
        "${TRAINING_CMD[@]}" "$@"
        echo "INFO: Rank 0 training finished."
        sleep 30
        ray stop --force >/dev/null 2>&1 || true
    elif [ "${NNODES}" -gt 1 ]; then
        local head_node_address="${RAY_MASTER_ADDR}:${RAY_MASTER_PORT}"
        echo "INFO: Worker rank ${NODE_RANK} joined Ray. Monitoring head ${head_node_address}."
        while ray health-check --address "${head_node_address}" >/dev/null 2>&1; do
            sleep 15
        done
        echo "INFO: Ray head is down; worker rank ${NODE_RANK} exiting."
    fi
}

main "$@"
