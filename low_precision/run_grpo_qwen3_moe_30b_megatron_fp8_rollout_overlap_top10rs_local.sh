#!/usr/bin/env bash
set -xeuo pipefail

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
export WANDB_API_KEY=${WANDB_API_KEY:-local-6a4cc4c8b917355ce21530f9c9be52014cc55ee2}
export WANDB_MODE=${WANDB_MODE:-online}
export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=${NVTE_FP8_BLOCK_SCALING_FP32_SCALES:-1}

run_timestamp=${RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}
project_name=${PROJECT_NAME:-router drift control}
exp_name_base=${EXPERIMENT_NAME_BASE:-GRPO-DEEPSCALER-Qwen3-30B-A3B-base-MEGATRON-VLLM-FP8-overlap-mismatch-threshold0.045RS-16K}
exp_name=${EXPERIMENT_NAME:-${exp_name_base}_${run_timestamp}}
trainer_logger=${TRAINER_LOGGER:-'["console","tensorboard","wandb"]'}

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=${KL_LOSS_COEF:-0.001}
kl_loss_type=${KL_LOSS_TYPE:-low_var_kl}

clip_ratio_low=0.2
clip_ratio_high=0.27

rollout_is=null
rollout_is_threshold=null
rollout_is_batch_normalize=false
rollout_rs=null
rollout_rs_threshold=null

enable_rollout_routing_replay=${ENABLE_ROLLOUT_ROUTING_REPLAY:-False}
enable_router_mismatch_rs=${ENABLE_ROUTER_MISMATCH_RS:-True}
router_mismatch_rs_threshold=${ROUTER_MISMATCH_RS_THRESHOLD:-0.045}
router_mismatch_rs_mode=${ROUTER_MISMATCH_RS_MODE:-threshold}
router_mismatch_rs_fraction=${ROUTER_MISMATCH_RS_FRACTION:-0.0}
router_mismatch_metric_mode=${ROUTER_MISMATCH_METRIC_MODE:-overlap_fraction}
router_mismatch_alignment_warmup_steps=${ROUTER_MISMATCH_ALIGNMENT_WARMUP_STEPS:-1}

max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-$((16 * 1024))}
enable_overlong_buffer=False
overlong_buffer_len=512
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=${MAX_NUM_GEN_BATCHES:-0}
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-256}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-256}
gen_prompt_bsz=${GEN_PROMPT_BSZ:-256}

WORKING_DIR=${WORKING_DIR:-"/inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence"}
RECIPE_DIR=${RECIPE_DIR:-"${WORKING_DIR}"}

RAY_DATA_HOME=${RAY_DATA_HOME:-"/inspire/hdd/project/qianghuaxuexi/public"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3-30B-A3B"}
CKPTS_DIR=${CKPTS_DIR:-"/inspire/hdd3/project/qianghuaxuexi/hujiarui-25046/ckpts/${project_name}/${exp_name}"}
export VERL_REWARD_DEBUG_DIR=${VERL_REWARD_DEBUG_DIR:-"${CKPTS_DIR}/reward_debug"}
export VERL_REWARD_DEBUG_STEPS=${VERL_REWARD_DEBUG_STEPS:-40}
export VERL_REWARD_DEBUG_SAMPLES=${VERL_REWARD_DEBUG_SAMPLES:-16}
export VERL_PERF_DEBUG_DIR=${VERL_PERF_DEBUG_DIR:-"${CKPTS_DIR}/perf_debug"}
export VERL_ROUTER_ANALYSIS_DUMP_DIR=${VERL_ROUTER_ANALYSIS_DUMP_DIR:-"${CKPTS_DIR}/router_analysis_dump"}
export VERL_ROUTER_ANALYSIS_DUMP_MODE=${VERL_ROUTER_ANALYSIS_DUMP_MODE:-tokens}
export VERL_ROUTER_ANALYSIS_DUMP_EVERY_N=${VERL_ROUTER_ANALYSIS_DUMP_EVERY_N:-5}
export VERL_ROUTER_ANALYSIS_DUMP_STEPS=${VERL_ROUTER_ANALYSIS_DUMP_STEPS:-150}
export VERL_ROUTER_ANALYSIS_DUMP_SAMPLES=${VERL_ROUTER_ANALYSIS_DUMP_SAMPLES:-16}
export VERL_ROUTER_ANALYSIS_DUMP_FLOAT_DTYPE=${VERL_ROUTER_ANALYSIS_DUMP_FLOAT_DTYPE:-float16}
export VERL_ROUTER_ANALYSIS_DUMP_TOPK_TOKENS=${VERL_ROUTER_ANALYSIS_DUMP_TOPK_TOKENS:-32}
GSM8K_DIR=${GSM8K_DIR:-"${RAY_DATA_HOME}/datasets/gsm8k"}
DEEPSCALER_DIR=${DEEPSCALER_DIR:-"${RAY_DATA_HOME}/datasets/deepscaler"}
TRAIN_FILE=${TRAIN_FILE:-"${DEEPSCALER_DIR}/train.parquet"}
AIME24_FILE=${AIME24_FILE:-"${RAY_DATA_HOME}/datasets/aime_2024/test.parquet"}
AIME24_25_FILE=${AIME24_25_FILE:-"${RAY_DATA_HOME}/datasets/aime_2024/aime24_aime25_x32.parquet"}
MATH500_FILE=${MATH500_FILE:-"${WORKING_DIR}/data/math500/test.parquet"}
MATH_BENCHMARK_FILE=${MATH_BENCHMARK_FILE:-"${WORKING_DIR}/data/math500/aime24_aime25_x32_math500.parquet"}
ZEBRALOGIC_FILE=${ZEBRALOGIC_FILE:-"${WORKING_DIR}/data/zebra_logic/test.parquet"}
FULL_BENCHMARK_FILE=${FULL_BENCHMARK_FILE:-"${WORKING_DIR}/data/zebra_logic/math_and_zebralogic.parquet"}
TEST_FILE=${TEST_FILE:-"${FULL_BENCHMARK_FILE}"}

NNODES=${PET_NNODES:-${NNODES:-2}}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
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

use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
actor_lr=${ACTOR_LR:-3e-6}
offload=${OFFLOAD:-true}

# Engine-only replacement for the original FSDP settings.
train_tp=${TRAIN_TP:-4}
train_pp=${TRAIN_PP:-1}
train_ep=${TRAIN_EP:-4}
train_etp=${TRAIN_ETP:-2}

gen_tp=${GEN_TP:-2}
rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.85}
rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}
rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-128}
rollout_enforce_eager=${ROLLOUT_ENFORCE_EAGER:-True}
rollout_compilation_mode=${ROLLOUT_COMPILATION_MODE:-NONE}
rollout_cudagraph_mode=${ROLLOUT_CUDAGRAPH_MODE:-NONE}
val_before_train=${VAL_BEFORE_TRAIN:-True}
test_freq=${TEST_FREQ:-10}
save_freq=${SAVE_FREQ:-50}
total_training_steps=${TOTAL_TRAINING_STEPS:-500}

export PYTHONPATH="${WORKING_DIR}:${RECIPE_DIR}:${PYTHONPATH:-}"
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
export VLLM_CONFIGURE_LOGGING=${VLLM_CONFIGURE_LOGGING:-1}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
# vLLM DeepGEMM FP8 MoE can crash during the V1 profiling/compile path with
# "Cannot access data pointer of Tensor that doesn't have storage". Keep FP8
# rollout enabled, but default to the safer vLLM MoE backend for this recipe.
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}
export VLLM_USE_DEEP_GEMM_E8M0=${VLLM_USE_DEEP_GEMM_E8M0:-0}
export VERL_DISABLE_BROKEN_DEEP_EP=${VERL_DISABLE_BROKEN_DEEP_EP:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}

cd "${WORKING_DIR}"
mkdir -p "${CKPTS_DIR}" "${CKPTS_DIR}/run_logs"

TRAINING_CMD=(
    python3 -m dapo.main_dapo
    --config-name=dapo_megatron_trainer
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
    actor_rollout_ref.nccl_timeout=1800
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=${use_kl_in_reward}
    algorithm.kl_ctrl.kl_coef=${kl_coef}
    algorithm.filter_groups.enable=${enable_filter_groups}
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches}
    algorithm.filter_groups.metric=${filter_groups_metric}
    algorithm.rollout_correction.rollout_is=${rollout_is}
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold}
    algorithm.rollout_correction.rollout_is_batch_normalize=${rollout_is_batch_normalize}
    algorithm.rollout_correction.rollout_rs=${rollout_rs}
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold}
    router.enable_mismatch_metrics=True
    router.enable_mismatch_rs=${enable_router_mismatch_rs}
    router.mismatch_rs_threshold=${router_mismatch_rs_threshold}
    router.mismatch_rs_mode=${router_mismatch_rs_mode}
    router.mismatch_rs_fraction=${router_mismatch_rs_fraction}
    router.mismatch_metric_mode=${router_mismatch_metric_mode}
    router.mismatch_alignment_warmup_steps=${router_mismatch_alignment_warmup_steps}
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
    actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs}
    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_p=${top_p}
    actor_rollout_ref.rollout.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    +actor_rollout_ref.rollout.quantization=fp8
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.enable_rollout_routing_replay=${enable_rollout_routing_replay}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.mode=${rollout_compilation_mode}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=${rollout_cudagraph_mode}
    actor_rollout_ref.ref.megatron.param_offload=${offload}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp}
    actor_rollout_ref.ref.megatron.use_remove_padding=True
    actor_rollout_ref.ref.megatron.dtype=bfloat16
    reward.reward_manager.name=dapo
    reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer}
    reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len}
    reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor}
    reward.reward_kwargs.overlong_buffer_cfg.log=False
    reward.reward_kwargs.max_resp_len=${max_response_length}
    trainer.logger=${trainer_logger}
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=${val_before_train}
    trainer.test_freq=${test_freq}
    trainer.save_freq=${save_freq}
    trainer.total_epochs=100
    +trainer.log_epoch_number=True
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.resume_mode=auto
    trainer.log_val_generations=1
    trainer.total_training_steps=${total_training_steps}
    trainer.max_actor_ckpt_to_keep=1
    trainer.max_critic_ckpt_to_keep=1
    +trainer.use_legacy_worker_impl=disable
    actor_rollout_ref.rollout.enforce_eager=${rollout_enforce_eager}
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

    echo "[GRPO-FP8-BASELINE] model=${MODEL_PATH}"
    echo "[GRPO-FP8-BASELINE] train=${TRAIN_FILE}"
    echo "[GRPO-FP8-BASELINE] val=${TEST_FILE}"
    echo "[GRPO-FP8-BASELINE] project=${project_name} experiment=${exp_name}"
    echo "[GRPO-FP8-BASELINE] nnodes=${NNODES} node_rank=${NODE_RANK} gpus_per_node=${NGPUS_PER_NODE}"
    echo "[GRPO-FP8-BASELINE] rollout_routing_replay=${enable_rollout_routing_replay}"
    echo "INFO: Ray temp dir: ${RAY_TMPDIR}"

    ray stop --force >/dev/null 2>&1 || true
    start_ray_cluster
    wait_for_ray_nodes

    if [ "${NODE_RANK}" = "0" ]; then
        echo "INFO: Rank 0 launching GRPO Megatron FP8 rollout baseline training with Ray address ${RAY_ADDRESS}"
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
