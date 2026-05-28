#!/usr/bin/env bash
# GRPO | Qwen3-30B-A3B (MoE) | Megatron BF16 training | SGLang FP8 rollout
# Synchronous trainer path for validating rollout-only FP8 on GSM8K.

set -xeuo pipefail

# SGLang native scheduler aborts can emit multi-GB core dumps into the working
# directory. Keep failed smoke runs from filling the shared project filesystem.
ulimit -c 0 || true

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-/tmp/flashinfer-workspace}
export SGLANG_IS_FLASHINFER_AVAILABLE=${SGLANG_IS_FLASHINFER_AVAILABLE:-false}
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=${SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK:-True}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://wandb1.sii.edu.cn/}
export WANDB_MODE=${WANDB_MODE:-online}

########################### user-adjustable ###########################
MODEL_PATH=${MODEL_PATH:-/inspire/hdd/project/qianghuaxuexi/public/models/Qwen3-30B-A3B}
TRAIN_FILE=${TRAIN_FILE:-/inspire/hdd/project/qianghuaxuexi/public/datasets/gsm8k/train.parquet}
TEST_FILE=${TEST_FILE:-/inspire/hdd/project/qianghuaxuexi/public/datasets/gsm8k/test.parquet}
CKPTS_DIR=${CKPTS_DIR:-/inspire/hdd/global_user/hujiarui-25046/verl_data/ckpts/qwen3_30b_a3b_gsm8k_sglang_fp8_sync}

NNODES=${PET_NNODES:-${NNODES:-2}}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
NODE_RANK=${PET_NODE_RANK:-${NODE_RANK:-0}}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
RAY_MASTER_ADDR=${RAY_MASTER_ADDR:-${MASTER_ADDR}}
RAY_MASTER_PORT=${RAY_MASTER_PORT:-6379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}

train_batch_size=${TRAIN_BATCH_SIZE:-16}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-1024}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-3072}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}
actor_clip_ratio_low=${ACTOR_CLIP_RATIO_LOW:-0.2}
actor_clip_ratio_high=${ACTOR_CLIP_RATIO_HIGH:-0.28}
actor_clip_ratio_c=${ACTOR_CLIP_RATIO_C:-10.0}
actor_ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}

actor_tp=${ACTOR_TP:-1}
actor_pp=${ACTOR_PP:-1}
actor_vpp=${ACTOR_VPP:-null}
actor_ep=${ACTOR_EP:-4}
actor_etp=${ACTOR_ETP:-1}
actor_cp=${ACTOR_CP:-1}

ref_tp=${REF_TP:-${actor_tp}}
ref_pp=${REF_PP:-${actor_pp}}
ref_vpp=${REF_VPP:-${actor_vpp}}
ref_ep=${REF_EP:-${actor_ep}}
ref_etp=${REF_ETP:-${actor_etp}}
ref_cp=${REF_CP:-${actor_cp}}

all_offload=${ALL_OFFLOAD:-True}

rollout_tp=${ROLLOUT_TP:-8}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.45}
rollout_n=${ROLLOUT_N:-1}
rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-4}
rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-3072}
rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN:-3072}
rollout_temperature=${ROLLOUT_TEMPERATURE:-1.0}
rollout_top_p=${ROLLOUT_TOP_P:-1.0}
rollout_top_k=${ROLLOUT_TOP_K:--1}
rollout_quantization=${ROLLOUT_QUANTIZATION:-fp8}
sglang_fp8_gemm_runner_backend=${SGLANG_FP8_GEMM_RUNNER_BACKEND:-triton}
sglang_log_level=${SGLANG_LOG_LEVEL:-info}

ref_log_prob_max_token_len_per_gpu=${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-3072}
ref_log_prob_micro_batch_size_per_gpu=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
rollout_log_prob_max_token_len_per_gpu=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-3072}
rollout_log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}

val_do_sample=${VAL_DO_SAMPLE:-True}
val_temperature=${VAL_TEMPERATURE:-1.0}
val_top_p=${VAL_TOP_P:-0.7}
val_top_k=${VAL_TOP_K:--1}
val_n=${VAL_N:-1}
log_val_generations=${LOG_VAL_GENERATIONS:-10}

total_epochs=${TOTAL_EPOCHS:-1}
total_training_steps=${TOTAL_TRAINING_STEPS:-4}
save_freq=${SAVE_FREQ:--1}
test_freq=${TEST_FREQ:--1}

project_name=${PROJECT_NAME:-GRPO-Qwen3-30B-A3B-GSM8K-sglang-fp8-sync}
experiment_name=${EXPERIMENT_NAME:-qwen3_30b_a3b_gsm8k_sglang_fp8_sync_$(date +%Y%m%d%H)}
trainer_entrypoint=${TRAINER_ENTRYPOINT:-verl.trainer.main_ppo}
########################### end user-adjustable ###########################

mkdir -p "${CKPTS_DIR}"

if [ "${actor_ep}" -gt 1 ] && [ $((NNODES * NGPUS_PER_NODE % actor_ep)) -ne 0 ]; then
    echo "actor_ep=${actor_ep} must divide total GPUs $((NNODES * NGPUS_PER_NODE))" >&2
    exit 2
fi

########################### parameter arrays ###########################

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
)

DATA=(
    data.train_files="['${TRAIN_FILE}']"
    data.val_files="['${TEST_FILE}']"
    data.train_batch_size=${train_batch_size}
    data.prompt_key=prompt
    data.return_raw_chat=True
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=False
    data.truncation=left
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    +actor_rollout_ref.model.override_config.model_config.max_position_embeddings=$((max_prompt_length + max_response_length))
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${actor_ppo_micro_batch_size_per_gpu}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.clip_ratio_low=${actor_clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${actor_clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=${actor_clip_ratio_c}
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.megatron.dtype=bfloat16
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${actor_vpp}
    actor_rollout_ref.actor.megatron.context_parallel_size=${actor_cp}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${actor_ep}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${actor_etp}
    actor_rollout_ref.actor.megatron.param_offload=${all_offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${all_offload}
    actor_rollout_ref.actor.megatron.grad_offload=${all_offload}
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${rollout_log_prob_max_token_len_per_gpu}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${rollout_log_prob_micro_batch_size_per_gpu}
    actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs}
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens}
    actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len}
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length}
    actor_rollout_ref.rollout.response_length=${max_response_length}
    actor_rollout_ref.rollout.temperature=${rollout_temperature}
    actor_rollout_ref.rollout.top_p=${rollout_top_p}
    actor_rollout_ref.rollout.top_k=${rollout_top_k}
    actor_rollout_ref.rollout.val_kwargs.do_sample=${val_do_sample}
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature}
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k}
    actor_rollout_ref.rollout.val_kwargs.n=${val_n}
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.free_cache_engine=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.moe_runner_backend=triton
    +actor_rollout_ref.rollout.engine_kwargs.sglang.fp8_gemm_runner_backend=${sglang_fp8_gemm_runner_backend}
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_flashinfer_autotune=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_flashinfer_allreduce_fusion=False
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_custom_all_reduce=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.log_level=${sglang_log_level}
)

case "${rollout_quantization}" in
    "")
        ;;
    fp8)
        ROLLOUT+=(+actor_rollout_ref.rollout.quantization=fp8)
        ;;
    *)
        echo "ROLLOUT_QUANTIZATION must be empty or fp8, got: ${rollout_quantization}" >&2
        exit 2
        ;;
esac

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ref_log_prob_max_token_len_per_gpu}
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ref_log_prob_micro_batch_size_per_gpu}
    actor_rollout_ref.ref.megatron.dtype=bfloat16
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${ref_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${ref_pp}
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=${ref_vpp}
    actor_rollout_ref.ref.megatron.context_parallel_size=${ref_cp}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${ref_ep}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ref_etp}
    actor_rollout_ref.ref.megatron.param_offload=${all_offload}
    actor_rollout_ref.ref.megatron.use_mbridge=True
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=False
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","tensorboard","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.total_training_steps=${total_training_steps}
    trainer.resume_mode=auto
    trainer.val_before_train=False
    trainer.log_val_generations=${log_val_generations}
    trainer.default_local_dir="${CKPTS_DIR}"
)

EXTRA=(
    +ray_kwargs.ray_init.address=auto
    model_engine=megatron
)

echo "[SyncConfig] model=${MODEL_PATH}"
echo "[SyncConfig] train=${TRAIN_FILE}"
echo "[SyncConfig] val=${TEST_FILE}"
echo "[SyncConfig] rollout=sglang quantization=${rollout_quantization:-bf16} training_dtype=bfloat16"
echo "[SyncConfig] rollout_tp=${rollout_tp} rollout_replicas=$((NNODES * NGPUS_PER_NODE / rollout_tp)) sglang_fp8_gemm=${sglang_fp8_gemm_runner_backend}"
echo "[SyncConfig] nnodes=${NNODES} node_rank=${NODE_RANK} gpus_per_node=${NGPUS_PER_NODE} total_training_steps=${total_training_steps}"

########################### launch ###########################

TRAINING_CMD=(
    python3 -m "${trainer_entrypoint}"
    "${ALGORITHM[@]}"
    "${DATA[@]}"
    "${MODEL[@]}"
    "${ACTOR[@]}"
    "${ROLLOUT[@]}"
    "${REF[@]}"
    "${TRAINER[@]}"
    "${EXTRA[@]}"
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
    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/verl_ray_${NODE_RANK}_${timestamp}}
    mkdir -p "${RAY_TMPDIR}" "${CKPTS_DIR}/run_logs"

    export GLOO_SOCKET_TIMEOUT=${GLOO_SOCKET_TIMEOUT:-600}
    export GLOO_TCP_TIMEOUT=${GLOO_TCP_TIMEOUT:-600}

    echo "INFO: Ray temp dir: ${RAY_TMPDIR}"
    ray stop --force >/dev/null 2>&1 || true
    start_ray_cluster
    wait_for_ray_nodes

    if [ "${NODE_RANK}" = "0" ]; then
        echo "INFO: Rank 0 launching VERL training with Ray address ${RAY_ADDRESS}"
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
