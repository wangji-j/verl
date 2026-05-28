#!/usr/bin/env bash
# Qwen3-30B-A3B GRPO with Megatron backend + SGLang + Fully Async Policy
# Smoke config for one 8-GPU node: 4 trainer GPUs + 4 rollout GPUs.

set -xeuo pipefail

export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-/tmp/flashinfer-workspace}
export SGLANG_IS_FLASHINFER_AVAILABLE=${SGLANG_IS_FLASHINFER_AVAILABLE:-false}
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=${SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK:-True}

project_name=${PROJECT_NAME:-'GRPO-Qwen3-30B-A3B-GSM8K-megatron-sglang'}
exp_name=${EXP_NAME:-"$(date +%Y%m%d%H)_smoke"}

MODEL_PATH=${MODEL_PATH:-/inspire/hdd/project/qianghuaxuexi/public/models/Qwen3-30B-A3B}
TRAIN_FILE=${TRAIN_FILE:-/inspire/hdd/project/qianghuaxuexi/public/datasets/gsm8k/train.parquet}
TEST_FILE=${TEST_FILE:-/inspire/hdd/project/qianghuaxuexi/public/datasets/gsm8k/test.parquet}
CKPTS_DIR=${CKPTS_DIR:-/inspire/hdd/global_user/hujiarui-25046/verl_data/ckpts/qwen3_30b_a3b_sglang_smoke}
mkdir -p "${CKPTS_DIR}"

rollout_mode="async"
rollout_name="sglang"
return_raw_chat=True

adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-1024}

loss_agg_mode="token-mean"
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
train_ppo_micro_batch_size_per_gpu=1
infer_ppo_micro_batch_size_per_gpu=1
optimizer_offload_fraction=${OFFLOAD_FRACTION:-1.}

# 4 trainer GPUs. Keep this conservative for smoke.
COMMON_PP=${COMMON_PP:-1}
COMMON_VPP=${COMMON_VPP:-null}
COMMON_CP=${COMMON_CP:-1}
COMMON_TP=${COMMON_TP:-1}
COMMON_EP=${COMMON_EP:-4}
COMMON_ETP=${COMMON_ETP:-1}

TRAIN_TP=${TRAIN_TP:-$COMMON_TP}
INFER_TP=${INFER_TP:-4}

ACTOR_PP=${ACTOR_PP:-$COMMON_PP}
ACTOR_VPP=${ACTOR_VPP:-$COMMON_VPP}
ACTOR_CP=${ACTOR_CP:-$COMMON_CP}
ACTOR_TP=${ACTOR_TP:-$TRAIN_TP}
ACTOR_EP=${ACTOR_EP:-$COMMON_EP}
ACTOR_ETP=${ACTOR_ETP:-$COMMON_ETP}
REF_PP=${REF_PP:-$COMMON_PP}
REF_VPP=${REF_VPP:-$COMMON_VPP}
REF_CP=${REF_CP:-$COMMON_CP}
REF_TP=${REF_TP:-$TRAIN_TP}
REF_EP=${REF_EP:-$COMMON_EP}
REF_ETP=${REF_ETP:-$COMMON_ETP}

USE_MBRIDGE=True
USE_DIST_CKPT=False

NNODES_ROLLOUT=${NNODES_ROLLOUT:-1}
NNODES_TRAIN=${NNODES_TRAIN:-1}
TRAIN_NGPUS_PER_NODE=${TRAIN_NGPUS_PER_NODE:-4}
ROLLOUT_NGPUS_PER_NODE=${ROLLOUT_NGPUS_PER_NODE:-4}

train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=${N_RESP_PER_PROMPT:-1}
train_prompt_mini_bsz=${PPO_MINI_BATCH_SIZE:-16}
total_rollout_steps=${TOTAL_ROLLOUT_STEPS:-20}
lr_warmup_steps=${LR_WARMUP_STEPS:-1}
test_freq=${TEST_FREQ:--1}
save_freq=${SAVE_FREQ:--1}
staleness_threshold=0.5
trigger_parameter_sync_step=${TRIGGER_PARAMETER_SYNC_STEP:-1}
require_batches=1
partial_rollout=True
val_before_train=False

required_samples=$((train_prompt_mini_bsz * require_batches))
min_rollout_steps_for_one_sync=$((required_samples * trigger_parameter_sync_step))
max_required_samples=$((required_samples * trigger_parameter_sync_step * 3 / 2))
expected_sync_cycles=$((total_rollout_steps / min_rollout_steps_for_one_sync))
if (( total_rollout_steps < min_rollout_steps_for_one_sync )); then
    echo "[AsyncConfig] invalid sample budget: TOTAL_ROLLOUT_STEPS=${total_rollout_steps}, but at least ${min_rollout_steps_for_one_sync} is required for one trainer sync cycle (PPO_MINI_BATCH_SIZE=${train_prompt_mini_bsz}, REQUIRE_BATCHES=${require_batches}, TRIGGER_PARAMETER_SYNC_STEP=${trigger_parameter_sync_step})." >&2
    exit 2
fi
echo "[AsyncConfig] required_samples=${required_samples}, min_rollout_steps_for_one_sync=${min_rollout_steps_for_one_sync}, max_required_samples=${max_required_samples}, total_rollout_steps=${total_rollout_steps}, expected_sync_cycles=${expected_sync_cycles}, save_freq=${save_freq}, test_freq=${test_freq}"

python -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name='fully_async_ppo_megatron_trainer.yaml' \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat} \
    data.gen_batch_size=${gen_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    +actor_rollout_ref.model.override_config.model_config.max_position_embeddings=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.optim.lr_decay_style='constant' \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.lr_decay_steps=${total_rollout_steps} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.actor.megatron.use_mbridge=${USE_MBRIDGE} \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP} \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${ACTOR_VPP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ACTOR_ETP} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type="alltoall" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${infer_ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${INFER_TP} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.moe_runner_backend=triton \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_flashinfer_autotune=True \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_custom_all_reduce=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${infer_ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${REF_TP} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${REF_PP} \
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=${REF_VPP} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${REF_CP} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${REF_EP} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${REF_ETP} \
    trainer.logger=['console','tensorboard','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=${val_before_train} \
    trainer.save_freq="${save_freq}" \
    trainer.total_epochs=10 \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.nnodes="${NNODES_TRAIN}" \
    trainer.n_gpus_per_node="${TRAIN_NGPUS_PER_NODE}" \
    rollout.nnodes="${NNODES_ROLLOUT}" \
    rollout.n_gpus_per_node="${ROLLOUT_NGPUS_PER_NODE}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    trainer.test_freq="${test_freq}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}"
