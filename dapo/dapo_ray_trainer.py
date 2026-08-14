# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import extract_reward
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from verl.utils.reward_score import default_compute_score
from verl.utils.rollout_skip import RolloutSkip


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    _ROUTER_MISMATCH_METRIC_PREFIX = "router/rollout_vs_train/"

    def _current_aware_router_mismatch_rs_enabled(self) -> bool:
        """Return whether RDC is applied immediately before every PPO mini-step."""
        return bool(OmegaConf.select(self.config, "router.enable_current_aware_mismatch_rs", default=False))

    def _validate_current_aware_router_mismatch_rs(self) -> None:
        """Fail early when the requested RDC experiment cannot have the intended semantics."""
        if not self._current_aware_router_mismatch_rs_enabled():
            return

        if not self.router_mismatch_metrics_enabled:
            raise ValueError(
                "router.enable_current_aware_mismatch_rs=True requires "
                "router.enable_mismatch_metrics=True."
            )
        if not self._router_mismatch_rs_enabled():
            raise ValueError(
                "router.enable_current_aware_mismatch_rs=True requires router.enable_mismatch_rs=True."
            )

        ppo_epochs = int(self.config.actor_rollout_ref.actor.ppo_epochs)
        if ppo_epochs != 1:
            raise ValueError(
                "Current-aware router filtering defines one current-route probe followed by one optimizer "
                f"update per PPO mini-step, so actor.ppo_epochs must be 1; got {ppo_epochs}."
            )

        router_replay_mode = str(self.config.actor_rollout_ref.actor.megatron.router_replay.mode)
        if router_replay_mode != "disabled":
            raise ValueError(
                "Current-aware router filtering must measure BF16 free routing, so "
                "actor.megatron.router_replay.mode must be 'disabled'; "
                f"got {router_replay_mode!r}."
            )

        alignment_warmup_steps = self._router_mismatch_alignment_warmup_steps()
        if alignment_warmup_steps != 1:
            raise ValueError(
                "Current-aware router filtering requires router.mismatch_alignment_warmup_steps=1 so the "
                "full theta_0 batch freezes one shared alignment before mini-step filtering; "
                f"got {alignment_warmup_steps}."
            )

        train_prompt_batch_size = int(self.config.data.train_batch_size)
        mini_prompt_batch_size = int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size)
        if mini_prompt_batch_size <= 0 or train_prompt_batch_size % mini_prompt_batch_size != 0:
            raise ValueError(
                "data.train_batch_size must be divisible by actor.ppo_mini_batch_size for current-aware "
                f"router filtering; got {train_prompt_batch_size} and {mini_prompt_batch_size}."
            )

    @classmethod
    def _rename_current_aware_router_metrics(cls, values: dict, mini_step: int) -> dict:
        """Move the existing static mismatch metrics into an RDC mini-step namespace."""
        target_prefix = f"router/rdc/mini_step_{mini_step}/"
        renamed = {}
        for key, value in values.items():
            if key.startswith(cls._ROUTER_MISMATCH_METRIC_PREFIX):
                key = target_prefix + key[len(cls._ROUTER_MISMATCH_METRIC_PREFIX) :]
            else:
                key = target_prefix + key
            renamed[key] = value
        return renamed

    def _update_actor_with_current_aware_router_filter(
        self,
        batch: DataProto,
        metrics: dict,
        timing_raw: dict,
    ) -> DataProto:
        """Probe current routes, reject within each length bucket, then update each mini-batch once.

        ``old_log_probs`` and TIS weights remain those computed once at the start of the
        rollout batch. Only the route comparison is refreshed: mini-step ``j`` compares
        the cached FP8 rollout routes with free BF16 routes under ``theta_(j-1)``.
        """
        if "routed_experts" not in batch.batch:
            raise RuntimeError(
                "Current-aware router filtering requires cached FP8 rollout routed_experts in the PPO batch."
            )

        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        mini_prompt_batch_size = int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size)
        mini_response_batch_size = mini_prompt_batch_size * rollout_n
        if len(batch) % mini_response_batch_size != 0:
            raise ValueError(
                "The response batch must split exactly into PPO mini-batches for current-aware router "
                f"filtering; got responses={len(batch)}, mini_response_batch_size={mini_response_batch_size}."
            )

        # The controller batch is rank-major after balancing/dispatch: all
        # samples for actor DP rank 0, then rank 1, and so on. Match the native
        # worker iterator by taking one slice from every rank partition for each
        # global mini-step. A plain contiguous split would make early steps draw
        # from only the first rank partition when actor DP > 1.
        actor_dp_size = int(self._get_dp_size(self.actor_rollout_wg, "actor"))
        if actor_dp_size <= 0 or len(batch) % actor_dp_size != 0:
            raise ValueError(
                "The RDC response batch must split evenly across actor DP ranks; "
                f"got responses={len(batch)}, actor_dp_size={actor_dp_size}."
            )
        if mini_response_batch_size % actor_dp_size != 0:
            raise ValueError(
                "Each RDC mini-batch must split evenly across actor DP ranks; "
                f"got mini_response_batch_size={mini_response_batch_size}, actor_dp_size={actor_dp_size}."
            )
        num_mini_steps = len(batch) // mini_response_batch_size
        responses_per_dp = len(batch) // actor_dp_size
        mini_responses_per_dp = mini_response_batch_size // actor_dp_size
        mini_batch_indices = []
        for mini_step_idx in range(num_mini_steps):
            mini_indices = []
            for dp_rank in range(actor_dp_size):
                start = dp_rank * responses_per_dp + mini_step_idx * mini_responses_per_dp
                mini_indices.extend(range(start, start + mini_responses_per_dp))
            mini_batch_indices.append(mini_indices)

        actor_metrics_across_steps: dict[str, list] = defaultdict(list)
        updated_response_mask = batch.batch["response_mask"].clone()
        total_rejected = 0.0
        total_valid = 0.0
        total_valid_tokens_before = 0.0
        total_valid_tokens_after = 0.0
        bucket_totals: dict[int, dict[str, float]] = defaultdict(lambda: {"count": 0.0, "rejected": 0.0})
        bucket_count = len(self._router_mismatch_rs_length_bucket_edges()) + 1

        metrics["router/rdc/enabled"] = 1.0
        metrics["router/rdc/mini_steps"] = float(num_mini_steps)
        metrics["router/rdc/mini_prompt_batch_size"] = float(mini_prompt_batch_size)
        metrics["router/rdc/mini_response_batch_size"] = float(mini_response_batch_size)
        metrics["router/rdc/actor_dp_size"] = float(actor_dp_size)
        metrics["router/rdc/mini_response_batch_size_per_dp"] = float(mini_responses_per_dp)
        metrics["router/rdc/configured_rejected_fraction"] = self._router_mismatch_rs_fraction()

        for mini_step, mini_indices in enumerate(mini_batch_indices, start=1):
            # Materialize only the mini-batch being probed/updated. Building all
            # advanced-indexed mini-batches up front would duplicate the full
            # cached route tensor on the controller.
            mini_batch = batch[mini_indices]
            # The actor has theta_(j-1) here. compute_old_log_prob is used only as
            # a teacher-forced current-route probe; its probabilities are discarded.
            # Do not send behavior routes into this forward: besides saving an
            # unnecessary device copy, removing them makes the free-routing
            # semantics explicit. Restore them only for the distance comparison.
            rollout_routes = mini_batch.batch.pop("routed_experts")
            try:
                with marked_timer("rdc_current_route", timing_raw, "blue"):
                    current_actor_output, current_actor_mfu = self._compute_old_log_prob(mini_batch)
            finally:
                mini_batch.batch["routed_experts"] = rollout_routes
            if "routed_experts" not in current_actor_output.batch:
                raise RuntimeError(
                    "Current-aware router filtering did not receive routed_experts from the current BF16 actor."
                )

            with marked_timer("rdc_router_mismatch", timing_raw, "blue"):
                router_result = self._compute_router_mismatch_result(mini_batch, current_actor_output)
            if router_result is None:
                raise RuntimeError("Current-aware router filtering could not compute router mismatch metrics.")

            step_metrics = dict(router_result.metrics)
            valid_tokens_before = float(mini_batch.batch["response_mask"].sum().item())
            with marked_timer("rdc_router_filter", timing_raw, "blue"):
                filter_metrics = self._apply_router_mismatch_rs(mini_batch, router_result)
            valid_tokens_after = float(mini_batch.batch["response_mask"].sum().item())
            rejected_tokens = max(valid_tokens_before - valid_tokens_after, 0.0)
            step_metrics.update(filter_metrics)
            step_metrics["current_actor_mfu"] = current_actor_mfu
            step_metrics["response_count"] = float(len(mini_batch))
            metrics.update(self._rename_current_aware_router_metrics(step_metrics, mini_step))

            rejected = float(filter_metrics.get("router/rollout_vs_train/rs_rejected_count", 0.0))
            kept = float(filter_metrics.get("router/rollout_vs_train/rs_kept_count", 0.0))
            valid_responses = rejected + kept
            actual_rejected_fraction = rejected / valid_responses if valid_responses > 0 else 0.0
            total_rejected += rejected
            total_valid += valid_responses
            total_valid_tokens_before += valid_tokens_before
            total_valid_tokens_after += valid_tokens_after
            step_filter_prefix = f"router/rdc/mini_step_{mini_step}/filter"
            step_filter_summary = {
                f"{step_filter_prefix}/total_response_count": float(len(mini_batch)),
                f"{step_filter_prefix}/valid_response_count": valid_responses,
                f"{step_filter_prefix}/filtered_response_count": rejected,
                f"{step_filter_prefix}/rejected_response_count": rejected,
                f"{step_filter_prefix}/kept_response_count": kept,
                f"{step_filter_prefix}/actual_rejected_fraction": actual_rejected_fraction,
                f"{step_filter_prefix}/configured_rejected_fraction": self._router_mismatch_rs_fraction(),
                f"{step_filter_prefix}/valid_token_count_before": valid_tokens_before,
                f"{step_filter_prefix}/valid_token_count_after": valid_tokens_after,
                f"{step_filter_prefix}/rejected_token_count": rejected_tokens,
                f"{step_filter_prefix}/rejected_token_fraction": (
                    rejected_tokens / valid_tokens_before if valid_tokens_before > 0 else 0.0
                ),
            }
            for bucket_idx in range(bucket_count):
                source_prefix = f"router/rollout_vs_train/rs_bucket_{bucket_idx}"
                bucket_totals[bucket_idx]["count"] += float(filter_metrics.get(f"{source_prefix}_count", 0.0))
                bucket_totals[bucket_idx]["rejected"] += float(
                    filter_metrics.get(f"{source_prefix}_rejected_count", 0.0)
                )

            # The probe's log-probs, entropy, and current routed-expert tensor
            # are not PPO anchors. Release them before the backward pass so the
            # extra teacher-forced probe does not inflate peak actor memory.
            del current_actor_output, router_result

            # Keep only the filtered response mask for training. The current
            # probabilities/entropy are deliberately not merged into the batch.
            updated_response_mask[mini_indices] = mini_batch.batch["response_mask"]
            mini_batch.batch.pop("router_mismatch_reject_mask", None)
            mini_batch.batch.pop("routed_experts")
            del rollout_routes
            # The native worker advances the LR scheduler once after all PPO
            # mini-batches in a rollout batch. RDC uses one RPC per mini-step,
            # so suppress scheduler advancement until the final RPC.
            mini_batch.meta_info["update_lr_scheduler_at_end"] = mini_step == num_mini_steps

            with marked_timer("rdc_actor_update", timing_raw, "red"):
                actor_output = self._update_actor(mini_batch)
            # These summaries are attached only after the corresponding actor
            # update returns successfully. The trainer flushes them together at
            # the end of the enclosing global training step.
            metrics.update(step_filter_summary)
            step_prefix = f"router/rdc/mini_step_{mini_step}"
            metrics[f"{step_prefix}/update_completed"] = 1.0
            metrics[f"{step_prefix}/cumulative_filtered_response_count"] = total_rejected
            metrics[f"{step_prefix}/cumulative_valid_response_count"] = total_valid
            metrics[f"{step_prefix}/cumulative_actual_rejected_fraction"] = (
                total_rejected / total_valid if total_valid > 0 else 0.0
            )
            cumulative_rejected_tokens = total_valid_tokens_before - total_valid_tokens_after
            metrics[f"{step_prefix}/cumulative_rejected_token_count"] = cumulative_rejected_tokens
            metrics[f"{step_prefix}/cumulative_rejected_token_fraction"] = (
                cumulative_rejected_tokens / total_valid_tokens_before if total_valid_tokens_before > 0 else 0.0
            )
            step_actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            for key, value in step_actor_metrics.items():
                actor_metrics_across_steps[key].append(value)

        # Advanced indexing creates independent mini-batches, so write every
        # filtered mask back in the original controller-batch order.
        batch.batch["response_mask"] = updated_response_mask
        batch.batch.pop("routed_experts")
        metrics["router/rdc/rejected_count"] = total_rejected
        metrics["router/rdc/valid_response_count"] = total_valid
        metrics["router/rdc/rejected_fraction"] = total_rejected / total_valid if total_valid > 0 else 0.0
        metrics["router/rdc/valid_token_count_before"] = total_valid_tokens_before
        metrics["router/rdc/valid_token_count_after"] = total_valid_tokens_after
        metrics["router/rdc/rejected_token_count"] = total_valid_tokens_before - total_valid_tokens_after
        metrics["router/rdc/rejected_token_fraction"] = (
            (total_valid_tokens_before - total_valid_tokens_after) / total_valid_tokens_before
            if total_valid_tokens_before > 0
            else 0.0
        )
        for bucket_idx, totals in bucket_totals.items():
            count = totals["count"]
            rejected = totals["rejected"]
            prefix = f"router/rdc/rs_bucket_{bucket_idx}"
            metrics[f"{prefix}_count"] = count
            metrics[f"{prefix}_rejected_count"] = rejected
            metrics[f"{prefix}_rejected_fraction"] = rejected / count if count > 0 else 0.0

        return DataProto.from_single_dict(data={}, meta_info={"metrics": dict(actor_metrics_across_steps)})

    def _reward_extract_debug_enabled(self) -> bool:
        return bool(os.getenv("VERL_REWARD_DEBUG_DIR", "").strip())

    def _maybe_log_extract_reward_debug(
        self, batch: DataProto, reward_tensor: torch.Tensor, reward_extra_infos_dict: dict
    ) -> None:
        debug_dir = os.getenv("VERL_REWARD_DEBUG_DIR", "").strip()
        if not debug_dir:
            return

        debug_steps = int(os.getenv("VERL_REWARD_DEBUG_STEPS", "0") or 0)
        debug_samples = int(os.getenv("VERL_REWARD_DEBUG_SAMPLES", "8") or 0)
        if debug_steps <= 0 or debug_samples <= 0:
            return

        global_step = int(batch.meta_info.get("global_steps", self.global_steps)) if batch.meta_info else self.global_steps
        if global_step < 0 or global_step >= debug_steps:
            return

        rows = []
        num_samples = min(len(batch), debug_samples)
        reward_extra_keys = list(reward_extra_infos_dict.keys()) if reward_extra_infos_dict else []

        for i in range(num_samples):
            data_item = batch[i]
            prompt_ids = data_item.batch["prompts"]
            response_ids = data_item.batch["responses"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum().item())
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum().item())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:] if valid_prompt_length > 0 else prompt_ids[:0]
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if eos_token and response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            reward_model = data_item.non_tensor_batch.get("reward_model", {})
            ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
            data_source = data_item.non_tensor_batch.get("data_source", "unknown")
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            rm_score = float(reward_tensor[i].sum().item())

            parser_result = None
            parser_error = None
            if ground_truth is not None:
                try:
                    parser_result = default_compute_score(
                        data_source=data_source,
                        solution_str=response_str,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                    )
                except Exception as exc:  # debug path must not affect training
                    parser_error = repr(exc)

            if isinstance(parser_result, dict):
                parser_score = parser_result.get("score")
                parser_acc = parser_result.get("acc")
                parser_pred = parser_result.get("pred")
            else:
                parser_score = parser_result
                parser_acc = parser_result
                parser_pred = None

            reward_extra = {}
            for key in reward_extra_keys:
                values = reward_extra_infos_dict.get(key, [])
                if i < len(values):
                    value = values[i]
                    if isinstance(value, np.generic):
                        value = value.item()
                    reward_extra[key] = value

            rows.append(
                {
                    "global_step": global_step,
                    "sample_index": i,
                    "uid": str(data_item.non_tensor_batch.get("uid", "")),
                    "data_source": str(data_source),
                    "ground_truth": ground_truth,
                    "rm_score": rm_score,
                    "parser_score_debug": parser_score,
                    "parser_acc_debug": bool(parser_acc) if isinstance(parser_acc, (bool, np.bool_)) else parser_acc,
                    "parser_pred_debug": parser_pred,
                    "parser_error": parser_error,
                    "reward_extra": reward_extra,
                    "prompt": prompt_str,
                    "response": response_str,
                    "response_head_1000": response_str[:1000],
                    "response_tail_1000": response_str[-1000:],
                    "response_length": valid_response_length,
                    "has_answer_prefix": "answer:" in response_str.lower()[-1000:],
                    "has_boxed": "\\boxed" in response_str[-1000:],
                }
            )

        if not rows:
            return
        os.makedirs(debug_dir, exist_ok=True)
        path = os.path.join(debug_dir, f"extract_reward_debug_step{global_step:04d}_pid{os.getpid()}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def _router_analysis_dump_dir(self) -> str:
        return os.getenv("VERL_ROUTER_ANALYSIS_DUMP_DIR", "").strip()

    def _router_analysis_dump_mode(self) -> str:
        return os.getenv("VERL_ROUTER_ANALYSIS_DUMP_MODE", "summary").strip().lower()

    def _router_analysis_dump_float_dtype(self) -> torch.dtype:
        dtype = os.getenv("VERL_ROUTER_ANALYSIS_DUMP_FLOAT_DTYPE", "float16").strip().lower()
        if dtype in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if dtype in {"fp32", "float32"}:
            return torch.float32
        return torch.float16

    def _router_analysis_dump_topk_tokens(self) -> int:
        return max(int(os.getenv("VERL_ROUTER_ANALYSIS_DUMP_TOPK_TOKENS", "32") or 32), 0)

    def _should_dump_router_analysis(self, global_step: int) -> bool:
        if self._router_analysis_dump_mode() in {"", "0", "false", "off", "none", "disable", "disabled"}:
            return False
        if not self._router_analysis_dump_dir():
            return False
        max_steps = int(os.getenv("VERL_ROUTER_ANALYSIS_DUMP_STEPS", "0") or 0)
        if max_steps > 0 and global_step >= max_steps:
            return False
        every_n = max(int(os.getenv("VERL_ROUTER_ANALYSIS_DUMP_EVERY_N", "1") or 1), 1)
        return global_step % every_n == 0

    @staticmethod
    def _tensor_to_cpu(tensor: torch.Tensor | None, *, dtype: torch.dtype | None = None) -> torch.Tensor | None:
        if tensor is None:
            return None
        tensor = tensor.detach()
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor.cpu()

    def _maybe_dump_router_analysis(self, batch: DataProto, old_log_prob: DataProto, router_result) -> None:
        global_step = int(getattr(self, "global_steps", -1))
        if router_result is None or not self._should_dump_router_analysis(global_step):
            return

        mode = self._router_analysis_dump_mode()
        if mode not in {"summary", "tokens", "sample", "full", "expert_counts"}:
            raise ValueError(
                "VERL_ROUTER_ANALYSIS_DUMP_MODE must be one of summary, tokens, sample, full, expert_counts; "
                f"got {mode!r}"
            )

        dump_dir = self._router_analysis_dump_dir()
        os.makedirs(dump_dir, exist_ok=True)
        float_dtype = self._router_analysis_dump_float_dtype()
        responses = batch.batch.get("responses")
        attention_mask = batch.batch.get("attention_mask")
        if responses is not None and attention_mask is not None:
            response_mask = attention_mask[:, -responses.shape[-1] :].bool()
        else:
            response_mask = batch.batch["response_mask"].bool()
        rs_reject_mask = batch.batch.get("router_mismatch_reject_mask")
        old_log_probs = old_log_prob.batch.get("old_log_probs")
        rollout_log_probs = batch.batch.get("rollout_log_probs")

        seq_valid_token_count = router_result.seq_valid_token_count
        seq_mismatch = router_result.seq_mismatch
        token_mismatch = router_result.token_mismatch

        with torch.no_grad():
            seq_prob_diff = None
            seq_logprob_diff = None
            prob_diff = None
            logprob_diff = None
            if old_log_probs is not None and rollout_log_probs is not None:
                mask = response_mask.to(device=old_log_probs.device)
                rollout_log_probs = rollout_log_probs.to(device=old_log_probs.device)
                valid_count = mask.float().sum(dim=-1).clamp_min(1.0)
                prob_diff = (old_log_probs.float().exp() - rollout_log_probs.float().exp()).abs()
                logprob_diff = (old_log_probs.float() - rollout_log_probs.float()).abs()
                seq_prob_diff = (prob_diff * mask.float()).sum(dim=-1) / valid_count
                seq_logprob_diff = (logprob_diff * mask.float()).sum(dim=-1) / valid_count

            token_level_scores = batch.batch.get("token_level_scores")
            seq_reward = token_level_scores.sum(dim=-1) if token_level_scores is not None else None
            extreme_tokens = None
            topk_tokens = self._router_analysis_dump_topk_tokens()
            if topk_tokens > 0 and token_mismatch is not None:
                mismatch_scores = token_mismatch.float()
                valid_mask = response_mask.to(device=mismatch_scores.device)
                scored = mismatch_scores.masked_fill(~valid_mask, float("-inf"))
                k = min(topk_tokens, scored.shape[-1])
                top_values, top_indices = torch.topk(scored, k=k, dim=-1, largest=True, sorted=True)
                top_valid = torch.isfinite(top_values)
                extreme_tokens = {
                    "topk": k,
                    "indices": self._tensor_to_cpu(top_indices, dtype=torch.int32),
                    "valid": self._tensor_to_cpu(top_valid),
                    "token_mismatch": self._tensor_to_cpu(top_values.masked_fill(~top_valid, 0.0), dtype=float_dtype),
                }
                if responses is not None:
                    response_ids = responses.to(device=top_indices.device)
                    extreme_tokens["token_ids"] = self._tensor_to_cpu(
                        torch.gather(response_ids, dim=-1, index=top_indices.clamp_min(0)),
                        dtype=torch.int32,
                    )
                if old_log_probs is not None:
                    old_on_topk = old_log_probs.to(device=top_indices.device)
                    extreme_tokens["old_log_probs"] = self._tensor_to_cpu(
                        torch.gather(old_on_topk, dim=-1, index=top_indices.clamp_min(0)),
                        dtype=float_dtype,
                    )
                if rollout_log_probs is not None:
                    rollout_on_topk = rollout_log_probs.to(device=top_indices.device)
                    extreme_tokens["rollout_log_probs"] = self._tensor_to_cpu(
                        torch.gather(rollout_on_topk, dim=-1, index=top_indices.clamp_min(0)),
                        dtype=float_dtype,
                    )
                if prob_diff is not None:
                    prob_diff_on_topk = prob_diff.to(device=top_indices.device)
                    extreme_tokens["prob_diff"] = self._tensor_to_cpu(
                        torch.gather(prob_diff_on_topk, dim=-1, index=top_indices.clamp_min(0)),
                        dtype=float_dtype,
                    )
                if logprob_diff is not None:
                    logprob_diff_on_topk = logprob_diff.to(device=top_indices.device)
                    extreme_tokens["logprob_diff"] = self._tensor_to_cpu(
                        torch.gather(logprob_diff_on_topk, dim=-1, index=top_indices.clamp_min(0)),
                        dtype=float_dtype,
                    )
            extreme_prob_diff_tokens = None
            if topk_tokens > 0 and prob_diff is not None:
                valid_mask = response_mask.to(device=prob_diff.device)
                scored = prob_diff.float().masked_fill(~valid_mask, float("-inf"))
                k = min(topk_tokens, scored.shape[-1])
                top_values, top_indices = torch.topk(scored, k=k, dim=-1, largest=True, sorted=True)
                top_valid = torch.isfinite(top_values)
                extreme_prob_diff_tokens = {
                    "topk": k,
                    "indices": self._tensor_to_cpu(top_indices, dtype=torch.int32),
                    "valid": self._tensor_to_cpu(top_valid),
                    "prob_diff": self._tensor_to_cpu(top_values.masked_fill(~top_valid, 0.0), dtype=float_dtype),
                }
                if responses is not None:
                    response_ids = responses.to(device=top_indices.device)
                    extreme_prob_diff_tokens["token_ids"] = self._tensor_to_cpu(
                        torch.gather(response_ids, dim=-1, index=top_indices.clamp_min(0)),
                        dtype=torch.int32,
                    )
                if token_mismatch is not None:
                    mismatch_on_topk = token_mismatch.to(device=top_indices.device)
                    extreme_prob_diff_tokens["token_mismatch"] = self._tensor_to_cpu(
                        torch.gather(mismatch_on_topk, dim=-1, index=top_indices.clamp_min(0)),
                        dtype=float_dtype,
                    )
                if old_log_probs is not None:
                    old_on_topk = old_log_probs.to(device=top_indices.device)
                    extreme_prob_diff_tokens["old_log_probs"] = self._tensor_to_cpu(
                        torch.gather(old_on_topk, dim=-1, index=top_indices.clamp_min(0)),
                        dtype=float_dtype,
                    )
                if rollout_log_probs is not None:
                    rollout_on_topk = rollout_log_probs.to(device=top_indices.device)
                    extreme_prob_diff_tokens["rollout_log_probs"] = self._tensor_to_cpu(
                        torch.gather(rollout_on_topk, dim=-1, index=top_indices.clamp_min(0)),
                        dtype=float_dtype,
                    )
                if logprob_diff is not None:
                    logprob_diff_on_topk = logprob_diff.to(device=top_indices.device)
                    extreme_prob_diff_tokens["logprob_diff"] = self._tensor_to_cpu(
                        torch.gather(logprob_diff_on_topk, dim=-1, index=top_indices.clamp_min(0)),
                        dtype=float_dtype,
                    )

            payload = {
                "metadata": {
                    "global_step": global_step,
                    "mode": mode,
                    "alignment": int(router_result.alignment),
                    "alignment_scores": router_result.alignment_scores,
                    "batch_size": int(batch.batch.batch_size[0]) if batch.batch is not None else 0,
                    "response_mask_shape": list(response_mask.shape),
                    "response_mask_tokens": int(response_mask.sum().item()),
                    "float_dtype": str(float_dtype),
                    "extreme_token_topk": topk_tokens,
                    "rollout_routes_shape": list(batch.batch["routed_experts"].shape)
                    if "routed_experts" in batch.batch
                    else None,
                    "train_routes_shape": list(old_log_prob.batch["routed_experts"].shape)
                    if "routed_experts" in old_log_prob.batch
                    else None,
                    "rollout_routes_dtype": str(batch.batch["routed_experts"].dtype)
                    if "routed_experts" in batch.batch
                    else None,
                    "train_routes_dtype": str(old_log_prob.batch["routed_experts"].dtype)
                    if "routed_experts" in old_log_prob.batch
                    else None,
                    "metric_names": sorted(router_result.metrics.keys()),
                },
                "metrics": dict(router_result.metrics),
                "response_summary": {
                    "seq_mismatch": self._tensor_to_cpu(seq_mismatch, dtype=torch.float32),
                    "seq_valid_token_count": self._tensor_to_cpu(seq_valid_token_count, dtype=torch.float32),
                    "seq_prob_diff": self._tensor_to_cpu(seq_prob_diff, dtype=torch.float32),
                    "seq_logprob_diff": self._tensor_to_cpu(seq_logprob_diff, dtype=torch.float32),
                    "seq_reward": self._tensor_to_cpu(seq_reward, dtype=torch.float32),
                    "rs_reject_mask": self._tensor_to_cpu(rs_reject_mask),
                    "extreme_tokens": extreme_tokens,
                    "extreme_prob_diff_tokens": extreme_prob_diff_tokens,
                    "expert_usage_distances": {
                        name: self._tensor_to_cpu(values, dtype=torch.float32)
                        for name, values in (router_result.expert_usage_distances or {}).items()
                    },
                },
            }

            uids = batch.non_tensor_batch.get("uid") if hasattr(batch, "non_tensor_batch") else None
            if uids is not None:
                payload["uids"] = [str(uid) for uid in uids]

            if mode in {"tokens", "sample", "full"}:
                payload["token_data"] = {
                    "responses": self._tensor_to_cpu(responses, dtype=torch.int32),
                    "response_mask": self._tensor_to_cpu(response_mask),
                    "attention_mask": self._tensor_to_cpu(attention_mask),
                    "token_mismatch": self._tensor_to_cpu(token_mismatch, dtype=float_dtype),
                    "old_log_probs": self._tensor_to_cpu(old_log_probs, dtype=float_dtype),
                    "rollout_log_probs": self._tensor_to_cpu(rollout_log_probs, dtype=float_dtype),
                    "logprob_delta": self._tensor_to_cpu(
                        old_log_probs.float() - rollout_log_probs.float()
                        if old_log_probs is not None and rollout_log_probs is not None
                        else None,
                        dtype=float_dtype,
                    ),
                    "logprob_diff": self._tensor_to_cpu(logprob_diff, dtype=float_dtype),
                    "prob_diff": self._tensor_to_cpu(prob_diff, dtype=float_dtype),
                }

            if mode in {"sample", "full"}:
                route_limit = int(os.getenv("VERL_ROUTER_ANALYSIS_DUMP_SAMPLES", "16") or 16)
                route_count = int(response_mask.shape[0]) if mode == "full" else min(route_limit, int(response_mask.shape[0]))
                route_slice = slice(0, route_count)
                payload["routed_experts"] = {
                    "sample_count": route_count,
                    "rollout": self._tensor_to_cpu(batch.batch["routed_experts"][route_slice])
                    if "routed_experts" in batch.batch
                    else None,
                    "train": self._tensor_to_cpu(old_log_prob.batch["routed_experts"][route_slice])
                    if "routed_experts" in old_log_prob.batch
                    else None,
                }

            if mode == "expert_counts":
                if router_result.expert_usage_counts is None:
                    raise RuntimeError(
                        "expert_counts dump mode requires per-response expert counts, but none were captured"
                    )
                payload["expert_usage"] = {
                    "distances_by_layer": {
                        name: self._tensor_to_cpu(values, dtype=float_dtype)
                        for name, values in (router_result.expert_usage_distances_by_layer or {}).items()
                    },
                    "counts": {
                        side: self._tensor_to_cpu(values)
                        for side, values in router_result.expert_usage_counts.items()
                    },
                }

        path = os.path.join(dump_dir, f"router_analysis_step{global_step:04d}_pid{os.getpid()}.pt")
        torch.save(payload, path)

    def compute_kl_related_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict):
        batch.batch["response_mask"] = compute_response_mask(batch)

        # recompute old_log_probs
        with marked_timer("old_log_prob", timing_raw, "blue"):
            # RDC needs a free BF16 theta_0 route here, while retaining the FP8
            # behavior route on the driver for later mini-steps. Do not serialize
            # that large cached tensor into the actor forward request.
            cached_rollout_routes = None
            if self._current_aware_router_mismatch_rs_enabled():
                cached_rollout_routes = batch.batch.pop("routed_experts", None)
            try:
                with marked_timer("old_log_prob_forward", timing_raw, "blue"):
                    old_log_prob_debug_device = (
                        batch.batch["responses"].device if "responses" in batch.batch else torch.device("cuda")
                    )
                    old_log_prob_debug_before = self._cuda_memory_snapshot(old_log_prob_debug_device)
                    old_log_prob_debug_t0 = time.perf_counter()
                    old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                    old_log_prob_debug_after = self._cuda_memory_snapshot(old_log_prob_debug_device)
                    self._write_perf_debug(
                        "old_log_prob_forward",
                        {
                            "duration_s": time.perf_counter() - old_log_prob_debug_t0,
                            "batch_size": int(batch.batch.batch_size[0]) if batch.batch is not None else 0,
                            "responses_shape": list(batch.batch["responses"].shape)
                            if "responses" in batch.batch
                            else None,
                            "response_mask_tokens": int(batch.batch["response_mask"].sum().item())
                            if "response_mask" in batch.batch
                            else None,
                            "returned_keys": list(old_log_prob.batch.keys()),
                            "mfu": old_log_prob_mfu,
                            **{f"before_{k}": v for k, v in old_log_prob_debug_before.items()},
                            **{f"after_{k}": v for k, v in old_log_prob_debug_after.items()},
                        },
                    )
            finally:
                if cached_rollout_routes is not None:
                    batch.batch["routed_experts"] = cached_rollout_routes
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            actor_config = self.config.actor_rollout_ref.actor
            entropy_agg = agg_loss(
                loss_mat=entropys,
                loss_mask=response_masks,
                loss_agg_mode=actor_config.loss_agg_mode,
                loss_scale_factor=actor_config.loss_scale_factor,
            )
            old_log_prob_metrics = {
                "actor/entropy": entropy_agg.detach().item(),
                "perf/mfu/actor_infer": old_log_prob_mfu,
            }
            metrics.update(old_log_prob_metrics)
            old_log_prob.batch.pop("entropys")
            if self.router_mismatch_metrics_enabled:
                missing_routed_experts = []
                if "routed_experts" not in batch.batch:
                    missing_routed_experts.append("rollout/vLLM batch.routed_experts")
                if "routed_experts" not in old_log_prob.batch:
                    missing_routed_experts.append("actor old_log_prob.routed_experts")
                if missing_routed_experts:
                    raise RuntimeError(
                        "router.enable_mismatch_metrics=True but routed_experts is missing from: "
                        + ", ".join(missing_routed_experts)
                    )
                with marked_timer("router_mismatch_metrics", timing_raw, "blue"):
                    router_result = self._compute_router_mismatch_result(batch, old_log_prob)
                if router_result is not None:
                    metrics.update(router_result.metrics)
                    # Static RS compares rollout routes with theta_0 once for the
                    # whole batch. RDC instead performs rejection immediately
                    # before each mini-step using theta_(j-1), so do not mask here.
                    if not self._current_aware_router_mismatch_rs_enabled():
                        with marked_timer("router_mismatch_rs", timing_raw, "blue"):
                            metrics.update(self._apply_router_mismatch_rs(batch, router_result))
                    self._maybe_dump_router_analysis(batch, old_log_prob, router_result)
                    batch.batch.pop("router_mismatch_reject_mask", None)
                old_log_prob.batch.pop("routed_experts")
                if not self._current_aware_router_mismatch_rs_enabled():
                    batch.batch.pop("routed_experts")
            batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            # compute reference log_prob
            with marked_timer("ref", timing_raw, "olive"):
                # Reference scoring never consumes router traces. Keep the RDC
                # behavior-route cache on the driver instead of copying it to the
                # reference worker.
                cached_rollout_routes = None
                if self._current_aware_router_mismatch_rs_enabled():
                    cached_rollout_routes = batch.batch.pop("routed_experts", None)
                try:
                    ref_log_prob = self._compute_ref_log_prob(batch)
                finally:
                    if cached_rollout_routes is not None:
                        batch.batch["routed_experts"] = cached_rollout_routes
                batch = batch.union(ref_log_prob)

        return batch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking

        self._validate_current_aware_router_mismatch_rs()

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0
        self.max_steps_duration = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        filter_groups_stats = defaultdict(int)
        current_epoch = self.global_steps // len(self.train_dataloader)

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                new_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                num_gen_batches += 1
                gen_batch = self._get_gen_batch(new_batch)
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            # compute reward model score on new_batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                                rm_scores = self._compute_reward_colocate(new_batch)
                                new_batch = new_batch.union(rm_scores)
                            reward_baseline_tensor, _ = extract_reward(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            new_batch.pop(batch_keys=list(keys_to_pop))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    if self.config.algorithm.use_kl_in_reward:
                        # We need these metrics for apply_kl_penalty if using kl in reward
                        new_batch = self.compute_kl_related_metrics(new_batch, metrics, timing_raw)
                        # otherwise, we will compute those after dynamic sampling

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                            # we first compute reward model score
                            batch_reward = self._compute_reward_colocate(new_batch)
                            new_batch = new_batch.union(batch_reward)

                        # we combine with rule-based rm
                        reward_tensor, reward_extra_infos_dict = extract_reward(new_batch)
                        self._maybe_log_extract_reward_debug(new_batch, reward_tensor, reward_extra_infos_dict)

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        total_prompt_groups = len(prompt_uid2metric_vals)
                        kept_prompt_uid_set = set(kept_prompt_uids)
                        filtered_prompt_uids = [
                            uid for uid in prompt_uid2metric_vals if uid not in kept_prompt_uid_set
                        ]
                        filtered_all_correct = 0
                        filtered_all_wrong = 0
                        filtered_other_constant = 0
                        for uid in filtered_prompt_uids:
                            metric_vals = np.asarray(prompt_uid2metric_vals[uid], dtype=float)
                            if np.all(metric_vals == 1):
                                filtered_all_correct += 1
                            elif np.all(metric_vals == 0):
                                filtered_all_wrong += 1
                            else:
                                filtered_other_constant += 1

                        filter_groups_stats["groups_total"] += total_prompt_groups
                        filter_groups_stats["groups_kept"] += len(kept_prompt_uids)
                        filter_groups_stats["groups_filtered"] += len(filtered_prompt_uids)
                        filter_groups_stats["groups_filtered_all_correct"] += filtered_all_correct
                        filter_groups_stats["groups_filtered_all_wrong"] += filtered_all_wrong
                        filter_groups_stats["groups_filtered_other_constant"] += filtered_other_constant
                        num_prompt_in_batch += len(kept_prompt_uids)

                        print(
                            "filter_groups: "
                            f"gen_batch={num_gen_batches} metric={metric_name} "
                            f"groups={total_prompt_groups} kept={len(kept_prompt_uids)} "
                            f"filtered={len(filtered_prompt_uids)} "
                            f"filtered_all_correct={filtered_all_correct} "
                            f"filtered_all_wrong={filtered_all_wrong} "
                            f"filtered_other_constant={filtered_other_constant} "
                            f"accumulated_kept={num_prompt_in_batch}/{self.config.data.train_batch_size}"
                        )

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                self.gen_steps += 1
                                is_last_step = self.global_steps >= self.total_training_steps
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    self.checkpoint_manager.sleep_replicas()

                    # === Updating ===
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    if not self.config.algorithm.use_kl_in_reward:
                        batch = self.compute_kl_related_metrics(batch, metrics, timing_raw)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            cached_rollout_routes = None
                            if self._current_aware_router_mismatch_rs_enabled():
                                cached_rollout_routes = batch.batch.pop("routed_experts", None)
                            try:
                                values = self._compute_values(batch)
                            finally:
                                if cached_rollout_routes is not None:
                                    batch.batch["routed_experts"] = cached_rollout_routes
                            batch = batch.union(values)

                    # Compute rollout correction weights and off-policy metrics (inherited from RayPPOTrainer)
                    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                        batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                        # IS and off-policy metrics already have rollout_corr/ prefix
                        metrics.update(is_metrics)
                        # add diff of probs too.
                        from verl.utils.debug.metrics import calculate_debug_metrics

                        metrics.update(calculate_debug_metrics(batch))

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            cached_rollout_routes = None
                            if self._current_aware_router_mismatch_rs_enabled():
                                cached_rollout_routes = batch.batch.pop("routed_experts", None)
                            try:
                                critic_output = self._update_critic(batch)
                            finally:
                                if cached_rollout_routes is not None:
                                    batch.batch["routed_experts"] = cached_rollout_routes
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            if self._current_aware_router_mismatch_rs_enabled():
                                actor_output = self._update_actor_with_current_aware_router_filter(
                                    batch,
                                    metrics,
                                    timing_raw,
                                )
                            else:
                                actor_output = self._update_actor(batch)

                        # Check if ESI/training plan is close to expiration
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, "green"):
                                self._save_checkpoint()

                        with marked_timer("update_weights", timing_raw, "red"):
                            self.checkpoint_manager.update_weights()
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                    elif self._current_aware_router_mismatch_rs_enabled():
                        # No actor update occurs during critic warmup. Release the
                        # cached rollout route tensor retained for RDC.
                        batch.batch.pop("routed_experts", None)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw.get("step", 0)
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                if self.config.trainer.get("log_epoch_number", False):
                    metrics["training/epoch_number"] = epoch + 1
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                metrics.update(
                    {
                        "train/filter_groups/groups_total": filter_groups_stats["groups_total"],
                        "train/filter_groups/groups_kept": filter_groups_stats["groups_kept"],
                        "train/filter_groups/groups_filtered": filter_groups_stats["groups_filtered"],
                        "train/filter_groups/groups_filtered_all_correct": filter_groups_stats[
                            "groups_filtered_all_correct"
                        ],
                        "train/filter_groups/groups_filtered_all_wrong": filter_groups_stats[
                            "groups_filtered_all_wrong"
                        ],
                        "train/filter_groups/groups_filtered_other_constant": filter_groups_stats[
                            "groups_filtered_other_constant"
                        ],
                    }
                )
                if filter_groups_stats["groups_total"] > 0:
                    metrics["train/filter_groups/kept_ratio"] = (
                        filter_groups_stats["groups_kept"] / filter_groups_stats["groups_total"]
                    )
                    metrics["train/filter_groups/filtered_all_correct_ratio"] = (
                        filter_groups_stats["groups_filtered_all_correct"] / filter_groups_stats["groups_total"]
                    )
                    metrics["train/filter_groups/filtered_all_wrong_ratio"] = (
                        filter_groups_stats["groups_filtered_all_wrong"] / filter_groups_stats["groups_total"]
                    )
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
                filter_groups_stats = defaultdict(int)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
        # check if last step checkpint exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)
