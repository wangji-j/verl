# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Entry point for the fully-async router-lag probe (diagnostic, not a new method).

Training runs the normal async recipe (RDC+TIS). On top of it this trainer:

  1. In the first sync window after resume (param version ``V0``) caches a
     length-stratified probe set of responses generated purely by ``V0``, with
     their FP8 rollout routes and the BF16 train routes from the decoupled
     old_log_prob recompute (which runs at ``V0`` in that window).
  2. Before every update at local step ``s`` re-forwards the probe set at the
     current weights (lag ``s - 1`` updates) and records routing drift against
     both references. See ``router_lag_probe.py`` for the protocol and metrics.
  3. Logs each training sample's generation version so the real lag distribution
     of this sync interval can be reconstructed.

The probe forward calls ``RayPPOTrainer._compute_old_log_prob`` directly so it runs
at the *current* weights, bypassing the async override that restores ``V_1``. It
never touches the training batch.

Usage mirrors ``fully_async_main``::

    python3 -m verl.experimental.fully_async_policy.fully_async_router_lag_probe_main \
        --config-path=config --config-name=fully_async_ppo_megatron_trainer.yaml \
        +router_lag_probe.enable=True +router_lag_probe.output_dir=... ...
"""

import json
import os

import numpy as np
import ray
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.fully_async_policy.router_lag_probe import (
    METRIC_PREFIX,
    bucket_quotas,
    compare_routes,
    crop_routes_to_response,
    length_buckets,
    masked_mean_abs_diff,
    merge_comparisons,
    pick_alignment,
    probe_settings,
    router_lag_probe_enabled,
    select_probe_indices,
    summarize,
    update_lag,
)
from verl.experimental.separation.utils import create_resource_pool_manager
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import Role
from verl.utils.device import auto_set_device

# Fields the actor forward needs (left_right_2_no_padding + no_padding_2_padding).
_PROBE_BATCH_KEYS = ("input_ids", "attention_mask", "position_ids", "prompts", "responses", "response_mask")


def _undecorated(actor_class):
    """Recover the plain class behind a ``@ray.remote`` actor class, to subclass it."""
    return actor_class.__ray_metadata__.modified_class.__ray_actor_class__


class RouterLagProbeFullyAsyncTrainer(_undecorated(FullyAsyncTrainer)):
    """FullyAsyncTrainer that measures routing drift on a fixed probe set at every local step."""

    # ---- state -------------------------------------------------------------------
    def _rlp_init(self):
        if getattr(self, "_rlp_ready", False):
            return
        self._rlp = probe_settings(self.config)
        os.makedirs(self._rlp.output_dir, exist_ok=True)
        self._rlp_v0 = None
        self._rlp_step = 0  # local steps seen since resume (1-based after increment)
        self._rlp_closed = False
        self._rlp_quota = bucket_quotas(self._rlp.probe_size, len(self._rlp.length_bucket_edges) + 1)
        self._rlp_budget = self._rlp.probe_size
        self._rlp_chunks = []
        self._rlp_alignment = {}
        self._rlp_generator = torch.Generator().manual_seed(self._rlp.seed)
        self._rlp_capture = False
        self._rlp_stash = None
        self._rlp_next_id = 0
        self._rlp_ready = True
        print(f"[ROUTER-LAG-PROBE] settings={self._rlp}")

    # ---- capture V0 train routes from the decoupled recompute --------------------
    def _compute_old_log_prob(self, batch: DataProto):
        old_log_prob, mfu = super()._compute_old_log_prob(batch)
        if getattr(self, "_rlp_capture", False):
            routes = old_log_prob.batch.get("routed_experts", None)
            if routes is None:
                raise RuntimeError("router_lag_probe needs actor routed_experts; set router.enable_mismatch_metrics=True")
            # Keep references only; _fit_compute_log_prob pops the key from the batch afterwards.
            self._rlp_stash = {"routes": routes, "log_probs": old_log_prob.batch["old_log_probs"]}
        return old_log_prob, mfu

    def _fit_compute_log_prob(self, batch: DataProto) -> DataProto:
        if not router_lag_probe_enabled(self.config):
            return super()._fit_compute_log_prob(batch)
        self._rlp_init()
        self._rlp_step += 1
        if self._rlp_v0 is None:
            self._rlp_v0 = int(self.current_param_version)
            if int(self.local_trigger_step) != 1:
                print(
                    f"[ROUTER-LAG-PROBE] WARNING: first step has local_trigger_step={self.local_trigger_step}; "
                    "lag bookkeeping assumes a fresh sync window"
                )

        snapshot = self._rlp_snapshot(batch)
        in_window = not self._rlp_closed and int(self.current_param_version) == self._rlp_v0
        self._rlp_capture = in_window
        self._rlp_stash = None
        try:
            batch = super()._fit_compute_log_prob(batch)
        finally:
            self._rlp_capture = False

        self._rlp_log_lag_record(snapshot)
        if in_window:
            self._rlp_cache(batch, snapshot)
            if int(self.local_trigger_step) == int(self.trigger_parameter_sync_step):
                self._rlp_close()
        elif not self._rlp_closed:
            self._rlp_close()
        self._rlp_stash = None

        lag = self._rlp_step - 1
        if self._rlp_chunks and lag <= self._rlp.max_lag:
            self._rlp_probe(lag)
        return batch

    # ---- bookkeeping -------------------------------------------------------------
    def _rlp_snapshot(self, batch: DataProto) -> dict:
        def versions(key):
            raw = batch.non_tensor_batch.get(key)
            if raw is None:
                return np.full(len(batch), -1, dtype=np.int64)
            return np.array([-1 if v is None else int(v) for v in raw], dtype=np.int64)

        response_mask = batch.batch["response_mask"]
        return {
            "vmin": versions("min_global_steps"),
            "vmax": versions("max_global_steps"),
            # RDC zeroes rejected rows of response_mask in place; keep the original.
            "response_mask": response_mask.clone(),
            "lengths": response_mask.sum(dim=-1).long(),
            "rollout_routes": batch.batch.get("routed_experts", None),
            "rollout_log_probs": batch.batch.get("rollout_log_probs", None),
        }

    def _rlp_log_lag_record(self, snapshot: dict):
        sync = int(self.trigger_parameter_sync_step)
        record = {
            "step": self._rlp_step,
            "param_version": int(self.current_param_version),
            "local_trigger_step": int(self.local_trigger_step),
            "sync_interval": sync,
            "v0": self._rlp_v0,
            "vmin": snapshot["vmin"].tolist(),
            "vmax": snapshot["vmax"].tolist(),
            "lag_oldest": update_lag(self._rlp_step, snapshot["vmin"], self._rlp_v0, sync).tolist(),
            "lag_newest": update_lag(self._rlp_step, snapshot["vmax"], self._rlp_v0, sync).tolist(),
            "response_length": snapshot["lengths"].tolist(),
        }
        with open(os.path.join(self._rlp.output_dir, "lag_records.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")

    def _rlp_cache(self, batch: DataProto, snapshot: dict):
        if self._rlp_budget <= 0:
            return
        if self._rlp_stash is None or snapshot["rollout_routes"] is None:
            raise RuntimeError("router_lag_probe: missing rollout or train routes in the V0 window")
        lengths = snapshot["lengths"]
        eligible = (
            torch.from_numpy(snapshot["vmin"] == self._rlp_v0)
            & torch.from_numpy(snapshot["vmax"] == self._rlp_v0)
            & (lengths > 0)
        )
        last_step_of_window = int(self.local_trigger_step) == int(self.trigger_parameter_sync_step)
        idx, taken = select_probe_indices(
            eligible,
            length_buckets(lengths, self._rlp.length_bucket_edges),
            self._rlp_quota,
            self._rlp_budget,
            batch_multiple=self._rlp.batch_multiple,
            backfill=last_step_of_window,
            generator=self._rlp_generator,
        )
        self.metrics[f"{METRIC_PREFIX}/eligible_count"] = float(eligible.sum())
        if idx.numel() == 0:
            return

        response_mask = snapshot["response_mask"][idx]
        response_len = int(response_mask.shape[1])
        keys = [k for k in _PROBE_BATCH_KEYS if k in batch.batch.keys()]
        probe_batch = batch.select(batch_keys=keys, non_tensor_batch_keys=[]).select_idxs(idx)
        probe_batch.batch["response_mask"] = response_mask.clone()
        probe_batch.meta_info = dict(batch.meta_info)

        rollout_log_probs = snapshot["rollout_log_probs"]
        self._rlp_chunks.append(
            {
                "ids": torch.arange(self._rlp_next_id, self._rlp_next_id + idx.numel()),
                "batch": probe_batch,
                "response_mask": response_mask,
                "lengths": lengths[idx].clone(),
                "mu_routes": crop_routes_to_response(snapshot["rollout_routes"][idx], response_len),
                "pi0_routes": crop_routes_to_response(self._rlp_stash["routes"][idx], response_len),
                "mu_log_probs": rollout_log_probs[idx].float().clone() if rollout_log_probs is not None else None,
                "pi0_log_probs": self._rlp_stash["log_probs"][idx].float().clone(),
            }
        )
        self._rlp_next_id += int(idx.numel())
        self._rlp_budget -= int(idx.numel())
        self._rlp_quota = [max(q - t, 0) for q, t in zip(self._rlp_quota, taken, strict=True)]
        print(f"[ROUTER-LAG-PROBE] step {self._rlp_step}: cached {idx.numel()} (per bucket {taken}), "
              f"budget left {self._rlp_budget}")

    def _rlp_close(self):
        self._rlp_closed = True
        total = sum(int(c["ids"].numel()) for c in self._rlp_chunks)
        lengths = torch.cat([c["lengths"] for c in self._rlp_chunks]) if self._rlp_chunks else torch.zeros(0)
        meta = {
            "v0": self._rlp_v0,
            "sync_interval": int(self.trigger_parameter_sync_step),
            "probe_size": total,
            "length_bucket_edges": list(self._rlp.length_bucket_edges),
            "bucket_counts": torch.bincount(
                length_buckets(lengths.long(), self._rlp.length_bucket_edges),
                minlength=len(self._rlp.length_bucket_edges) + 1,
            ).tolist()
            if total
            else [],
            "lengths": lengths.tolist(),
        }
        with open(os.path.join(self._rlp.output_dir, "probe_set.json"), "w") as f:
            json.dump(meta, f)
        print(f"[ROUTER-LAG-PROBE] probe set closed: {total} responses, buckets {meta['bucket_counts']}")

    # ---- probe -------------------------------------------------------------------
    def _rlp_probe(self, lag: int):
        s = self._rlp
        mu_parts, pi0_parts, lp_mu, lp_pi0 = [], [], [], []
        for chunk in self._rlp_chunks:
            probe_batch = chunk["batch"]
            probe_batch.meta_info["global_token_num"] = torch.sum(
                probe_batch.batch["attention_mask"], dim=-1
            ).tolist()
            # Plain recompute at the current weights (no V_1 restore).
            out, _ = RayPPOTrainer._compute_old_log_prob(self, probe_batch)
            mask = chunk["response_mask"]
            cur_routes = crop_routes_to_response(out.batch["routed_experts"], int(mask.shape[1]))
            for name, ref in (("mu", chunk["mu_routes"]), ("pi0", chunk["pi0_routes"])):
                if name not in self._rlp_alignment:
                    self._rlp_alignment[name] = pick_alignment(
                        ref, cur_routes, mask, num_experts=s.num_experts, smoothing_tau=s.smoothing_tau
                    )
                    print(f"[ROUTER-LAG-PROBE] frozen alignment {name}: {self._rlp_alignment[name]}")
            common = dict(num_experts=s.num_experts, smoothing_tau=s.smoothing_tau, chunk=s.metric_chunk)
            mu_parts.append(compare_routes(chunk["mu_routes"], cur_routes, mask,
                                           alignment=self._rlp_alignment["mu"], **common))
            pi0_parts.append(compare_routes(chunk["pi0_routes"], cur_routes, mask,
                                            alignment=self._rlp_alignment["pi0"], **common))
            cur_lp = out.batch["old_log_probs"].float()
            if chunk["mu_log_probs"] is not None:
                lp_mu.append(masked_mean_abs_diff(cur_lp, chunk["mu_log_probs"], mask))
            lp_pi0.append(masked_mean_abs_diff(cur_lp, chunk["pi0_log_probs"], mask))
            del out, cur_routes

        mu = merge_comparisons(mu_parts)
        pi0 = merge_comparisons(pi0_parts)
        record = {
            "lag": lag,
            "step": self._rlp_step,
            "param_version": int(self.current_param_version),
            "local_trigger_step": int(self.local_trigger_step),
            "v0": self._rlp_v0,
            "alignment": dict(self._rlp_alignment),
            "probe_ids": torch.cat([c["ids"] for c in self._rlp_chunks]),
            "lengths": torch.cat([c["lengths"] for c in self._rlp_chunks]),
            "mu_vs_cur": mu,
            "pi0_vs_cur": pi0,
            "logprob_absdiff_mu": torch.cat(lp_mu) if lp_mu else None,
            "logprob_absdiff_pi0": torch.cat(lp_pi0),
        }
        torch.save(record, os.path.join(s.output_dir, f"probe_lag{lag:04d}.pt"))

        metrics = {f"{METRIC_PREFIX}/lag": float(lag), f"{METRIC_PREFIX}/probe_count": float(record["lengths"].numel())}
        for name, comp in (("mu_vs_cur", mu), ("pi0_vs_cur", pi0)):
            metrics.update({f"{METRIC_PREFIX}/{name}/{k}": v for k, v in summarize(comp).items()})
        metrics[f"{METRIC_PREFIX}/pi0_vs_cur/logprob_absdiff_mean"] = float(record["logprob_absdiff_pi0"].mean())
        if record["logprob_absdiff_mu"] is not None:
            metrics[f"{METRIC_PREFIX}/mu_vs_cur/logprob_absdiff_mean"] = float(record["logprob_absdiff_mu"].mean())
        self.metrics.update(metrics)
        print(
            f"[ROUTER-LAG-PROBE] lag {lag}: n={metrics[f'{METRIC_PREFIX}/probe_count']:.0f} "
            f"D(mu,cur)={metrics[f'{METRIC_PREFIX}/mu_vs_cur/d_route_mean']:.5f} "
            f"D(pi0,cur)={metrics[f'{METRIC_PREFIX}/pi0_vs_cur/d_route_mean']:.5f} "
            f"flip(mu,cur)={metrics[f'{METRIC_PREFIX}/mu_vs_cur/flip_rate_mean']:.4f}"
        )


RouterLagProbeFullyAsyncTrainerActor = ray.remote(num_cpus=10)(RouterLagProbeFullyAsyncTrainer)


class RouterLagProbeTaskRunner(_undecorated(FullyAsyncTaskRunner)):
    """TaskRunner that swaps in the router-lag probe trainer. Identical otherwise."""

    def _create_trainer(self, config) -> None:
        print("[ROUTER-LAG-PROBE MAIN] Creating RouterLagProbeFullyAsyncTrainer...")
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }

        trainer = RouterLagProbeFullyAsyncTrainerActor.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            device_name=config.trainer.device,
        )

        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[ROUTER-LAG-PROBE MAIN] RouterLagProbeFullyAsyncTrainer created and initialized successfully")


RouterLagProbeTaskRunnerActor = ray.remote(num_cpus=1)(RouterLagProbeTaskRunner)


def main():
    # Hydra is invoked programmatically (not via @hydra.main) so this module can
    # reuse fully_async_main's config directory without shadowing its own.
    import hydra

    @hydra.main(config_path="config", config_name="fully_async_ppo_trainer", version_base=None)
    def _run(config):
        from verl.experimental.reward_loop import migrate_legacy_reward_impl
        from verl.trainer.main_ppo import run_ppo

        if not hasattr(config, "async_training"):
            raise RuntimeError("must set async_training config")
        if not router_lag_probe_enabled(config):
            raise RuntimeError(
                "This entry point is the router-lag probe; pass +router_lag_probe.enable=True "
                "(or use fully_async_main for an unmodified run)."
            )
        probe_settings(config)  # validate early, on the driver
        if not bool(OmegaConf.select(config, "router.enable_mismatch_metrics", default=False)):
            raise RuntimeError("router_lag_probe needs router.enable_mismatch_metrics=True (train-side routes)")
        if not bool(OmegaConf.select(config, "actor_rollout_ref.rollout.enable_return_routed_experts", default=False)):
            raise RuntimeError("router_lag_probe needs actor_rollout_ref.rollout.enable_return_routed_experts=True")
        if bool(OmegaConf.select(config, "algorithm.rollout_correction.bypass_mode", default=False)):
            raise RuntimeError("router_lag_probe needs bypass_mode=False (V0 train routes come from the recompute)")

        from time import time

        start_time = time()
        auto_set_device(config)
        config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
        config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
        config = migrate_legacy_reward_impl(config)
        run_ppo(config, task_runner_class=RouterLagProbeTaskRunnerActor)
        print(f"total time: {time() - start_time:.2f} seconds")

    _run()


if __name__ == "__main__":
    main()
