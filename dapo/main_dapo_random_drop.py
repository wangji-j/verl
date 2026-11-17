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
"""Sync/off-k entry point for the random sequence-drop ablation (TIS, no RDC, no R3).

Control arm for the off-k RDC experiments: drop a uniformly random 8% of
responses per training step instead of the highest-router-drift 8%. Same question
as the async arm -- is RDC's gain from removing 8% of sequences, or from removing
*those* 8%?

Where the mask goes, and why there:
``RayDAPOTrainer`` applies static RDC's mask inside ``compute_kl_related_metrics``
(``dapo_ray_trainer.py``, called from ``fit`` before ``compute_advantage``). This
subclass overrides that same method and applies the random mask at the same point,
so the two arms differ only in the selection rule. Masking must happen before
``compute_advantage``: GRPO multiplies the advantage by ``response_mask``
(``core_algos.py``, ``scores.unsqueeze(-1) * response_mask``) while the group
baseline is computed from raw scores, so a masked response still anchors its
group's baseline but contributes no gradient. That is exactly RDC's behavior.

off-k staleness needs no RDC machinery. It comes from ``ppo_mini_batch_size`` <
``train_batch_size``: the actor worker runs ``train/mini`` sequential optimizer
steps over one fixed rollout batch, so mini-step j trains at theta_(j-1) against
routes captured at theta_0. The current-aware RDC driver loop
(``_update_actor_with_current_aware_router_filter``) is only needed to *re-probe*
routes each mini-step; a random draw has nothing to re-probe. So this arm runs the
native update path, masks once per rollout batch, and every mini-step inherits
that mask. Dropping 8% of the full batch leaves each mini-batch with ~8% masked
(hypergeometric, unbiased), so per-mini-step masking would be equivalent in
expectation at far more plumbing.

Usage (see low_precision/off-policy-randomdrop8-TIS.sh)::

    python3 -m dapo.main_dapo_random_drop --config-name=dapo_megatron_trainer \
        +random_drop.enable=True +random_drop.fraction=0.08 ...
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl import DataProto

# The drop helper is engine-agnostic despite living under the async package; it
# only touches response_mask. Imported rather than duplicated so both arms share
# one selection rule and one set of tested semantics.
from verl.experimental.fully_async_policy.random_drop_ablation import (
    apply_random_sequence_drop,
    random_drop_enabled,
)
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler, run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device
from verl.utils.profiler import marked_timer

# Absolute, module-level imports on purpose. Ray re-imports this module in the
# worker under its top-level name (the traceback repr shows
# main_dapo_random_drop, not dapo.main_dapo_random_drop), so a relative import
# would not resolve there. Defining the trainer subclass at module level rather
# than inside a factory also keeps cloudpickle from having to ship the class by
# value -- which previously dropped DataProto from the worker's globals and blew
# up on the type annotations at class-creation time.
from dapo.dapo_ray_trainer import RayDAPOTrainer


class RandomDropDAPOTrainer(RayDAPOTrainer):
    """RayDAPOTrainer that drops a uniformly random fraction of each PPO mini-batch."""

    # Set while inside the mini-step loop, so the _update_actor override knows a
    # mini-batch (already masked) is being handed to the parent and must not be
    # split again. Guards against re-entering the loop recursively.
    _in_random_drop_loop = False

    def _update_actor(self, batch: DataProto) -> DataProto:
        """Intercept fit()'s single full-batch call and run the mini-step loop instead."""
        if self._in_random_drop_loop or not random_drop_enabled(self.config):
            return super()._update_actor(batch)
        metrics = getattr(self, "_random_drop_metrics", None)
        timing_raw = getattr(self, "_random_drop_timing_raw", None)
        if metrics is None or timing_raw is None:
            raise RuntimeError(
                "random_drop: _update_actor was called outside the instrumented fit() path; "
                "metrics/timing sinks are missing."
            )
        self._in_random_drop_loop = True
        try:
            return self._update_actor_with_random_drop(batch, metrics, timing_raw)
        finally:
            self._in_random_drop_loop = False

    def compute_kl_related_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict):
        # fit() does not pass its metrics/timing dicts into _update_actor, but it
        # does pass them here, earlier in the same training step. Capture the
        # references so the mini-step loop can record into the same dicts the
        # trainer flushes at the end of the step.
        self._random_drop_metrics = metrics
        self._random_drop_timing_raw = timing_raw
        return super().compute_kl_related_metrics(batch, metrics, timing_raw)

    def _update_actor_with_random_drop(self, batch: DataProto, metrics: dict, timing_raw: dict) -> DataProto:
        """Drop a random fraction inside each mini-batch, then update that mini-batch.

        Structural mirror of ``_update_actor_with_current_aware_router_filter``:
        same driver-side mini-step loop, same DP-aware index split, one actor RPC
        per mini-step, same LR-scheduler suppression. The only difference is the
        selection rule -- a uniform draw instead of a route-drift ranking. There is
        no route probe because a random draw has nothing to probe, which also makes
        this arm cheaper per mini-step.
        """
        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        mini_prompt_batch_size = int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size)
        mini_response_batch_size = mini_prompt_batch_size * rollout_n
        if len(batch) % mini_response_batch_size != 0:
            raise ValueError(
                "The response batch must split exactly into PPO mini-batches for per-mini-batch "
                f"random drop; got responses={len(batch)}, mini_response_batch_size={mini_response_batch_size}."
            )

        # The controller batch is rank-major after balancing/dispatch: all samples
        # for actor DP rank 0, then rank 1, and so on. Take one slice from every
        # rank partition per global mini-step to match the native worker iterator.
        # A plain contiguous split would make early mini-steps draw from only the
        # first rank partition when actor DP > 1. Copied from the RDC path so both
        # arms split identically.
        actor_dp_size = int(self._get_dp_size(self.actor_rollout_wg, "actor"))
        if actor_dp_size <= 0 or len(batch) % actor_dp_size != 0:
            raise ValueError(
                "The response batch must split evenly across actor DP ranks; "
                f"got responses={len(batch)}, actor_dp_size={actor_dp_size}."
            )
        if mini_response_batch_size % actor_dp_size != 0:
            raise ValueError(
                "Each mini-batch must split evenly across actor DP ranks; "
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

        metrics["random_drop/mini_steps"] = float(num_mini_steps)
        metrics["random_drop/mini_response_batch_size"] = float(mini_response_batch_size)
        metrics["random_drop/actor_dp_size"] = float(actor_dp_size)

        updated_response_mask = batch.batch["response_mask"].clone()
        total_rejected = 0.0
        total_valid = 0.0
        actor_output = None

        for mini_step, mini_indices in enumerate(mini_batch_indices, start=1):
            mini_batch = batch[mini_indices]

            with marked_timer("random_drop_filter", timing_raw, "blue"):
                # Seed per (global step, mini-step) so every mini-step draws an
                # independent subset and a resumed run replays the same draws.
                drop_metrics = apply_random_sequence_drop(
                    mini_batch,
                    self.config,
                    step=int(getattr(self, "global_steps", 0)) * 1000 + mini_step,
                    reward_tensor=mini_batch.batch.get("token_level_scores"),
                )

            prefix = f"random_drop/mini_step_{mini_step}"
            for key, value in drop_metrics.items():
                metrics[f"{prefix}/{key.split('/', 1)[1]}"] = value
            total_rejected += drop_metrics.get("random_drop/rejected_count", 0.0)
            total_valid += drop_metrics.get("random_drop/valid_count", 0.0)

            updated_response_mask[mini_indices] = mini_batch.batch["response_mask"]
            mini_batch.batch.pop("random_drop_reject_mask", None)
            # The native worker advances the LR scheduler once after all PPO
            # mini-batches in a rollout batch. One RPC per mini-step here, so
            # suppress advancement until the final RPC.
            mini_batch.meta_info["update_lr_scheduler_at_end"] = mini_step == num_mini_steps

            with marked_timer("random_drop_actor_update", timing_raw, "red"):
                actor_output = self._update_actor(mini_batch)
            metrics[f"{prefix}/update_completed"] = 1.0

        batch.batch["response_mask"] = updated_response_mask
        metrics["random_drop/total_rejected_count"] = total_rejected
        metrics["random_drop/total_valid_count"] = total_valid
        metrics["random_drop/total_realized_fraction"] = (
            total_rejected / total_valid if total_valid > 0 else 0.0
        )
        return actor_output


class RandomDropTaskRunner(TaskRunner):
    """DAPOTaskRunner with the random-drop trainer swapped in."""

    def run(self, config):
        from pprint import pprint

        from verl.utils.fs import copy_to_local

        print(f"[RANDOM-DROP] TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        if os.environ.get("DAPO_PRINT_CONFIG", "0") == "1":
            pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        print(f"[RANDOM-DROP] Using trainer: {RandomDropDAPOTrainer.__name__}")
        trainer = RandomDropDAPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="config", config_name="dapo_trainer", version_base=None)
def main(config):
    if not random_drop_enabled(config):
        raise RuntimeError(
            "This entry point is the random-drop ablation; pass +random_drop.enable=True "
            "(or use dapo.main_dapo for an unmodified run)."
        )
    # Fail fast on anything that would make this a two-variable comparison.
    for key in ("router.enable_mismatch_rs", "router.enable_current_aware_mismatch_rs"):
        if bool(OmegaConf.select(config, key, default=False)):
            raise RuntimeError(f"random_drop is the no-RDC control arm, but {key}=True. Disable RDC.")
    if bool(OmegaConf.select(config, "actor_rollout_ref.rollout.enable_rollout_routing_replay", default=False)):
        raise RuntimeError("random_drop is the no-R3 control arm, but enable_rollout_routing_replay=True.")

    # The hook assumes the use_kl_in_reward=False call site. RayDAPOTrainer calls
    # compute_kl_related_metrics from two places: with use_kl_in_reward=True it runs
    # early, on the pre-filter `new_batch` and before token_level_scores exists;
    # with False it runs once on the assembled batch just before compute_advantage,
    # which is where static RDC masks. Only the latter is the right anchor, so
    # refuse the other rather than silently masking the wrong batch.
    if bool(OmegaConf.select(config, "algorithm.use_kl_in_reward", default=False)):
        raise RuntimeError(
            "random_drop requires algorithm.use_kl_in_reward=False (the recipe default): with KL in "
            "reward, compute_kl_related_metrics runs at a different point in fit() and the mask would "
            "land on the wrong batch."
        )

    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    task_runner_num_cpus = int(os.environ.get("DAPO_TASK_RUNNER_NUM_CPUS", "32"))
    run_ppo(config, task_runner_class=ray.remote(num_cpus=task_runner_num_cpus)(RandomDropTaskRunner))


if __name__ == "__main__":
    main()
