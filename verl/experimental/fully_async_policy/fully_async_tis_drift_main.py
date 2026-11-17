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
"""Entry point for the fully-async probability-drift-drop ablation (TIS, no RDC).

Second control arm for RDC: rank responses by rollout-vs-train **probability**
drift (the log-ratio TIS itself truncates) and drop the top ``ceil(n*fraction)``
within each length bucket -- instead of RDC's router-drift ranking or the
random-drop arm's uniform draw. TIS still corrects every surviving token; the
drop only zeroes ``response_mask`` rows. See ``tis_drift_ablation.py`` for the
score definitions (A: TIS-truncated fraction, B: mean |log-ratio|) and config.

Anchor in the async setting: there is no per-mini-step loop, so the score uses the
train-side ``old_log_probs`` recompute already on the batch -- exactly the anchor
the static async RDC arm (``async_RDC_TIS_lengthbucket.sh``) compares against. This
mirrors ``fully_async_random_drop_main.py``: subclass ``FullyAsyncTrainer`` and
override ``_fit_compute_log_prob`` so the mask lands at the same point RDC would.

Usage mirrors ``fully_async_main``::

    python3 -m verl.experimental.fully_async_policy.fully_async_tis_drift_main \
        --config-path=config --config-name=fully_async_ppo_megatron_trainer.yaml \
        +tis_drift.enable=True +tis_drift.score=trunc_frac +tis_drift.fraction=0.08 ...
"""

import ray
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.fully_async_policy.tis_drift_ablation import (
    apply_tis_drift_bucket_drop,
    compute_tis_drift_seq_score,
    tis_drift_enabled,
)
from verl.experimental.separation.utils import create_resource_pool_manager
from verl.trainer.ppo.utils import Role
from verl.utils.device import auto_set_device


def _undecorated(actor_class):
    """Recover the plain class behind a ``@ray.remote`` actor class, to subclass it."""
    return actor_class.__ray_metadata__.modified_class.__ray_actor_class__


class TisDriftFullyAsyncTrainer(_undecorated(FullyAsyncTrainer)):
    """FullyAsyncTrainer that masks the highest probability-drift responses each step."""

    def _fit_compute_log_prob(self, batch: DataProto) -> DataProto:
        # Run the normal path first (old_log_prob recompute, TIS bookkeeping, and the
        # RDC hook -- inert here because router.enable_mismatch_rs=False), then rank
        # by probability drift and mask at the same point RDC would have masked.
        batch = super()._fit_compute_log_prob(batch)
        if tis_drift_enabled(self.config):
            # Async anchor: the train-side old_log_probs already on the batch (the
            # static RDC anchor). compute_tis_drift_seq_score reads old_log_probs
            # when probe_log_probs is omitted.
            seq_score, seq_valid_token_count = compute_tis_drift_seq_score(batch, self.config)
            drop_metrics = apply_tis_drift_bucket_drop(
                batch,
                self.config,
                seq_score,
                seq_valid_token_count,
                reward_tensor=getattr(self, "reward_tensor", None),
            )
            if drop_metrics:
                self.metrics.update(drop_metrics)
        return batch


TisDriftFullyAsyncTrainerActor = ray.remote(num_cpus=10)(TisDriftFullyAsyncTrainer)


class TisDriftTaskRunner(_undecorated(FullyAsyncTaskRunner)):
    """TaskRunner that swaps in the probability-drift trainer. Identical otherwise."""

    def _create_trainer(self, config) -> None:
        print("[TIS-DRIFT MAIN] Creating TisDriftFullyAsyncTrainer...")
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }

        trainer = TisDriftFullyAsyncTrainerActor.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            device_name=config.trainer.device,
        )

        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[TIS-DRIFT MAIN] TisDriftFullyAsyncTrainer created and initialized successfully")


TisDriftTaskRunnerActor = ray.remote(num_cpus=1)(TisDriftTaskRunner)


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
        if not tis_drift_enabled(config):
            raise RuntimeError(
                "This entry point is the probability-drift ablation; pass +tis_drift.enable=True "
                "(or use fully_async_main for an unmodified run)."
            )
        # Fail fast on the combination that would silently make this run a
        # two-variable comparison instead of a clean control arm.
        if bool(OmegaConf.select(config, "router.enable_mismatch_rs", default=False)) or bool(
            OmegaConf.select(config, "router.enable_current_aware_mismatch_rs", default=False)
        ):
            raise RuntimeError(
                "tis_drift is the no-RDC control arm, but router.enable_mismatch_rs / "
                "enable_current_aware_mismatch_rs is True. Disable RDC to keep the ablation clean."
            )

        from time import time

        start_time = time()
        auto_set_device(config)
        config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
        config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
        config = migrate_legacy_reward_impl(config)
        run_ppo(config, task_runner_class=TisDriftTaskRunnerActor)
        print(f"total time: {time() - start_time:.2f} seconds")

    _run()


if __name__ == "__main__":
    main()
