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
"""Entry point for the fully-async random-sequence-drop ablation (TIS, no RDC).

Control arm for RDC: drop a uniformly random subset of responses per training
step instead of the highest-router-drift ones. See ``random_drop_ablation.py``
for the ablation's design and config keys.

Everything lives in this file rather than in ``fully_async_main.py`` /
``fully_async_trainer.py`` so concurrent runs on the shared checkout are not
disturbed. The mechanism: ``FullyAsyncTrainer`` inherits
``_fit_compute_log_prob`` from ``SeparateRayPPOTrainer``, so we subclass it and
override that one method. Ray pickles actor classes by module reference, so the
subclass must be importable here (a driver-side monkeypatch would not reach the
actor process).

Usage mirrors ``fully_async_main``::

    python3 -m verl.experimental.fully_async_policy.fully_async_random_drop_main \
        --config-path=config --config-name=fully_async_ppo_megatron_trainer.yaml \
        +random_drop.enable=True +random_drop.fraction=0.08 ...
"""

import ray
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.fully_async_policy.random_drop_ablation import (
    apply_random_sequence_drop,
    random_drop_enabled,
)
from verl.experimental.separation.utils import create_resource_pool_manager
from verl.trainer.ppo.utils import Role
from verl.utils.device import auto_set_device


def _undecorated(actor_class):
    """Recover the plain class behind a ``@ray.remote`` actor class, to subclass it."""
    return actor_class.__ray_metadata__.modified_class.__ray_actor_class__


class RandomDropFullyAsyncTrainer(_undecorated(FullyAsyncTrainer)):
    """FullyAsyncTrainer that masks a random subset of responses each step."""

    def _fit_compute_log_prob(self, batch: DataProto) -> DataProto:
        # Run the normal path first (old_log_prob recompute, TIS bookkeeping, and
        # the RDC hook -- inert here because router.enable_mismatch_rs=False), then
        # apply the random mask at the same point RDC would have applied its own.
        batch = super()._fit_compute_log_prob(batch)
        if random_drop_enabled(self.config):
            drop_metrics = apply_random_sequence_drop(
                batch,
                self.config,
                step=int(getattr(self, "global_steps", 0)),
                reward_tensor=getattr(self, "reward_tensor", None),
            )
            if drop_metrics:
                self.metrics.update(drop_metrics)
        return batch


RandomDropFullyAsyncTrainerActor = ray.remote(num_cpus=10)(RandomDropFullyAsyncTrainer)


class RandomDropTaskRunner(_undecorated(FullyAsyncTaskRunner)):
    """TaskRunner that swaps in the random-drop trainer. Identical otherwise."""

    def _create_trainer(self, config) -> None:
        print("[RANDOM-DROP MAIN] Creating RandomDropFullyAsyncTrainer...")
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }

        trainer = RandomDropFullyAsyncTrainerActor.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            device_name=config.trainer.device,
        )

        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[RANDOM-DROP MAIN] RandomDropFullyAsyncTrainer created and initialized successfully")


RandomDropTaskRunnerActor = ray.remote(num_cpus=1)(RandomDropTaskRunner)


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
        if not random_drop_enabled(config):
            raise RuntimeError(
                "This entry point is the random-drop ablation; pass +random_drop.enable=True "
                "(or use fully_async_main for an unmodified run)."
            )
        # Fail fast on the combination that would silently make this run a
        # two-variable comparison instead of a clean control arm.
        if bool(OmegaConf.select(config, "router.enable_mismatch_rs", default=False)) or bool(
            OmegaConf.select(config, "router.enable_current_aware_mismatch_rs", default=False)
        ):
            raise RuntimeError(
                "random_drop is the no-RDC control arm, but router.enable_mismatch_rs / "
                "enable_current_aware_mismatch_rs is True. Disable RDC to keep the ablation clean."
            )

        from time import time

        start_time = time()
        auto_set_device(config)
        config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
        config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
        config = migrate_legacy_reward_impl(config)
        run_ppo(config, task_runner_class=RandomDropTaskRunnerActor)
        print(f"total time: {time() - start_time:.2f} seconds")

    _run()


if __name__ == "__main__":
    main()
