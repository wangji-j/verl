from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from dapo.dapo_ray_trainer import RayDAPOTrainer
from verl import DataProto


def test_current_aware_static_probe_keeps_behavior_routes_on_driver():
    trainer = object.__new__(RayDAPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "router": {"enable_current_aware_mismatch_rs": True},
            "actor_rollout_ref": {
                "actor": {
                    "loss_agg_mode": "token-mean",
                    "loss_scale_factor": None,
                }
            },
        }
    )
    trainer.router_mismatch_metrics_enabled = True
    trainer.use_reference_policy = True

    behavior_routes = torch.zeros(2, 3, 1, 1, dtype=torch.uint8)
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.ones(2, 3, dtype=torch.long),
            "routed_experts": behavior_routes.clone(),
        }
    )
    events = []

    def probe_theta_zero(probe_batch):
        assert "routed_experts" not in probe_batch.batch
        events.append("actor_probe_without_behavior_routes")
        return (
            DataProto.from_dict(
                tensors={
                    "old_log_probs": torch.zeros(2, 3),
                    "entropys": torch.ones(2, 3),
                    "routed_experts": torch.ones(2, 3, 1, 1, dtype=torch.uint8),
                }
            ),
            0.5,
        )

    def compute_mismatch(mismatch_batch, current_output):
        torch.testing.assert_close(mismatch_batch.batch["routed_experts"], behavior_routes)
        assert "routed_experts" in current_output.batch
        events.append("compare_after_behavior_routes_restored")
        return SimpleNamespace(metrics={"router/rollout_vs_train/seq_mismatch_mean": 0.5})

    def compute_ref(ref_batch):
        assert "routed_experts" not in ref_batch.batch
        events.append("ref_without_behavior_routes")
        return DataProto.from_dict(tensors={"ref_log_prob": torch.zeros(2, 3)})

    def fail_static_filter(*_args, **_kwargs):
        raise AssertionError("static rejection must be skipped in current-aware mode")

    trainer._compute_old_log_prob = probe_theta_zero
    trainer._compute_router_mismatch_result = compute_mismatch
    trainer._compute_ref_log_prob = compute_ref
    trainer._maybe_dump_router_analysis = lambda *_args, **_kwargs: None
    trainer._cuda_memory_snapshot = lambda *_args, **_kwargs: {}
    trainer._write_perf_debug = lambda *_args, **_kwargs: None
    trainer._apply_router_mismatch_rs = fail_static_filter

    metrics = {}
    with patch("dapo.dapo_ray_trainer.compute_response_mask", return_value=torch.ones(2, 3)):
        result = trainer.compute_kl_related_metrics(batch, metrics, defaultdict(float))

    assert events == [
        "actor_probe_without_behavior_routes",
        "compare_after_behavior_routes_restored",
        "ref_without_behavior_routes",
    ]
    torch.testing.assert_close(result.batch["routed_experts"], behavior_routes)
    assert "old_log_probs" in result.batch
    assert "ref_log_prob" in result.batch
    assert metrics["router/rollout_vs_train/seq_mismatch_mean"] == 0.5


def test_current_aware_router_filter_probes_then_updates_each_nonoverlapping_minibatch():
    trainer = object.__new__(RayDAPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": 4},
            "router": {
                "mismatch_rs_fraction": 0.25,
                "mismatch_rs_length_bucket_edges": [2],
            },
            "actor_rollout_ref": {
                "rollout": {"n": 2},
                "actor": {"ppo_mini_batch_size": 2},
            },
        }
    )

    sample_ids = torch.arange(8)
    batch = DataProto.from_dict(
        tensors={
            "sample_ids": sample_ids,
            "response_mask": torch.ones(8, 3),
            "routed_experts": torch.zeros(8, 3, 1, 1, dtype=torch.uint8),
        }
    )
    events = []
    update_index = 0

    def probe_current_routes(mini_batch):
        assert "routed_experts" not in mini_batch.batch
        events.append(("probe", update_index, mini_batch.batch["sample_ids"].tolist()))
        return (
            DataProto.from_dict(
                tensors={
                    "routed_experts": torch.ones(
                        len(mini_batch),
                        mini_batch.batch["response_mask"].shape[-1],
                        1,
                        1,
                        dtype=torch.uint8,
                    )
                }
            ),
            0.5,
        )

    def compute_mismatch(mini_batch, current_actor_output):
        assert "routed_experts" in mini_batch.batch
        assert "routed_experts" in current_actor_output.batch
        return SimpleNamespace(metrics={"router/rollout_vs_train/seq_mismatch_mean": 0.5})

    def apply_filter(mini_batch, _router_result):
        reject = torch.zeros(len(mini_batch), dtype=torch.bool)
        reject[0] = True
        mini_batch.batch["router_mismatch_reject_mask"] = reject
        mini_batch.batch["response_mask"][reject] = 0
        return {
            "router/rollout_vs_train/rs_rejected_count": 1.0,
            "router/rollout_vs_train/rs_kept_count": 3.0,
            "router/rollout_vs_train/rs_bucket_0_count": 4.0,
            "router/rollout_vs_train/rs_bucket_0_rejected_count": 1.0,
        }

    def update_actor(mini_batch):
        nonlocal update_index
        assert "routed_experts" not in mini_batch.batch
        assert mini_batch.meta_info["update_lr_scheduler_at_end"] is (update_index == 1)
        events.append(("update", update_index, mini_batch.batch["sample_ids"].tolist()))
        update_index += 1
        return DataProto.from_single_dict(
            data={},
            meta_info={"metrics": {"actor/pg_loss": [float(update_index)]}},
        )

    trainer._compute_old_log_prob = probe_current_routes
    trainer._compute_router_mismatch_result = compute_mismatch
    trainer._apply_router_mismatch_rs = apply_filter
    trainer._update_actor = update_actor
    trainer.actor_rollout_wg = object()
    trainer._get_dp_size = lambda *_args, **_kwargs: 2

    metrics = {}
    actor_output = trainer._update_actor_with_current_aware_router_filter(
        batch,
        metrics,
        defaultdict(float),
    )

    assert events == [
        ("probe", 0, [0, 1, 4, 5]),
        ("update", 0, [0, 1, 4, 5]),
        ("probe", 1, [2, 3, 6, 7]),
        ("update", 1, [2, 3, 6, 7]),
    ]
    assert "routed_experts" not in batch.batch
    torch.testing.assert_close(
        batch.batch["response_mask"].sum(dim=-1),
        torch.tensor([0.0, 3.0, 0.0, 3.0, 3.0, 3.0, 3.0, 3.0]),
    )
    assert metrics["router/rdc/mini_steps"] == 2.0
    assert metrics["router/rdc/actor_dp_size"] == 2.0
    assert metrics["router/rdc/mini_response_batch_size_per_dp"] == 2.0
    assert metrics["router/rdc/rejected_count"] == 2.0
    assert metrics["router/rdc/rejected_fraction"] == 0.25
    assert metrics["router/rdc/mini_step_1/filter/total_response_count"] == 4.0
    assert metrics["router/rdc/mini_step_1/filter/filtered_response_count"] == 1.0
    assert metrics["router/rdc/mini_step_1/filter/kept_response_count"] == 3.0
    assert metrics["router/rdc/mini_step_1/filter/actual_rejected_fraction"] == 0.25
    assert metrics["router/rdc/mini_step_1/filter/valid_token_count_before"] == 12.0
    assert metrics["router/rdc/mini_step_1/filter/valid_token_count_after"] == 9.0
    assert metrics["router/rdc/mini_step_1/filter/rejected_token_count"] == 3.0
    assert metrics["router/rdc/mini_step_1/update_completed"] == 1.0
    assert metrics["router/rdc/mini_step_1/cumulative_filtered_response_count"] == 1.0
    assert metrics["router/rdc/mini_step_2/filter/filtered_response_count"] == 1.0
    assert metrics["router/rdc/mini_step_2/cumulative_filtered_response_count"] == 2.0
    assert metrics["router/rdc/rejected_token_count"] == 6.0
    assert metrics["router/rdc/rejected_token_fraction"] == 0.25
    assert metrics["router/rdc/rs_bucket_0_rejected_fraction"] == 0.25
    assert metrics["router/rdc/mini_step_2/seq_mismatch_mean"] == 0.5
    assert actor_output.meta_info["metrics"]["actor/pg_loss"] == [1.0, 2.0]
