"""CPU test for the static Router Drift Control (RDC) rejection-sampling hook
wired into the fully-async / disaggregated training path.

The hook lives in ``SeparateRayPPOTrainer._fit_compute_log_prob``: when
``router.enable_mismatch_rs`` is set and both rollout-side and train-side
``routed_experts`` are present, it compares routes and masks over-drifted
responses (zeroing ``response_mask`` rows) before ``batch.union(old_log_prob)``.
The RDC helpers themselves are inherited from ``RayPPOTrainer`` and are exercised
by their own tests; here we only assert the async-path wiring/ordering.
"""

from collections import defaultdict
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer


def _make_trainer(enable_rs: bool) -> SeparateRayPPOTrainer:
    trainer = object.__new__(SeparateRayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            # bypass_mode=False -> decoupled branch that recomputes old_log_prob
            # (the only path that yields train-side routed_experts for RDC).
            "algorithm": {"rollout_correction": {"bypass_mode": False}},
            "router": {
                "enable_mismatch_metrics": True,
                "enable_mismatch_rs": enable_rs,
            },
            "actor_rollout_ref": {
                "actor": {
                    "loss_agg_mode": "token-mean",
                    "loss_scale_factor": None,
                    "router_replay": {"mode": "disabled"},
                }
            },
        }
    )
    # NOTE: deliberately do NOT set trainer.router_mismatch_metrics_enabled here.
    # FullyAsyncTrainer re-implements __init__ without super().__init__(), so this
    # attribute is absent at runtime; the hook must lazily create it via
    # _ensure_router_mismatch_state(). Leaving it unset exercises that path.
    trainer.metrics = {}
    trainer.timing_raw = defaultdict(float)
    return trainer


def _make_batch() -> DataProto:
    return DataProto.from_dict(
        tensors={
            "responses": torch.ones(4, 3, dtype=torch.long),
            "response_mask": torch.ones(4, 3),
            # rollout-side routes captured at generation-time params
            "routed_experts": torch.zeros(4, 3, 1, 1, dtype=torch.uint8),
        }
    )


def _old_log_prob() -> DataProto:
    return DataProto.from_dict(
        tensors={
            "old_log_probs": torch.zeros(4, 3),
            "entropys": torch.ones(4, 3),
            # train-side routes from the actor forward
            "routed_experts": torch.ones(4, 3, 1, 1, dtype=torch.uint8),
        }
    )


def test_static_rdc_masks_and_logs_when_enabled():
    trainer = _make_trainer(enable_rs=True)
    batch = _make_batch()
    events = []

    trainer._compute_old_log_prob = lambda _b: (_old_log_prob(), 0.5)

    def compute_mismatch(rollout_batch, old_log_prob):
        assert "routed_experts" in rollout_batch.batch
        assert "routed_experts" in old_log_prob.batch
        events.append("compute")
        return SimpleNamespace(metrics={"router/rollout_vs_train/seq_mismatch_mean": 0.5})

    def apply_rs(rollout_batch, _result):
        events.append("apply")
        reject = torch.zeros(len(rollout_batch), dtype=torch.bool)
        reject[0] = True
        rollout_batch.batch["router_mismatch_reject_mask"] = reject
        rollout_batch.batch["response_mask"][reject] = 0
        return {"router/rollout_vs_train/rs_rejected_count": 1.0}

    trainer._compute_router_mismatch_result = compute_mismatch
    trainer._apply_router_mismatch_rs = apply_rs

    result = trainer._fit_compute_log_prob(batch)

    # RDC ran, compute-then-apply, before the union.
    assert events == ["compute", "apply"]
    # The lazy state init fired (mirrors the FullyAsyncTrainer __init__ gap).
    assert trainer.router_mismatch_metrics_enabled is True
    assert trainer._router_mismatch_frozen_alignment is None
    # The over-drifted response (row 0) had its mask zeroed; others untouched.
    torch.testing.assert_close(
        result.batch["response_mask"].sum(dim=-1), torch.tensor([0.0, 3.0, 3.0, 3.0])
    )
    # RS metrics merged and the log-prob union completed.
    assert trainer.metrics["router/rollout_vs_train/rs_rejected_count"] == 1.0
    assert "old_log_probs" in result.batch


def test_static_rdc_is_noop_when_disabled():
    trainer = _make_trainer(enable_rs=False)
    batch = _make_batch()

    trainer._compute_old_log_prob = lambda _b: (_old_log_prob(), 0.5)

    def fail(*_args, **_kwargs):
        raise AssertionError("RDC must not run when router.enable_mismatch_rs=False")

    trainer._compute_router_mismatch_result = fail
    trainer._apply_router_mismatch_rs = fail

    result = trainer._fit_compute_log_prob(batch)

    # No masking happened; the normal log-prob path is intact.
    torch.testing.assert_close(
        result.batch["response_mask"].sum(dim=-1), torch.tensor([3.0, 3.0, 3.0, 3.0])
    )
    assert "old_log_probs" in result.batch
