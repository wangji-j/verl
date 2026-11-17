"""CPU tests for the random sequence-drop ablation (RDC control arm).

The ablation must behave like RDC's rejection sampling in every respect except
which responses it picks: same intervention (zero ``response_mask`` rows, keep
the rows in the batch so the GRPO group baseline is untouched), same count rule
(``ceil(n_valid * fraction)``), uniform selection instead of drift-ranked.
"""

import math

import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.fully_async_policy.random_drop_ablation import (
    apply_random_sequence_drop,
    random_drop_enabled,
)


def _config(**overrides) -> OmegaConf:
    cfg = {"random_drop": {"enable": True, "fraction": 0.08, "rounding": "ceil", "count": None, "seed": 7}}
    cfg["random_drop"].update(overrides)
    return OmegaConf.create(cfg)


def _batch(n: int = 32, length: int = 10, valid_rows: int | None = None) -> DataProto:
    response_mask = torch.ones(n, length)
    if valid_rows is not None:
        # Mark the tail rows as degenerate (no unmasked tokens).
        response_mask[valid_rows:] = 0
    return DataProto.from_dict(
        tensors={
            "responses": torch.ones(n, length, dtype=torch.long),
            "response_mask": response_mask,
        }
    )


def test_disabled_is_noop():
    batch = _batch()
    cfg = _config(enable=False)
    assert not random_drop_enabled(cfg)
    metrics = apply_random_sequence_drop(batch, cfg, step=0)
    assert metrics == {}
    assert batch.batch["response_mask"].sum().item() == 32 * 10
    assert "random_drop_reject_mask" not in batch.batch


def test_drops_ceil_fraction_and_zeroes_only_those_rows():
    batch = _batch(n=32, length=10)
    metrics = apply_random_sequence_drop(batch, _config(fraction=0.08), step=1)

    expected_k = math.ceil(32 * 0.08)  # 3
    reject = batch.batch["random_drop_reject_mask"]
    assert int(reject.sum().item()) == expected_k
    assert metrics["random_drop/rejected_count"] == float(expected_k)
    assert metrics["random_drop/kept_count"] == float(32 - expected_k)
    assert metrics["random_drop/realized_fraction"] == pytest.approx(expected_k / 32)

    mask = batch.batch["response_mask"]
    # Dropped rows fully zeroed; kept rows fully intact.
    assert mask[reject].sum().item() == 0
    assert mask[~reject].sum().item() == (32 - expected_k) * 10
    # Rows are masked, never removed -- the GRPO baseline still sees every response.
    assert mask.shape[0] == 32


def test_ignores_rows_with_no_valid_tokens():
    # 20 valid rows out of 32: k must be computed over the valid ones only, and a
    # degenerate row must never be counted as a "drop".
    batch = _batch(n=32, length=10, valid_rows=20)
    metrics = apply_random_sequence_drop(batch, _config(fraction=0.10), step=2)

    reject = batch.batch["random_drop_reject_mask"]
    assert metrics["random_drop/valid_count"] == 20.0
    assert int(reject.sum().item()) == math.ceil(20 * 0.10)
    assert not bool(reject[20:].any().item())


def test_seed_makes_selection_reproducible_and_step_varies_it():
    def run(step: int, seed: int) -> torch.Tensor:
        batch = _batch(n=64)
        apply_random_sequence_drop(batch, _config(fraction=0.25, seed=seed), step=step)
        return batch.batch["random_drop_reject_mask"]

    assert torch.equal(run(5, 7), run(5, 7))  # replayable across a resume
    assert not torch.equal(run(5, 7), run(6, 7))  # a fresh subset each step
    assert not torch.equal(run(5, 7), run(5, 8))  # independent arms differ


def test_selection_is_uniform_over_positions():
    # Each position should be dropped about equally often; a length- or
    # position-correlated selection would fail this.
    n, steps, fraction = 32, 4000, 0.25
    counts = torch.zeros(n)
    for step in range(steps):
        batch = _batch(n=n)
        apply_random_sequence_drop(batch, _config(fraction=fraction, seed=11), step=step)
        counts += batch.batch["random_drop_reject_mask"].float()

    expected = steps * fraction
    # Binomial sd = sqrt(steps * p * (1-p)) ~= 27 here; 5 sd is a wide band that
    # still catches any systematic bias.
    assert torch.all((counts - expected).abs() < 5 * math.sqrt(steps * fraction * (1 - fraction)))


def test_stochastic_rounding_is_unbiased():
    # ceil(32 * 0.08) = 3 is a realized 9.4%; stochastic rounding should average
    # to the requested 2.56 instead.
    n, steps, fraction = 32, 3000, 0.08
    total = 0
    for step in range(steps):
        batch = _batch(n=n)
        m = apply_random_sequence_drop(batch, _config(fraction=fraction, rounding="stochastic", seed=3), step=step)
        total += int(m["random_drop/rejected_count"])
    assert total / steps == pytest.approx(n * fraction, abs=0.1)


def test_explicit_count_overrides_fraction():
    batch = _batch(n=32)
    metrics = apply_random_sequence_drop(batch, _config(fraction=0.5, count=4), step=0)
    assert metrics["random_drop/rejected_count"] == 4.0


def test_rejects_dropping_every_response():
    batch = _batch(n=8)
    with pytest.raises(RuntimeError, match="mask all"):
        apply_random_sequence_drop(batch, _config(fraction=1.0), step=0)


def test_unknown_rounding_mode_raises():
    batch = _batch(n=8)
    with pytest.raises(ValueError, match="rounding"):
        apply_random_sequence_drop(batch, _config(rounding="floor"), step=0)


def test_score_metrics_use_passed_reward_tensor():
    # At hook time the async trainer has not written token_level_scores onto the
    # batch yet, so the reward tensor arrives as an argument.
    batch = _batch(n=32)
    reward = torch.zeros(32, 10)
    reward[:, -1] = 1.0
    metrics = apply_random_sequence_drop(batch, _config(fraction=0.25), step=0, reward_tensor=reward)
    assert metrics["random_drop/score_before_mean"] == pytest.approx(1.0)
    assert metrics["random_drop/rejected_positive_fraction"] == pytest.approx(1.0)
    assert metrics["random_drop/kept_positive_fraction"] == pytest.approx(1.0)
