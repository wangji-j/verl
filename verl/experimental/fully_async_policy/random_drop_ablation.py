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
"""Random sequence-drop ablation for the fully-async (disaggregated) recipe.

This is the *control arm* for RDC (Router Drift Control): instead of dropping the
responses with the largest rollout-vs-train router drift, drop a uniformly random
subset of the same size. If RDC's gain came only from "training on 8% fewer
sequences" (a batch-size / regularization effect), this control reproduces it.
If the gain came from *which* sequences are removed, this control does not.

Design constraints that keep the comparison honest -- the random arm must differ
from the RDC arm in the selection rule and nothing else:

1. Same intervention point. The mask is applied inside ``_fit_compute_log_prob``,
   right after the train-side ``old_log_prob`` forward, exactly where
   ``RayPPOTrainer._apply_router_mismatch_rs`` applies the RDC mask.
2. Same mechanism. We zero ``response_mask`` rows instead of deleting rows from
   the batch. So a dropped response still contributes its reward to the GRPO
   group baseline (``compute_grpo_outcome_advantage`` groups over raw scores and
   ignores ``response_mask``) but contributes no gradient. Physically removing
   rows would also shift the baseline and confound the ablation.
3. Same count rule. ``ceil(n_valid * fraction)`` clamped to ``[1, n_valid]``,
   matching ``top_fraction`` / ``length_bucket_top_fraction`` in
   ``verl/trainer/ppo/ray_trainer.py``. See ``rounding`` below for the caveat.

Nothing here reads router routes, so the run needs neither
``rollout.enable_return_routed_experts`` nor ``router.enable_mismatch_metrics``.

Config (all keys are additive, so pass them with a ``+`` on the CLI)::

    +random_drop.enable=True
    +random_drop.fraction=0.08
    +random_drop.rounding=ceil      # ceil | round | stochastic
    +random_drop.count=null         # int: exact per-step count, overrides fraction
    +random_drop.seed=1234

``rounding`` matters at small batch sizes. Fully-async trains on
``ppo_mini_batch_size * require_batches`` *sequences* per step (32 by default,
not 32 prompts), and ``ceil(32 * 0.08) = 3`` is a realized 9.4% rather than 8%.
``ceil`` is the default because it is what the RDC modes do -- keep it when the
point is to match an RDC run. Use ``stochastic`` (drop 2 or 3 with probability
weighted so the expectation is exactly 2.56) when the point is an unbiased 8%
over training. ``count`` pins the per-step number outright, which is the easiest
way to match a specific RDC run's observed ``rs_rejected_count``.
"""

import logging
import math
import os

import torch
from omegaconf import OmegaConf

from verl import DataProto

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

METRIC_PREFIX = "random_drop"


def random_drop_enabled(config) -> bool:
    return bool(OmegaConf.select(config, "random_drop.enable", default=False))


def _resolve_count(n_valid: int, config, generator: torch.Generator) -> tuple[int, float]:
    """Return ``(k, target_fraction)``: how many of ``n_valid`` responses to drop."""
    explicit = OmegaConf.select(config, "random_drop.count", default=None)
    fraction = float(OmegaConf.select(config, "random_drop.fraction", default=0.08))

    if explicit is not None:
        return min(max(int(explicit), 0), n_valid), fraction
    if fraction <= 0 or n_valid == 0:
        return 0, fraction

    rounding = str(OmegaConf.select(config, "random_drop.rounding", default="ceil")).lower()
    exact = n_valid * fraction
    if rounding == "ceil":
        # Matches top_fraction / length_bucket_top_fraction in ray_trainer.py.
        k = max(math.ceil(exact), 1)
    elif rounding == "round":
        k = max(round(exact), 1)
    elif rounding == "stochastic":
        # Unbiased: E[k] == n_valid * fraction exactly.
        floor_k = math.floor(exact)
        k = floor_k + int(torch.rand(1, generator=generator).item() < (exact - floor_k))
    else:
        raise ValueError(f"random_drop.rounding must be 'ceil', 'round', or 'stochastic', got {rounding!r}")
    return min(k, n_valid), fraction


def apply_random_sequence_drop(
    batch: DataProto,
    config,
    step: int,
    reward_tensor: torch.Tensor | None = None,
) -> dict[str, float]:
    """Zero ``response_mask`` for a uniformly random subset of valid responses.

    Args:
        batch: must carry ``response_mask``; mutated in place. Also gains a
            ``random_drop_reject_mask`` bool tensor, mirroring RDC's
            ``router_mismatch_reject_mask`` so downstream dumps can treat both alike.
        config: the full trainer config (read via ``random_drop.*``).
        step: training step, mixed into the seed so each step draws a different
            subset while a resumed run replays the same draws.
        reward_tensor: optional per-token rewards for the dropped-vs-kept score
            metrics. At the point this hook runs, the async trainer holds them on
            ``self.reward_tensor`` and has not yet written ``token_level_scores``
            onto the batch (that happens in ``_fit_compute_advantage``), so the
            caller passes them explicitly.

    Returns:
        Metrics dict, ``{}`` when the ablation is disabled.
    """
    if not random_drop_enabled(config):
        return {}
    if "response_mask" not in batch.batch:
        raise RuntimeError("random_drop.enable=True requires response_mask on the batch.")

    response_mask = batch.batch["response_mask"]
    # A response with no unmasked tokens is already inert (e.g. a zero-length
    # generation); excluding it keeps the realized fraction honest.
    seq_valid_token_count = response_mask.sum(dim=-1)
    valid_seq = seq_valid_token_count > 0
    valid_indices = torch.nonzero(valid_seq, as_tuple=False).flatten()
    n_valid = int(valid_indices.numel())

    base_seed = OmegaConf.select(config, "random_drop.seed", default=None)
    if base_seed is None:
        base_seed = OmegaConf.select(config, "data.seed", default=None)
    base_seed = 1234 if base_seed is None else int(base_seed)
    # A private generator, so this draw neither consumes nor perturbs global RNG
    # state that rollout / dataloader ordering depends on.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(base_seed * 1_000_003 + int(step))

    k, target_fraction = _resolve_count(n_valid, config, generator)

    reject = torch.zeros_like(valid_seq, dtype=torch.bool)
    if k > 0:
        # randperm over valid rows only: a uniform k-subset without replacement.
        perm = torch.randperm(n_valid, generator=generator)[:k]
        reject[valid_indices[perm.to(valid_indices.device)]] = True

    batch.batch["random_drop_reject_mask"] = reject
    rejected_count = int(reject.sum().item())
    if rejected_count > 0:
        if rejected_count >= n_valid:
            # Every response masked => zero loss tokens => the optimizer step is a
            # no-op at best and a NaN at worst. Cheap to catch, expensive to debug.
            raise RuntimeError(
                f"random_drop would mask all {n_valid} valid responses in the batch; "
                "lower random_drop.fraction or random_drop.count."
            )
        response_mask[reject] = 0

    kept_count = max(n_valid - rejected_count, 0)
    metrics = {
        f"{METRIC_PREFIX}/target_fraction": target_fraction,
        f"{METRIC_PREFIX}/realized_fraction": float(rejected_count / n_valid) if n_valid > 0 else 0.0,
        f"{METRIC_PREFIX}/rejected_count": float(rejected_count),
        f"{METRIC_PREFIX}/kept_count": float(kept_count),
        f"{METRIC_PREFIX}/valid_count": float(n_valid),
    }

    # Length and score summaries: with uniform selection these should match the
    # batch means up to sampling noise. If they drift, selection is not uniform.
    if n_valid > 0:
        lengths = seq_valid_token_count.float()
        metrics[f"{METRIC_PREFIX}/length_before_mean"] = float(lengths[valid_seq].mean().item())
        if rejected_count > 0:
            metrics[f"{METRIC_PREFIX}/rejected_length_mean"] = float(lengths[reject].mean().item())
        if kept_count > 0:
            metrics[f"{METRIC_PREFIX}/kept_length_mean"] = float(lengths[valid_seq & ~reject].mean().item())

        token_level_scores = batch.batch.get("token_level_scores")
        if token_level_scores is None:
            token_level_scores = reward_tensor
        if token_level_scores is not None and token_level_scores.shape[0] == int(reject.numel()):
            sequence_scores = token_level_scores.float().sum(dim=-1)
            metrics[f"{METRIC_PREFIX}/score_before_mean"] = float(sequence_scores[valid_seq].mean().item())
            if rejected_count > 0:
                rejected_scores = sequence_scores[reject]
                metrics[f"{METRIC_PREFIX}/rejected_score_mean"] = float(rejected_scores.mean().item())
                metrics[f"{METRIC_PREFIX}/rejected_positive_fraction"] = float(
                    (rejected_scores > 0).float().mean().item()
                )
            if kept_count > 0:
                kept_scores = sequence_scores[valid_seq & ~reject]
                metrics[f"{METRIC_PREFIX}/kept_score_mean"] = float(kept_scores.mean().item())
                metrics[f"{METRIC_PREFIX}/kept_positive_fraction"] = float((kept_scores > 0).float().mean().item())

    logger.info(
        "[RANDOM-DROP] step=%s valid=%s dropped=%s realized_fraction=%.4f",
        step,
        n_valid,
        rejected_count,
        metrics[f"{METRIC_PREFIX}/realized_fraction"],
    )
    return metrics
