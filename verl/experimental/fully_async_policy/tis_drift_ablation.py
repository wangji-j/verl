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
"""Probability-drift sequence-drop ablation: the second control arm for RDC.

RDC (Router Drift Control) drops the responses with the largest *router* drift
between rollout (V_gen, FP8 vLLM) and train (BF16 actor). This arm keeps the RDC
recipe byte-for-byte -- same length buckets, same ``ceil(n*fraction)`` per bucket,
same "zero ``response_mask``, do not delete rows" mechanism, same TIS correction
on every surviving token -- and changes exactly one thing: the *ranking signal*.
Instead of router drift it ranks by the rollout-vs-probe **probability** drift,
i.e. the same log-ratio TIS itself truncates.

Purpose of the three-way comparison (all with TIS on):

  * RDC arm            -> rank by router (expert-routing) drift.
  * this arm           -> rank by probability drift (A: TIS-truncated fraction,
                          or B: mean |log-ratio|).
  * random-drop arm    -> no ranking, drop a uniform random subset.

If the RDC gain is really about *which* sequences are removed, RDC should beat
this probability-drift arm; if drift magnitude alone (regardless of whether it is
routing- or probability-flavored) is what matters, the two should tie.

Score definitions -- ``log_ratio_t = logp_probe_t - logp_rollout_t`` (the exact
quantity TIS uses; ``logp_probe`` is the train-side anchor, see below), threshold
``C`` = ``tis_drift.clip_c`` (default 2.0, matching ``rollout_is_threshold``):

  * A. ``trunc_frac``          -> fraction of a sequence's valid tokens whose TIS
                                  weight is truncated, i.e. ``exp(log_ratio_t) > C``
                                  <=> ``log_ratio_t > ln(C)``. Literal "fraction
                                  truncated by TIS".
  * B. ``mean_abs_logratio``   -> mean over valid tokens of ``|log_ratio_t|``.
                                  Threshold-free drift magnitude.

Which anchor is ``logp_probe``?
  * off-policy (sync, dapo trainer): a *current* re-probe of theta_(j-1) at every
    PPO mini-step, so the score tracks the live off-policy discrepancy the way the
    current-aware RDC arm tracks live routing drift. The caller passes
    ``probe_log_probs`` explicitly.
  * fully-async (disaggregated): there is no per-mini-step loop; the anchor is the
    train-side ``old_log_probs`` recompute already on the batch, exactly the anchor
    the static async RDC arm compares against. The caller omits ``probe_log_probs``
    and this module reads ``batch.batch["old_log_probs"]``.

Config (all keys additive, pass with ``+`` on the CLI)::

    +tis_drift.enable=True
    +tis_drift.score=trunc_frac        # trunc_frac | mean_abs_logratio
    +tis_drift.clip_c=2.0              # only used by trunc_frac; matches rollout_is_threshold
    +tis_drift.fraction=0.08
    +tis_drift.length_bucket_edges=[2048,4096,8192,12288]

Nothing here reads router routes, so the run needs neither
``rollout.enable_return_routed_experts`` nor ``router.enable_mismatch_metrics``.
"""

import logging
import math
import os

import torch
from omegaconf import OmegaConf

from verl import DataProto

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

METRIC_PREFIX = "tis_drift"

_DEFAULT_LENGTH_BUCKET_EDGES = [2048, 4096, 8192, 12288]
# Clamp log-ratios before any exp/abs so a handful of pathological tokens cannot
# dominate a sequence score. Matches SAFETY_BOUND semantics in rollout_corr_helper.
_SAFETY_BOUND = 20.0


def tis_drift_enabled(config) -> bool:
    return bool(OmegaConf.select(config, "tis_drift.enable", default=False))


def _score_mode(config) -> str:
    mode = str(OmegaConf.select(config, "tis_drift.score", default="trunc_frac")).lower()
    if mode not in {"trunc_frac", "mean_abs_logratio"}:
        raise ValueError(
            f"tis_drift.score must be 'trunc_frac' or 'mean_abs_logratio', got {mode!r}"
        )
    return mode


def _clip_c(config) -> float:
    c = float(OmegaConf.select(config, "tis_drift.clip_c", default=2.0))
    if c <= 0:
        raise ValueError(f"tis_drift.clip_c must be positive, got {c}")
    return c


def _fraction(config) -> float:
    return float(OmegaConf.select(config, "tis_drift.fraction", default=0.08))


def _length_bucket_edges(config) -> list[int]:
    edges = OmegaConf.select(config, "tis_drift.length_bucket_edges", default=None)
    if edges is None:
        return list(_DEFAULT_LENGTH_BUCKET_EDGES)
    return [int(e) for e in edges]


def compute_tis_drift_seq_score(
    batch: DataProto,
    config,
    probe_log_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sequence probability-drift score and per-sequence valid-token count.

    Args:
        batch: must carry ``response_mask`` and ``rollout_log_probs``. When
            ``probe_log_probs`` is None it must also carry ``old_log_probs``.
        config: full trainer config (read via ``tis_drift.*``).
        probe_log_probs: train-side anchor log-probs, shape (n_resp, resp_len),
            aligned with ``response_mask``. When None, ``old_log_probs`` is used
            (the async/static anchor). When given (off-policy current-aware), it is
            the current theta_(j-1) re-probe.

    Returns:
        ``(seq_score, seq_valid_token_count)``, both shape (n_resp,). ``seq_score``
        is 0 for sequences with no valid tokens (they are never selected).
    """
    if "response_mask" not in batch.batch:
        raise RuntimeError("tis_drift requires response_mask on the batch.")
    if "rollout_log_probs" not in batch.batch:
        raise RuntimeError(
            "tis_drift requires rollout_log_probs on the batch "
            "(actor_rollout_ref.rollout.calculate_log_probs=True)."
        )

    response_mask = batch.batch["response_mask"]
    rollout_log_probs = batch.batch["rollout_log_probs"]
    if probe_log_probs is None:
        if "old_log_probs" not in batch.batch:
            raise RuntimeError(
                "tis_drift with no explicit probe_log_probs requires old_log_probs on the batch."
            )
        probe_log_probs = batch.batch["old_log_probs"]

    device = response_mask.device
    probe_log_probs = probe_log_probs.to(device=device)
    rollout_log_probs = rollout_log_probs.to(device=device)
    mask = response_mask.to(dtype=torch.float32)

    # Exactly TIS's log-ratio: log(pi_train / pi_rollout). Clamp before exp/abs.
    log_ratio = (probe_log_probs - rollout_log_probs).to(dtype=torch.float32)
    log_ratio = torch.clamp(log_ratio, min=-_SAFETY_BOUND, max=_SAFETY_BOUND)

    seq_valid_token_count = mask.sum(dim=-1)
    safe_count = seq_valid_token_count.clamp(min=1.0)

    mode = _score_mode(config)
    if mode == "trunc_frac":
        # A token is TIS-truncated iff exp(log_ratio) > C  <=>  log_ratio > ln(C).
        ln_c = math.log(_clip_c(config))
        truncated = (log_ratio > ln_c).to(dtype=torch.float32) * mask
        seq_score = truncated.sum(dim=-1) / safe_count
    else:  # mean_abs_logratio
        abs_lr = log_ratio.abs() * mask
        seq_score = abs_lr.sum(dim=-1) / safe_count

    # Sequences with no valid tokens get score 0 so they never rank into the top.
    seq_score = torch.where(seq_valid_token_count > 0, seq_score, torch.zeros_like(seq_score))
    return seq_score, seq_valid_token_count


def apply_tis_drift_bucket_drop(
    batch: DataProto,
    config,
    seq_score: torch.Tensor,
    seq_valid_token_count: torch.Tensor,
    reward_tensor: torch.Tensor | None = None,
) -> dict[str, float]:
    """Zero ``response_mask`` for the top ``ceil(n*fraction)`` scorers within each length bucket.

    This mirrors ``length_bucket_top_fraction`` in ``ray_trainer._apply_router_mismatch_rs``
    exactly (same bucket edges convention, same ceil count rule, same "zero rows,
    keep them for the GRPO baseline" mechanism), so the only difference from the RDC
    arm is that ``seq_score`` is a probability-drift score rather than router drift.

    Args:
        batch: carries ``response_mask``; mutated in place. Gains a bool
            ``tis_drift_reject_mask`` mirroring RDC's ``router_mismatch_reject_mask``.
        config: full trainer config (read via ``tis_drift.*``).
        seq_score: per-sequence ranking signal, shape (n_resp,). Higher = drop first.
        seq_valid_token_count: per-sequence valid-token count for bucketing, (n_resp,).
        reward_tensor: optional per-token rewards for dropped-vs-kept score metrics
            (async passes ``self.reward_tensor`` before token_level_scores is written).

    Returns:
        Metrics dict under ``tis_drift/``, ``{}`` when the ablation is disabled.
    """
    if not tis_drift_enabled(config):
        return {}
    if "response_mask" not in batch.batch:
        raise RuntimeError("tis_drift requires response_mask on the batch.")

    response_mask = batch.batch["response_mask"]
    device = response_mask.device
    seq_score = seq_score.to(device=device)
    seq_valid_token_count = seq_valid_token_count.to(device=device)
    valid_seq = seq_valid_token_count > 0

    fraction = _fraction(config)
    bucket_edges = _length_bucket_edges(config)
    bucket_lowers = [0] + bucket_edges
    bucket_uppers = bucket_edges + [None]

    reject = torch.zeros_like(valid_seq, dtype=torch.bool)
    effective_threshold = 0.0
    thresholds: list[float] = []
    bucket_metrics: dict[str, float] = {}

    for bucket_idx, (lower, upper) in enumerate(zip(bucket_lowers, bucket_uppers, strict=True)):
        in_bucket = valid_seq & (seq_valid_token_count >= float(lower))
        if upper is not None:
            in_bucket = in_bucket & (seq_valid_token_count < float(upper))
        bucket_indices = torch.nonzero(in_bucket, as_tuple=False).flatten()
        bucket_count = int(bucket_indices.numel())
        bucket_reject_count = 0
        bucket_threshold = 0.0
        if fraction > 0 and bucket_count > 0:
            k = min(max(math.ceil(bucket_count * fraction), 1), bucket_count)
            bucket_scores = seq_score[bucket_indices]
            topk_values, topk_pos = torch.topk(bucket_scores, k=k, largest=True, sorted=False)
            reject[bucket_indices[topk_pos]] = True
            bucket_reject_count = int(k)
            bucket_threshold = float(topk_values.min().item())
            thresholds.append(bucket_threshold)
        if bucket_count > 0:
            bucket_scores_all = seq_score[bucket_indices]
            bucket_lengths_all = seq_valid_token_count[bucket_indices]
            prefix = f"{METRIC_PREFIX}/bucket_{bucket_idx}"
            bucket_metrics[f"{prefix}_lower"] = float(lower)
            bucket_metrics[f"{prefix}_upper"] = float(upper) if upper is not None else -1.0
            bucket_metrics[f"{prefix}_count"] = float(bucket_count)
            bucket_metrics[f"{prefix}_rejected_count"] = float(bucket_reject_count)
            bucket_metrics[f"{prefix}_rejected_fraction"] = float(bucket_reject_count / bucket_count)
            bucket_metrics[f"{prefix}_threshold"] = float(bucket_threshold)
            bucket_metrics[f"{prefix}_score_mean"] = float(bucket_scores_all.float().mean().item())
            bucket_metrics[f"{prefix}_length_mean"] = float(bucket_lengths_all.float().mean().item())
    if thresholds:
        effective_threshold = min(thresholds)

    batch.batch["tis_drift_reject_mask"] = reject
    valid_count = int(valid_seq.sum().item())
    rejected_count = int(reject.sum().item())
    if rejected_count > 0:
        if rejected_count >= valid_count:
            # Masking every valid response gives a zero-token (NaN-prone) update.
            raise RuntimeError(
                f"tis_drift would mask all {valid_count} valid responses in the batch; "
                "lower tis_drift.fraction."
            )
        response_mask[reject] = 0

    kept_count = max(valid_count - rejected_count, 0)
    metrics = {
        f"{METRIC_PREFIX}/score_mode": 0.0 if _score_mode(config) == "trunc_frac" else 1.0,
        f"{METRIC_PREFIX}/clip_c": _clip_c(config),
        f"{METRIC_PREFIX}/configured_fraction": fraction,
        f"{METRIC_PREFIX}/threshold": effective_threshold,
        f"{METRIC_PREFIX}/rejected_count": float(rejected_count),
        f"{METRIC_PREFIX}/kept_count": float(kept_count),
        f"{METRIC_PREFIX}/valid_count": float(valid_count),
        f"{METRIC_PREFIX}/rejected_fraction": float(rejected_count / valid_count) if valid_count > 0 else 0.0,
    }
    if valid_count > 0:
        metrics[f"{METRIC_PREFIX}/score_before_mean"] = float(seq_score[valid_seq].float().mean().item())
        lengths = seq_valid_token_count.float()
        metrics[f"{METRIC_PREFIX}/length_before_mean"] = float(lengths[valid_seq].mean().item())
        if rejected_count > 0:
            metrics[f"{METRIC_PREFIX}/rejected_score_mean"] = float(seq_score[reject].float().mean().item())
            metrics[f"{METRIC_PREFIX}/rejected_length_mean"] = float(lengths[reject].mean().item())
        if kept_count > 0:
            kept_seq = valid_seq & ~reject
            metrics[f"{METRIC_PREFIX}/kept_score_mean"] = float(seq_score[kept_seq].float().mean().item())
            metrics[f"{METRIC_PREFIX}/kept_length_mean"] = float(lengths[kept_seq].mean().item())

        token_level_scores = batch.batch.get("token_level_scores")
        if token_level_scores is None:
            token_level_scores = reward_tensor
        if token_level_scores is not None and token_level_scores.shape[0] == int(reject.numel()):
            sequence_scores = token_level_scores.float().sum(dim=-1)
            metrics[f"{METRIC_PREFIX}/reward_before_mean"] = float(sequence_scores[valid_seq].mean().item())
            if rejected_count > 0:
                rejected_scores = sequence_scores[reject]
                metrics[f"{METRIC_PREFIX}/rejected_reward_mean"] = float(rejected_scores.mean().item())
                metrics[f"{METRIC_PREFIX}/rejected_positive_fraction"] = float(
                    (rejected_scores > 0).float().mean().item()
                )
            if kept_count > 0:
                kept_scores = sequence_scores[valid_seq & ~reject]
                metrics[f"{METRIC_PREFIX}/kept_reward_mean"] = float(kept_scores.mean().item())
                metrics[f"{METRIC_PREFIX}/kept_positive_fraction"] = float(
                    (kept_scores > 0).float().mean().item()
                )
    metrics.update(bucket_metrics)

    logger.info(
        "[TIS-DRIFT] mode=%s valid=%s dropped=%s rejected_fraction=%.4f threshold=%.4f",
        _score_mode(config),
        valid_count,
        rejected_count,
        metrics[f"{METRIC_PREFIX}/rejected_fraction"],
        effective_threshold,
    )
    return metrics
