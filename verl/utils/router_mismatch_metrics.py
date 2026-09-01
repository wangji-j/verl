# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class RouterMismatchResult:
    metrics: dict[str, float]
    alignment: int
    alignment_scores: dict[int, float] | None = None
    seq_mismatch: torch.Tensor | None = None
    seq_valid_token_count: torch.Tensor | None = None
    token_mismatch: torch.Tensor | None = None
    expert_usage_distances: dict[str, torch.Tensor] | None = None
    expert_usage_distances_by_layer: dict[str, torch.Tensor] | None = None
    expert_usage_counts: dict[str, torch.Tensor] | None = None


def compute_length_conditional_percentiles(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    *,
    local_window: int = 256,
    censored_length: int | None = None,
    min_censored_count: int = 32,
) -> torch.Tensor:
    """Rank each score against responses with nearby lengths.

    Non-censored responses use a centered sliding window after sorting by
    length. Responses at the configured maximum length are right-censored and
    ranked only against each other when that group is large enough. Ties use
    their midpoint empirical percentile.
    """

    if scores.ndim != 1 or lengths.ndim != 1 or scores.shape != lengths.shape:
        raise ValueError(
            "scores and lengths must be one-dimensional tensors with identical shape, "
            f"got scores={tuple(scores.shape)} lengths={tuple(lengths.shape)}"
        )
    if local_window <= 0:
        raise ValueError(f"local_window must be positive, got {local_window}")
    if min_censored_count <= 0:
        raise ValueError(f"min_censored_count must be positive, got {min_censored_count}")

    percentiles = torch.full_like(scores, float("-inf"), dtype=torch.float32)
    valid = torch.isfinite(scores) & (lengths > 0)
    if not bool(valid.any().item()):
        return percentiles

    censored = torch.zeros_like(valid)
    if censored_length is not None:
        candidate = valid & (lengths >= censored_length)
        if int(candidate.sum().item()) >= min_censored_count:
            censored = candidate

    def assign_local(indices: torch.Tensor) -> None:
        count = int(indices.numel())
        if count == 0:
            return
        order = torch.argsort(lengths[indices], stable=True)
        sorted_indices = indices[order]
        sorted_scores = scores[sorted_indices].float()
        width = min(max(local_window, 32), count)
        half = width // 2
        positions = torch.arange(count, device=scores.device)
        starts = (positions - half).clamp(min=0, max=count - width)
        offsets = torch.arange(width, device=scores.device)
        neighbors = sorted_scores[starts[:, None] + offsets[None, :]]
        values = sorted_scores[:, None]
        local_percentiles = (
            (neighbors < values).sum(dim=-1).float()
            + 0.5 * (neighbors == values).sum(dim=-1).float()
        ) / float(width)
        percentiles[sorted_indices] = local_percentiles

    assign_local(torch.nonzero(valid & ~censored, as_tuple=False).flatten())

    censored_indices = torch.nonzero(censored, as_tuple=False).flatten()
    if int(censored_indices.numel()) > 0:
        censored_scores = scores[censored_indices].float()
        comparisons = censored_scores[:, None]
        censored_percentiles = (
            (censored_scores[None, :] < comparisons).sum(dim=-1).float()
            + 0.5 * (censored_scores[None, :] == comparisons).sum(dim=-1).float()
        ) / float(censored_indices.numel())
        percentiles[censored_indices] = censored_percentiles

    return percentiles


def compute_router_mismatch_metrics(
    rollout_routed_experts: torch.Tensor,
    train_routed_experts: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    metric_prefix: str = "router",
    alignments: Iterable[int] = (0, -1, 1),
    metric_mode: str = "exact_set",
    expert_usage_smoothing_tau: float = 4096.0,
    expert_usage_num_experts: int | None = None,
    capture_expert_usage_counts: bool = False,
) -> RouterMismatchResult:
    """Compare rollout and training router choices on response-token positions.

    ``alignment`` shifts the training route positions before comparing:
    ``0`` compares identical sequence positions, ``-1`` compares the route
    from the previous training input position, and ``1`` compares the next
    position. The best alignment is selected by response-token top-k match rate.
    """

    compute_device = _select_compute_device(rollout_routed_experts, train_routed_experts)
    rollout = _normalize_routes(rollout_routed_experts, device=compute_device)
    train = _normalize_routes(train_routed_experts, device=compute_device)
    mask = response_mask.to(dtype=torch.bool, device=compute_device)

    if rollout.ndim != 4 or train.ndim != 4:
        raise ValueError(
            "routed_experts must have shape [batch, seq_len, num_layers, topk], "
            f"got rollout={tuple(rollout.shape)} train={tuple(train.shape)}"
        )

    if rollout.shape[0] != train.shape[0]:
        raise ValueError(f"Batch size mismatch: rollout={rollout.shape[0]} train={train.shape[0]}")

    response_len = mask.shape[1]
    rollout_resp = rollout[:, -response_len:]
    seq_len = min(rollout_resp.shape[1], train.shape[1], response_len)
    layer_count = min(rollout_resp.shape[2], train.shape[2])
    topk = min(rollout_resp.shape[3], train.shape[3])
    if metric_mode not in {"exact_set", "overlap_fraction", "expert_usage_l1"}:
        raise ValueError(
            "metric_mode must be 'exact_set', 'overlap_fraction', or "
            f"'expert_usage_l1', got {metric_mode!r}"
        )

    if seq_len == 0 or layer_count == 0 or topk == 0:
        return RouterMismatchResult(
            metrics={
                f"{metric_prefix}/response_token_match_rate": 0.0,
                f"{metric_prefix}/response_token_count": 0.0,
            },
            alignment=0,
        )

    rollout_resp = _canonicalize_topk(rollout_resp[:, -seq_len:, :layer_count, :topk])
    train = _canonicalize_topk(train[:, :, :layer_count, :topk])
    mask = mask[:, -seq_len:]

    best: tuple[float, int] | None = None
    alignment_scores: dict[int, float] = {}
    for alignment in alignments:
        train_resp, aligned_mask = _aligned_train_response(train, response_len=seq_len, alignment=int(alignment))
        valid_mask = mask & aligned_mask
        layer_match = _topk_expert_set_match(rollout_resp, train_resp)
        score, _ = _response_token_match_rate(layer_match, valid_mask)
        alignment_scores[int(alignment)] = float(score)
        if best is None or score > best[0]:
            best = (score, int(alignment))

    assert best is not None
    _, best_alignment = best
    train_resp, aligned_mask = _aligned_train_response(train, response_len=seq_len, alignment=best_alignment)
    valid_mask = mask & aligned_mask
    layer_match = _topk_expert_set_match(rollout_resp, train_resp)
    metrics = _compare_layer_match(layer_match, valid_mask, metric_prefix=metric_prefix)
    if metric_mode == "expert_usage_l1":
        usage = _expert_usage_l1_mismatch(
            rollout_resp,
            train_resp,
            valid_mask,
            smoothing_tau=expert_usage_smoothing_tau,
            num_experts=expert_usage_num_experts,
            metric_prefix=metric_prefix,
            capture_counts=capture_expert_usage_counts,
        )
        metrics.update(usage["metrics"])
        metrics[f"{metric_prefix}/seq_mismatch_metric_mode"] = 2.0
        seq_mismatch = usage["seq_mismatch"]
        seq_valid_token_count = usage["seq_valid_token_count"]
        token_mismatch = None
        expert_usage_distances = usage["distances"]
        expert_usage_distances_by_layer = usage["distances_by_layer"]
        expert_usage_counts = usage["counts"]
    elif metric_mode == "overlap_fraction":
        layer_mismatch = _topk_expert_overlap_mismatch(rollout_resp, train_resp)
        metrics[f"{metric_prefix}/seq_mismatch_metric_mode"] = 1.0
        seq_mismatch, seq_valid_token_count, token_mismatch = _compute_sequence_mismatch(layer_mismatch, valid_mask)
        metrics.update(_sequence_mismatch_metrics(seq_mismatch, seq_valid_token_count, metric_prefix=metric_prefix))
        expert_usage_distances = None
        expert_usage_distances_by_layer = None
        expert_usage_counts = None
    else:
        layer_mismatch = (~layer_match).float()
        metrics[f"{metric_prefix}/seq_mismatch_metric_mode"] = 0.0
        seq_mismatch, seq_valid_token_count, token_mismatch = _compute_sequence_mismatch(layer_mismatch, valid_mask)
        metrics.update(_sequence_mismatch_metrics(seq_mismatch, seq_valid_token_count, metric_prefix=metric_prefix))
        expert_usage_distances = None
        expert_usage_distances_by_layer = None
        expert_usage_counts = None
    return RouterMismatchResult(
        metrics=metrics,
        alignment=best_alignment,
        alignment_scores=alignment_scores,
        seq_mismatch=seq_mismatch,
        seq_valid_token_count=seq_valid_token_count,
        token_mismatch=token_mismatch,
        expert_usage_distances=expert_usage_distances,
        expert_usage_distances_by_layer=expert_usage_distances_by_layer,
        expert_usage_counts=expert_usage_counts,
    )


def _normalize_routes(routes: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    routes = routes.detach().to(device=device)
    if routes.is_nested:
        routes = routes.to_padded_tensor(0)
    if not routes.dtype.is_floating_point and not routes.dtype.is_complex:
        return routes
    routes = routes.to(dtype=torch.int64)
    return routes


def _select_compute_device(rollout: torch.Tensor, train: torch.Tensor) -> torch.device:
    mode = os.getenv("VERL_ROUTER_MISMATCH_DEVICE", "auto").strip().lower()
    if mode in {"cpu", "host"}:
        return torch.device("cpu")
    if mode in {"cuda", "gpu"} and torch.cuda.is_available():
        if train.is_cuda:
            return train.device
        if rollout.is_cuda:
            return rollout.device
        return torch.device("cuda")
    if train.is_cuda:
        return train.device
    if rollout.is_cuda:
        return rollout.device
    return torch.device("cpu")


def _aligned_train_response(train: torch.Tensor, *, response_len: int, alignment: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, train_len = train.shape[:2]
    start = train_len - response_len + alignment
    out = torch.zeros((batch_size, response_len, *train.shape[2:]), dtype=train.dtype, device=train.device)
    mask = torch.zeros((batch_size, response_len), dtype=torch.bool, device=train.device)

    src_start = max(start, 0)
    dst_start = max(-start, 0)
    copy_len = min(train_len - src_start, response_len - dst_start)
    if copy_len > 0:
        out[:, dst_start : dst_start + copy_len] = train[:, src_start : src_start + copy_len]
        mask[:, dst_start : dst_start + copy_len] = True
    return out, mask


def _canonicalize_topk(routes: torch.Tensor) -> torch.Tensor:
    return torch.sort(routes, dim=-1).values


def _topk_expert_set_match(rollout: torch.Tensor, train: torch.Tensor) -> torch.Tensor:
    return (rollout == train).all(dim=-1)


def _topk_expert_overlap_mismatch(rollout: torch.Tensor, train: torch.Tensor) -> torch.Tensor:
    overlap = (rollout.unsqueeze(-1) == train.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
    return 1.0 - overlap.float() / max(rollout.shape[-1], 1)


def _expert_usage_l1_mismatch(
    rollout: torch.Tensor,
    train: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    smoothing_tau: float,
    num_experts: int | None,
    metric_prefix: str,
    capture_counts: bool,
) -> dict[str, torch.Tensor | dict[str, float]]:
    """Response-level drift between rollout/train expert-usage histograms.

    The smoothed score is equivalent to shrinking both rollout and train
    empirical histograms toward the same layer prior before computing TV/L1.
    Because the same prior is used on both sides, it can be computed directly
    from count differences as 0.5 * |c_rollout - c_train|_1 / (N + tau).
    """

    valid = response_mask.to(dtype=torch.bool)
    batch_size, _, layer_count, topk = rollout.shape
    if num_experts is None or num_experts <= 0:
        expert_count = max(int(rollout.max().item()), int(train.max().item())) + 1
    else:
        expert_count = int(num_experts)
    expert_count = max(expert_count, 1)

    token_count = valid.float().sum(dim=-1)
    slot_count = token_count * float(topk)
    valid_seq = slot_count > 0
    slot_count_safe = slot_count.clamp_min(1.0)
    tau = max(float(smoothing_tau), 0.0)

    raw_by_layer = torch.zeros((batch_size, layer_count), dtype=torch.float32, device=rollout.device)
    smooth_by_layer = torch.zeros_like(raw_by_layer)
    l2_by_layer = torch.zeros_like(raw_by_layer)
    linf_by_layer = torch.zeros_like(raw_by_layer)
    hellinger_sq_by_layer = torch.zeros_like(raw_by_layer)
    js_normalized_by_layer = torch.zeros_like(raw_by_layer)
    effective_support_by_layer = torch.zeros_like(raw_by_layer)
    rollout_counts_dump = None
    train_counts_dump = None
    if capture_counts:
        count_shape = (batch_size, layer_count, expert_count)
        rollout_counts_dump = torch.empty(count_shape, dtype=torch.uint16, device=rollout.device)
        train_counts_dump = torch.empty_like(rollout_counts_dump)
    batch_chunk = max(int(os.getenv("VERL_ROUTER_USAGE_L1_BATCH_CHUNK", "32")), 1)

    for layer_idx in range(layer_count):
        rollout_counts = torch.zeros((batch_size, expert_count), dtype=torch.float32, device=rollout.device)
        train_counts = torch.zeros_like(rollout_counts)

        for start in range(0, batch_size, batch_chunk):
            end = min(start + batch_chunk, batch_size)
            local_valid = valid[start:end]
            if not bool(local_valid.any().item()):
                continue

            rollout_ids = rollout[start:end, :, layer_idx, :]
            train_ids = train[start:end, :, layer_idx, :]
            local_rows = torch.arange(end - start, device=rollout.device).view(-1, 1, 1)
            local_rows = local_rows.expand_as(rollout_ids)
            valid_slots = local_valid.unsqueeze(-1).expand_as(rollout_ids)

            rollout_linear = (local_rows[valid_slots].to(dtype=torch.int64) * expert_count) + rollout_ids[
                valid_slots
            ].clamp(min=0, max=expert_count - 1).to(dtype=torch.int64)
            train_linear = (local_rows[valid_slots].to(dtype=torch.int64) * expert_count) + train_ids[
                valid_slots
            ].clamp(min=0, max=expert_count - 1).to(dtype=torch.int64)

            minlength = (end - start) * expert_count
            rollout_counts[start:end] = torch.bincount(rollout_linear, minlength=minlength).reshape(
                end - start, expert_count
            )
            train_counts[start:end] = torch.bincount(train_linear, minlength=minlength).reshape(
                end - start, expert_count
            )

        count_tv = 0.5 * (rollout_counts - train_counts).abs().sum(dim=-1)
        raw_by_layer[:, layer_idx] = count_tv / slot_count_safe
        smooth_by_layer[:, layer_idx] = count_tv / (slot_count + tau).clamp_min(1.0)

        rollout_prob = rollout_counts / slot_count_safe.unsqueeze(-1)
        train_prob = train_counts / slot_count_safe.unsqueeze(-1)
        delta = rollout_prob - train_prob
        abs_delta = delta.abs()
        l2_by_layer[:, layer_idx] = torch.linalg.vector_norm(delta, ord=2, dim=-1)
        linf_by_layer[:, layer_idx] = abs_delta.max(dim=-1).values
        hellinger_sq_by_layer[:, layer_idx] = 0.5 * (
            rollout_prob.sqrt() - train_prob.sqrt()
        ).square().sum(dim=-1)

        mixture = 0.5 * (rollout_prob + train_prob)
        rollout_js = torch.where(
            rollout_prob > 0,
            rollout_prob * (rollout_prob.clamp_min(torch.finfo(torch.float32).tiny).log() - mixture.log()),
            0.0,
        )
        train_js = torch.where(
            train_prob > 0,
            train_prob * (train_prob.clamp_min(torch.finfo(torch.float32).tiny).log() - mixture.log()),
            0.0,
        )
        js_normalized_by_layer[:, layer_idx] = 0.5 * (rollout_js + train_js).sum(dim=-1) / math.log(2.0)

        abs_sum = abs_delta.sum(dim=-1)
        abs_square_sum = abs_delta.square().sum(dim=-1)
        nonzero_drift = abs_square_sum > 0
        effective_support_by_layer[nonzero_drift, layer_idx] = (
            abs_sum[nonzero_drift].square() / abs_square_sum[nonzero_drift]
        )

        if capture_counts:
            rollout_counts_dump[:, layer_idx] = rollout_counts.to(dtype=torch.uint16)
            train_counts_dump[:, layer_idx] = train_counts.to(dtype=torch.uint16)

    seq_mismatch = torch.zeros((batch_size,), dtype=torch.float32, device=rollout.device)
    seq_mismatch_raw = torch.zeros_like(seq_mismatch)
    distance_layers = {
        "tv_raw": raw_by_layer,
        "tv_smooth": smooth_by_layer,
        "l2": l2_by_layer,
        "linf": linf_by_layer,
        "hellinger_sq": hellinger_sq_by_layer,
        "js_normalized": js_normalized_by_layer,
        "effective_support": effective_support_by_layer,
    }
    distance_sequences = {
        name: torch.zeros((batch_size,), dtype=torch.float32, device=rollout.device) for name in distance_layers
    }
    if layer_count > 0:
        seq_mismatch[valid_seq] = smooth_by_layer[valid_seq].mean(dim=-1)
        seq_mismatch_raw[valid_seq] = raw_by_layer[valid_seq].mean(dim=-1)
        for name, values in distance_layers.items():
            distance_sequences[name][valid_seq] = values[valid_seq].mean(dim=-1)

    metrics: dict[str, float] = {
        f"{metric_prefix}/expert_usage_l1_num_experts": float(expert_count),
        f"{metric_prefix}/expert_usage_l1_smoothing_tau": tau,
    }
    if not bool(valid_seq.any().item()):
        metrics.update(
            {
                f"{metric_prefix}/expert_usage_l1_mean": 0.0,
                f"{metric_prefix}/expert_usage_l1_max": 0.0,
                f"{metric_prefix}/expert_usage_l1_p90": 0.0,
                f"{metric_prefix}/expert_usage_l1_raw_mean": 0.0,
                f"{metric_prefix}/expert_usage_l1_raw_p90": 0.0,
                f"{metric_prefix}/expert_usage_l1_smoothing_delta_mean": 0.0,
                f"{metric_prefix}/expert_usage_l1_shrink_factor_mean": 0.0,
                f"{metric_prefix}/expert_usage_l1_shrink_factor_p10": 0.0,
                f"{metric_prefix}/expert_usage_l1_shrink_factor_p50": 0.0,
                f"{metric_prefix}/expert_usage_l1_shrink_factor_p90": 0.0,
            }
        )
    else:
        smooth_values = seq_mismatch[valid_seq].float()
        raw_values = seq_mismatch_raw[valid_seq].float()
        shrink = slot_count[valid_seq].float() / (slot_count[valid_seq].float() + tau).clamp_min(1.0)
        metrics.update(
            {
                f"{metric_prefix}/expert_usage_l1_mean": float(smooth_values.mean().item()),
                f"{metric_prefix}/expert_usage_l1_max": float(smooth_values.max().item()),
                f"{metric_prefix}/expert_usage_l1_p90": float(torch.quantile(smooth_values, 0.9).item()),
                f"{metric_prefix}/expert_usage_l1_raw_mean": float(raw_values.mean().item()),
                f"{metric_prefix}/expert_usage_l1_raw_max": float(raw_values.max().item()),
                f"{metric_prefix}/expert_usage_l1_raw_p90": float(torch.quantile(raw_values, 0.9).item()),
                f"{metric_prefix}/expert_usage_l1_smoothing_delta_mean": float(
                    (raw_values - smooth_values).mean().item()
                ),
                f"{metric_prefix}/expert_usage_l1_shrink_factor_mean": float(shrink.mean().item()),
                f"{metric_prefix}/expert_usage_l1_shrink_factor_p10": float(torch.quantile(shrink, 0.1).item()),
                f"{metric_prefix}/expert_usage_l1_shrink_factor_p50": float(torch.quantile(shrink, 0.5).item()),
                f"{metric_prefix}/expert_usage_l1_shrink_factor_p90": float(torch.quantile(shrink, 0.9).item()),
            }
        )
        valid_layer_values = smooth_by_layer[valid_seq].float()
        valid_layer_raw_values = raw_by_layer[valid_seq].float()
        for layer_idx in range(layer_count):
            layer_prefix = f"{metric_prefix}/layer_{layer_idx}"
            metrics[f"{layer_prefix}/expert_usage_l1"] = float(valid_layer_values[:, layer_idx].mean().item())
            metrics[f"{layer_prefix}/expert_usage_l1_raw"] = float(
                valid_layer_raw_values[:, layer_idx].mean().item()
            )

        for name in ("l2", "linf", "hellinger_sq", "js_normalized", "effective_support"):
            values = distance_sequences[name][valid_seq].float()
            distance_prefix = f"{metric_prefix}/expert_usage_{name}"
            metrics[f"{distance_prefix}_mean"] = float(values.mean().item())
            metrics[f"{distance_prefix}_p90"] = float(torch.quantile(values, 0.9).item())
            metrics[f"{distance_prefix}_max"] = float(values.max().item())

    return {
        "metrics": metrics,
        "seq_mismatch": seq_mismatch,
        "seq_valid_token_count": token_count,
        "distances": distance_sequences,
        "distances_by_layer": distance_layers,
        "counts": (
            {"rollout": rollout_counts_dump, "train": train_counts_dump}
            if capture_counts
            else None
        ),
    }


def _response_token_match_rate(layer_match: torch.Tensor, response_mask: torch.Tensor) -> tuple[float, int]:
    row_match = layer_match.all(dim=-1)
    valid = response_mask.to(dtype=torch.bool)
    token_count = int(valid.sum().item())
    if token_count <= 0:
        return 0.0, 0
    return float(row_match[valid].float().mean().item()), token_count


def _compare_layer_match(
    layer_match: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    metric_prefix: str,
) -> dict[str, float]:
    score, token_count = _response_token_match_rate(layer_match, response_mask)
    if token_count <= 0:
        return {
            f"{metric_prefix}/response_token_match_rate": 0.0,
            f"{metric_prefix}/response_token_count": 0.0,
        }

    valid = response_mask.to(dtype=torch.bool)
    metrics = {
        f"{metric_prefix}/response_token_match_rate": score,
        f"{metric_prefix}/response_token_count": float(token_count),
    }

    for layer_idx in range(layer_match.shape[2]):
        layer_prefix = f"{metric_prefix}/layer_{layer_idx}"
        layer_row_match = layer_match[:, :, layer_idx]
        metrics[f"{layer_prefix}/response_token_match_rate"] = float(layer_row_match[valid].float().mean().item())
    return metrics


def _compute_sequence_mismatch(
    layer_mismatch: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_mismatch = layer_mismatch.float().mean(dim=-1)
    valid = response_mask.to(dtype=torch.bool)
    valid_f = valid.float()
    seq_valid_token_count = valid_f.sum(dim=-1)
    seq_mismatch_sum = (token_mismatch * valid_f).sum(dim=-1)
    seq_mismatch = torch.zeros_like(seq_mismatch_sum, dtype=torch.float32)
    non_empty = seq_valid_token_count > 0
    seq_mismatch[non_empty] = seq_mismatch_sum[non_empty] / seq_valid_token_count[non_empty]
    return seq_mismatch, seq_valid_token_count, token_mismatch


def _sequence_mismatch_metrics(
    seq_mismatch: torch.Tensor,
    seq_valid_token_count: torch.Tensor,
    *,
    metric_prefix: str,
) -> dict[str, float]:
    valid_seq = seq_valid_token_count > 0
    if not bool(valid_seq.any()):
        return {
            f"{metric_prefix}/seq_mismatch_mean": 0.0,
            f"{metric_prefix}/seq_mismatch_max": 0.0,
            f"{metric_prefix}/seq_valid_token_count_mean": 0.0,
        }

    values = seq_mismatch[valid_seq].float()
    counts = seq_valid_token_count[valid_seq].float()
    return {
        f"{metric_prefix}/seq_mismatch_mean": float(values.mean().item()),
        f"{metric_prefix}/seq_mismatch_max": float(values.max().item()),
        f"{metric_prefix}/seq_valid_token_count_mean": float(counts.mean().item()),
    }
