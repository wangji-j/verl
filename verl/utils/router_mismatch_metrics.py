# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

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


def compute_router_mismatch_metrics(
    rollout_routed_experts: torch.Tensor,
    train_routed_experts: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    metric_prefix: str = "router",
    alignments: Iterable[int] = (0, -1, 1),
    metric_mode: str = "exact_set",
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
    if metric_mode not in {"exact_set", "overlap_fraction"}:
        raise ValueError(f"metric_mode must be 'exact_set' or 'overlap_fraction', got {metric_mode!r}")

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
    if metric_mode == "overlap_fraction":
        layer_mismatch = _topk_expert_overlap_mismatch(rollout_resp, train_resp)
        metrics[f"{metric_prefix}/seq_mismatch_metric_mode"] = 1.0
    else:
        layer_mismatch = (~layer_match).float()
        metrics[f"{metric_prefix}/seq_mismatch_metric_mode"] = 0.0
    seq_mismatch, seq_valid_token_count = _compute_sequence_mismatch(layer_mismatch, valid_mask)
    metrics.update(_sequence_mismatch_metrics(seq_mismatch, seq_valid_token_count, metric_prefix=metric_prefix))
    return RouterMismatchResult(
        metrics=metrics,
        alignment=best_alignment,
        alignment_scores=alignment_scores,
        seq_mismatch=seq_mismatch,
        seq_valid_token_count=seq_valid_token_count,
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
) -> tuple[torch.Tensor, torch.Tensor]:
    token_mismatch = layer_mismatch.float().mean(dim=-1)
    valid = response_mask.to(dtype=torch.bool)
    valid_f = valid.float()
    seq_valid_token_count = valid_f.sum(dim=-1)
    seq_mismatch_sum = (token_mismatch * valid_f).sum(dim=-1)
    seq_mismatch = torch.zeros_like(seq_mismatch_sum, dtype=torch.float32)
    non_empty = seq_valid_token_count > 0
    seq_mismatch[non_empty] = seq_mismatch_sum[non_empty] / seq_valid_token_count[non_empty]
    return seq_mismatch, seq_valid_token_count


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
