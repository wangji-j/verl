# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class RouterMismatchResult:
    metrics: dict[str, float]
    alignment: int


def compute_router_mismatch_metrics(
    rollout_routed_experts: torch.Tensor,
    train_routed_experts: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    metric_prefix: str = "router",
    alignments: Iterable[int] = (0, -1, 1),
) -> RouterMismatchResult:
    """Compare rollout and training router choices on response-token positions.

    ``alignment`` shifts the training route positions before comparing:
    ``0`` compares identical sequence positions, ``-1`` compares the route
    from the previous training input position, and ``1`` compares the next
    position. The best alignment is selected by response-token top-k match rate.
    """

    rollout = _normalize_routes(rollout_routed_experts)
    train = _normalize_routes(train_routed_experts)
    mask = response_mask.to(dtype=torch.bool, device="cpu")

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
    if seq_len == 0 or layer_count == 0 or topk == 0:
        return RouterMismatchResult(
            metrics={
                f"{metric_prefix}/response_token_match_rate": 0.0,
                f"{metric_prefix}/response_token_count": 0.0,
            },
            alignment=0,
        )

    rollout_resp = rollout_resp[:, -seq_len:, :layer_count, :topk]
    mask = mask[:, -seq_len:]

    best: tuple[float, int] | None = None
    for alignment in alignments:
        train_resp, aligned_mask = _aligned_train_response(train, response_len=seq_len, alignment=int(alignment))
        train_resp = train_resp[:, :, :layer_count, :topk]
        valid_mask = mask & aligned_mask
        metrics = _compare_aligned_routes(
            rollout_resp,
            train_resp,
            valid_mask,
            metric_prefix=f"{metric_prefix}/alignment_{int(alignment)}",
        )
        score = metrics[f"{metric_prefix}/alignment_{int(alignment)}/response_token_match_rate"]
        if best is None or score > best[0]:
            best = (score, int(alignment))

    assert best is not None
    _, best_alignment = best
    train_resp, aligned_mask = _aligned_train_response(train, response_len=seq_len, alignment=best_alignment)
    train_resp = train_resp[:, :, :layer_count, :topk]
    metrics = _compare_aligned_routes(rollout_resp, train_resp, mask & aligned_mask, metric_prefix=metric_prefix)
    return RouterMismatchResult(metrics=metrics, alignment=best_alignment)


def _normalize_routes(routes: torch.Tensor) -> torch.Tensor:
    routes = routes.detach().to(device="cpu", dtype=torch.int64)
    if routes.is_nested:
        routes = routes.to_padded_tensor(0)
    return routes


def _aligned_train_response(train: torch.Tensor, *, response_len: int, alignment: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, train_len = train.shape[:2]
    start = train_len - response_len + alignment
    out = torch.zeros((batch_size, response_len, *train.shape[2:]), dtype=train.dtype)
    mask = torch.zeros((batch_size, response_len), dtype=torch.bool)

    src_start = max(start, 0)
    dst_start = max(-start, 0)
    copy_len = min(train_len - src_start, response_len - dst_start)
    if copy_len > 0:
        out[:, dst_start : dst_start + copy_len] = train[:, src_start : src_start + copy_len]
        mask[:, dst_start : dst_start + copy_len] = True
    return out, mask


def _compare_aligned_routes(
    rollout: torch.Tensor,
    train: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    metric_prefix: str,
) -> dict[str, float]:
    row_match = (rollout == train).all(dim=-1).all(dim=-1)
    valid = response_mask.to(dtype=torch.bool)
    token_count = int(valid.sum().item())
    if token_count <= 0:
        return {
            f"{metric_prefix}/response_token_match_rate": 0.0,
            f"{metric_prefix}/response_token_count": 0.0,
        }

    metrics = {
        f"{metric_prefix}/response_token_match_rate": float(row_match[valid].float().mean().item()),
        f"{metric_prefix}/response_token_count": float(token_count),
    }

    for layer_idx in range(rollout.shape[2]):
        layer_prefix = f"{metric_prefix}/layer_{layer_idx}"
        layer_row_match = (rollout[:, :, layer_idx] == train[:, :, layer_idx]).all(dim=-1)
        metrics[f"{layer_prefix}/response_token_match_rate"] = float(layer_row_match[valid].float().mean().item())
    return metrics
