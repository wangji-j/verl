# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

import torch
from tensordict import TensorDict

logger = logging.getLogger(__name__)


class FSDPRouterTrace:
    """Forward-hook based router observer for HF MoE models under FSDP."""

    def __init__(self) -> None:
        self.enabled = False
        self._active = False
        self._records: dict[int, list[torch.Tensor]] = {}
        self._handles = []
        self._layer_by_module_id: dict[int, int] = {}

    @property
    def num_layers(self) -> int:
        return len(self._layer_by_module_id)

    def install(self, module: torch.nn.Module) -> int:
        if self._handles:
            return len(self._handles)

        candidates = []
        for name, child in module.named_modules():
            class_name = child.__class__.__name__.lower()
            is_router = "router" in class_name
            is_moe_block = ("sparsemoe" in class_name) or ("moe" in class_name and hasattr(child, "top_k"))
            has_gate_router = hasattr(child, "gate") and hasattr(child, "top_k") and hasattr(child, "num_experts")
            if (is_router or is_moe_block or has_gate_router) and callable(getattr(child, "forward", None)):
                candidates.append((name, child))

        for layer_id, (name, router) in enumerate(candidates):
            self._layer_by_module_id[id(router)] = layer_id
            self._handles.append(router.register_forward_hook(self._make_hook(layer_id)))
            logger.info("FSDP router mismatch metrics: attached router hook layer=%s module=%s", layer_id, name)

        return len(self._handles)

    @contextmanager
    def capture(self, enabled: bool) -> Iterator[None]:
        old_active = self._active
        self._active = bool(enabled and self.enabled and self._handles)
        if self._active:
            self._records.clear()
        try:
            yield
        finally:
            self._active = old_active

    def consume(self, micro_batch: TensorDict) -> torch.Tensor | None:
        if not self._records:
            return None

        layers = sorted(self._records)
        layer_values = []
        for layer_id in layers:
            entries = self._records[layer_id]
            if len(entries) != 1:
                logger.warning(
                    "FSDP router mismatch metrics expected one record for layer %s, got %s; using the first",
                    layer_id,
                    len(entries),
                )
            layer_values.append(entries[0])

        min_rows = min(value.shape[0] for value in layer_values)
        min_topk = min(value.shape[1] for value in layer_values)
        routes = torch.stack([value[:min_rows, :min_topk] for value in layer_values], dim=1)
        routes = self._filter_valid_rows(routes, micro_batch)
        self._records.clear()

        input_ids = micro_batch["input_ids"]
        if not input_ids.is_nested:
            return routes

        offsets = input_ids.offsets().to(device=routes.device)
        expected_rows = int(offsets[-1].item()) if offsets.numel() else 0
        if routes.shape[0] < expected_rows:
            logger.warning(
                "FSDP router mismatch metrics captured fewer rows than input tokens: captured=%s expected=%s",
                routes.shape[0],
                expected_rows,
            )
            return None
        return torch.nested.nested_tensor_from_jagged(routes[:expected_rows], offsets)

    def _make_hook(self, layer_id: int):
        def hook(_module, _inputs, output):
            if not self._active:
                return
            selected = _extract_selected_experts(_module, output)
            if selected is None:
                return
            self._records.setdefault(layer_id, []).append(selected.detach().to(dtype=torch.int64, device="cpu"))

        return hook

    @staticmethod
    def _filter_valid_rows(routes: torch.Tensor, micro_batch: TensorDict) -> torch.Tensor:
        attention_mask = micro_batch.get("attention_mask", None)
        if attention_mask is None or attention_mask.is_nested:
            return routes
        valid = attention_mask.detach().to(dtype=torch.bool, device="cpu").reshape(-1)
        if valid.numel() == routes.shape[0]:
            return routes[valid]
        return routes


def _extract_selected_experts(module: torch.nn.Module, output) -> torch.Tensor | None:
    if isinstance(output, tuple) and len(output) >= 3 and isinstance(output[2], torch.Tensor):
        selected = output[2]
    elif isinstance(output, dict) and isinstance(output.get("selected_experts"), torch.Tensor):
        selected = output["selected_experts"]
    elif isinstance(output, torch.Tensor) and output.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        selected = output
    else:
        router_logits = None
        if isinstance(output, tuple) and len(output) >= 2 and isinstance(output[1], torch.Tensor):
            router_logits = output[1]
        elif isinstance(output, dict) and isinstance(output.get("router_logits"), torch.Tensor):
            router_logits = output["router_logits"]
        top_k = getattr(module, "top_k", None)
        if router_logits is None or top_k is None:
            return None
        selected = torch.topk(router_logits, int(top_k), dim=-1).indices

    if selected.ndim == 1:
        selected = selected.unsqueeze(-1)
    elif selected.ndim > 2:
        selected = selected.reshape(-1, selected.shape[-1])
    return selected
