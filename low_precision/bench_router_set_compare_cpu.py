#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verl.utils.router_mismatch_metrics import compute_router_mismatch_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="CPU benchmark for unordered top-k router expert set comparison.")
    parser.add_argument(
        "--mode",
        choices=("core", "exact_metrics"),
        default="core",
        help="core only compares top-k sets; exact_metrics calls the same router mismatch metric path used by training.",
    )
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--rollout-extra-tokens", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--chunk-tokens", type=int, default=16)
    parser.add_argument("--valid-token-fraction", type=float, default=1.0)
    parser.add_argument("--alignments", default="1", help="Comma-separated alignment candidates, e.g. 1 or 0,-1,1.")
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads. 0 keeps current default.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output-json", default=None, help="Optional path to write structured benchmark results.")
    parser.add_argument(
        "--allocate-full",
        action="store_true",
        help="Allocate full tensors instead of reusing one chunk. This needs large CPU memory.",
    )
    return parser.parse_args()


def compare_unordered_topk(rollout: torch.Tensor, train: torch.Tensor):
    rollout_sorted = torch.sort(rollout, dim=-1).values
    train_sorted = torch.sort(train, dim=-1).values
    layer_match = (rollout_sorted == train_sorted).all(dim=-1)
    token_match = layer_match.all(dim=-1)
    return layer_match, token_match


def make_chunk(batch: int, tokens: int, layers: int, topk: int, num_experts: int):
    rollout = torch.randint(0, num_experts, (batch, tokens, layers, topk), dtype=torch.int32, device="cpu")
    train = torch.randint(0, num_experts, (batch, tokens, layers, topk), dtype=torch.uint8, device="cpu")
    return rollout, train


def make_response_mask(batch: int, tokens: int, valid_fraction: float):
    valid_fraction = min(max(valid_fraction, 0.0), 1.0)
    valid_tokens = int(round(tokens * valid_fraction))
    mask = torch.zeros((batch, tokens), dtype=torch.bool, device="cpu")
    if valid_tokens > 0:
        mask[:, :valid_tokens] = True
    return mask


def parse_alignments(value: str):
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def run_core_once(args):
    chunks = (args.tokens + args.chunk_tokens - 1) // args.chunk_tokens
    matched_layers = 0
    matched_tokens = 0
    total_layers = 0
    total_tokens = 0

    if args.allocate_full:
        rollout_full, train_full = make_chunk(args.batch, args.tokens, args.layers, args.topk, args.num_experts)
    else:
        rollout_chunk, train_chunk = make_chunk(args.batch, args.chunk_tokens, args.layers, args.topk, args.num_experts)

    t0 = time.perf_counter()
    for chunk_idx in range(chunks):
        cur_tokens = min(args.chunk_tokens, args.tokens - chunk_idx * args.chunk_tokens)
        if args.allocate_full:
            rollout = rollout_full[:, chunk_idx * args.chunk_tokens : chunk_idx * args.chunk_tokens + cur_tokens]
            train = train_full[:, chunk_idx * args.chunk_tokens : chunk_idx * args.chunk_tokens + cur_tokens]
        elif cur_tokens == args.chunk_tokens:
            rollout, train = rollout_chunk, train_chunk
        else:
            rollout, train = rollout_chunk[:, :cur_tokens], train_chunk[:, :cur_tokens]

        layer_match, token_match = compare_unordered_topk(rollout, train)
        matched_layers += int(layer_match.sum().item())
        matched_tokens += int(token_match.sum().item())
        total_layers += layer_match.numel()
        total_tokens += token_match.numel()

    elapsed = time.perf_counter() - t0
    return {
        "elapsed_s": elapsed,
        "layer_match_rate": matched_layers / max(total_layers, 1),
        "token_match_rate": matched_tokens / max(total_tokens, 1),
        "total_layer_positions": total_layers,
        "total_token_positions": total_tokens,
    }


def run_exact_metrics_once(args):
    alignments = parse_alignments(args.alignments)
    chunks = 1 if args.allocate_full else (args.tokens + args.chunk_tokens - 1) // args.chunk_tokens
    metric_elapsed = 0.0
    last_metrics = {}
    alignment_counts: dict[int, int] = {}

    if args.allocate_full:
        rollout_full = torch.randint(
            0,
            args.num_experts,
            (args.batch, args.tokens + args.rollout_extra_tokens, args.layers, args.topk),
            dtype=torch.int32,
            device="cpu",
        )
        train_full = torch.randint(
            0,
            args.num_experts,
            (args.batch, args.tokens, args.layers, args.topk),
            dtype=torch.uint8,
            device="cpu",
        )
        mask_full = make_response_mask(args.batch, args.tokens, args.valid_token_fraction)
    else:
        rollout_chunk = torch.randint(
            0,
            args.num_experts,
            (args.batch, args.chunk_tokens + args.rollout_extra_tokens, args.layers, args.topk),
            dtype=torch.int32,
            device="cpu",
        )
        train_chunk = torch.randint(
            0,
            args.num_experts,
            (args.batch, args.chunk_tokens, args.layers, args.topk),
            dtype=torch.uint8,
            device="cpu",
        )
        mask_chunk = make_response_mask(args.batch, args.chunk_tokens, args.valid_token_fraction)

    t0 = time.perf_counter()
    for chunk_idx in range(chunks):
        if args.allocate_full:
            rollout = rollout_full
            train = train_full
            mask = mask_full
        else:
            cur_tokens = min(args.chunk_tokens, args.tokens - chunk_idx * args.chunk_tokens)
            if cur_tokens == args.chunk_tokens:
                rollout, train, mask = rollout_chunk, train_chunk, mask_chunk
            else:
                rollout = rollout_chunk[:, : cur_tokens + args.rollout_extra_tokens]
                train = train_chunk[:, :cur_tokens]
                mask = mask_chunk[:, :cur_tokens]

        result = compute_router_mismatch_metrics(
            rollout,
            train,
            mask,
            metric_prefix="router/rollout_vs_train",
            alignments=alignments,
            metric_mode="exact_set",
        )
        last_metrics = result.metrics
        alignment_counts[result.alignment] = alignment_counts.get(result.alignment, 0) + 1
    metric_elapsed = time.perf_counter() - t0

    prefix = "router/rollout_vs_train"
    return {
        "elapsed_s": metric_elapsed,
        "alignment_counts": alignment_counts,
        "response_token_match_rate": float(last_metrics.get(f"{prefix}/response_token_match_rate", 0.0)),
        "response_token_count_last_chunk": float(last_metrics.get(f"{prefix}/response_token_count", 0.0)),
        "seq_mismatch_mean_last_chunk": float(last_metrics.get(f"{prefix}/seq_mismatch_mean", 0.0)),
        "seq_mismatch_max_last_chunk": float(last_metrics.get(f"{prefix}/seq_mismatch_max", 0.0)),
    }


def run_once(args):
    if args.mode == "exact_metrics":
        return run_exact_metrics_once(args)
    return run_core_once(args)


def main():
    args = parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    config = {
        "mode": args.mode,
        "batch": args.batch,
        "tokens": args.tokens,
        "rollout_extra_tokens": args.rollout_extra_tokens,
        "layers": args.layers,
        "topk": args.topk,
        "num_experts": args.num_experts,
        "chunk_tokens": args.chunk_tokens,
        "valid_token_fraction": args.valid_token_fraction,
        "alignments": parse_alignments(args.alignments),
        "threads": torch.get_num_threads(),
        "rollout_dtype": "torch.int32",
        "train_dtype": "torch.uint8",
        "allocate_full": args.allocate_full,
    }
    print("config:", config)

    for _ in range(args.warmup):
        run_once(args)

    results = []
    output = {"config": config, "results": []}
    for _ in range(args.repeat):
        result = run_once(args)
        results.append(result["elapsed_s"])
        output["results"].append(result)
        print("result:", result)

    output["elapsed_s_mean"] = sum(results) / max(len(results), 1)
    print("elapsed_s_mean:", output["elapsed_s_mean"])
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
