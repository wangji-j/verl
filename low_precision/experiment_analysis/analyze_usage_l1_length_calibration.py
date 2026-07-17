#!/usr/bin/env python3
"""Evaluate response-length calibration for Expert Usage L1 dumps."""

from __future__ import annotations

import argparse
import gc
import glob
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


def as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().float().numpy()


def local_median_mad(length: np.ndarray, score: np.ndarray, neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    """Estimate continuous conditional median/MAD using sliding nearest-neighbor windows."""
    order = np.argsort(length, kind="stable")
    sorted_score = score[order]
    count = len(score)
    width = min(max(neighbors, 32), count)
    half = width // 2
    centers = np.unique(np.linspace(0, count - 1, min(65, count), dtype=np.int64))
    knot_length = []
    knot_median = []
    knot_mad = []
    for center in centers:
        start = max(0, min(count - width, center - half))
        stop = start + width
        window = sorted_score[start:stop]
        median = float(np.median(window))
        mad = float(1.4826 * np.median(np.abs(window - median)))
        knot_length.append(float(np.median(length[order[start:stop]])))
        knot_median.append(median)
        knot_mad.append(mad)

    knot_length = np.asarray(knot_length)
    unique_length, unique_indices = np.unique(knot_length, return_index=True)
    median_curve = np.interp(length, unique_length, np.asarray(knot_median)[unique_indices])
    mad_curve = np.interp(length, unique_length, np.asarray(knot_mad)[unique_indices])
    positive = mad_curve[mad_curve > 0]
    floor = float(np.median(positive) * 0.1) if positive.size else 1e-8
    z_score = (score - median_curve) / np.maximum(mad_curve, floor)
    return z_score, median_curve


def calibrate_censored_max(length: np.ndarray, score: np.ndarray, calibrated: np.ndarray) -> np.ndarray:
    """Calibrate max-length responses separately because their lengths are right-censored."""
    result = calibrated.copy()
    censored = length >= length.max()
    if censored.sum() < 32:
        return result
    values = score[censored]
    median = float(np.median(values))
    mad = float(1.4826 * np.median(np.abs(values - median)))
    result[censored] = (values - median) / max(mad, 1e-8)
    return result


def conditional_percentile(length: np.ndarray, score: np.ndarray, neighbors: int) -> np.ndarray:
    """Return the score percentile among responses with neighboring lengths."""
    order = np.argsort(length, kind="stable")
    count = len(score)
    width = min(max(neighbors, 32), count)
    half = width // 2
    percentile = np.empty(count, dtype=np.float64)
    for position, original_index in enumerate(order):
        start = max(0, min(count - width, position - half))
        window = score[order[start : start + width]]
        value = score[original_index]
        percentile[original_index] = ((window < value).sum() + 0.5 * (window == value).sum()) / width

    # Max-length responses are right-censored and form a separate point mass.
    censored_indices = np.flatnonzero(length >= length.max())
    if len(censored_indices) >= 32:
        values = score[censored_indices]
        for original_index in censored_indices:
            value = score[original_index]
            percentile[original_index] = ((values < value).sum() + 0.5 * (values == value).sum()) / len(values)
    return percentile


def safe_corr(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3 or np.ptp(left[finite]) == 0 or np.ptp(right[finite]) == 0:
        return {"pearson": float("nan"), "spearman": float("nan")}
    return {
        "pearson": float(pearsonr(left[finite], right[finite]).statistic),
        "spearman": float(spearmanr(left[finite], right[finite]).statistic),
    }


def select_top(score: np.ndarray, fraction: float) -> np.ndarray:
    count = max(1, int(np.ceil(len(score) * fraction)))
    selected = np.zeros(len(score), dtype=bool)
    selected[np.argpartition(score, -count)[-count:]] = True
    return selected


def selection_stats(mask: np.ndarray, length: np.ndarray, reward: np.ndarray) -> dict[str, float]:
    kept = ~mask
    correct = reward > 0
    return {
        "selected_count": int(mask.sum()),
        "selected_length_mean": float(length[mask].mean()),
        "kept_length_mean": float(length[kept].mean()),
        "length_mean_ratio_selected_over_kept": float(length[mask].mean() / length[kept].mean()),
        "selected_length_median": float(np.median(length[mask])),
        "kept_length_median": float(np.median(length[kept])),
        "selected_correct_rate": float(correct[mask].mean()),
        "kept_correct_rate": float(correct[kept].mean()),
        "correct_rate_gap_selected_minus_kept": float(correct[mask].mean() - correct[kept].mean()),
        "selected_reward_mean": float(reward[mask].mean()),
        "kept_reward_mean": float(reward[kept].mean()),
        "selected_max_length_rate": float((length[mask] >= length.max()).mean()),
        "kept_max_length_rate": float((length[kept] >= length.max()).mean()),
    }


def length_decile_rates(mask: np.ndarray, length: np.ndarray) -> list[dict[str, float]]:
    # Rank-based deciles remain well-defined when many responses share the max length.
    order = np.argsort(length, kind="stable")
    bins = np.empty(len(length), dtype=np.int64)
    bins[order] = np.minimum(9, np.arange(len(length)) * 10 // len(length))
    result = []
    for decile in range(10):
        current = bins == decile
        result.append(
            {
                "decile": decile + 1,
                "count": int(current.sum()),
                "length_mean": float(length[current].mean()),
                "selection_rate": float(mask[current].mean()),
            }
        )
    return result


def summarize_records(records: list[dict], fractions: list[float]) -> dict:
    output: dict[str, object] = {}
    for fraction in fractions:
        key = f"top_{fraction:.0%}"
        output[key] = {}
        for mode in ("raw", "calibrated", "censor_aware", "conditional_percentile"):
            entries = [record["fractions"][key][mode] for record in records]
            scalar_keys = [name for name, value in entries[0].items() if name != "length_deciles" and np.isscalar(value)]
            summary = {
                name: float(np.mean([entry[name] for entry in entries])) for name in scalar_keys
            }
            deciles = []
            for index in range(10):
                deciles.append(
                    {
                        "decile": index + 1,
                        "length_mean": float(np.mean([entry["length_deciles"][index]["length_mean"] for entry in entries])),
                        "selection_rate": float(
                            np.mean([entry["length_deciles"][index]["selection_rate"] for entry in entries])
                        ),
                    }
                )
            summary["length_deciles"] = deciles
            output[key][mode] = summary
        output[key]["selection_jaccard"] = float(
            np.mean([record["fractions"][key]["selection_jaccard"] for record in records])
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--neighbors", type=int, default=256)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    paths = sorted(glob.glob(str(args.dump_dir / "*.pt")))
    if args.max_files > 0:
        paths = paths[: args.max_files]
    if not paths:
        raise SystemExit(f"No .pt dumps found in {args.dump_dir}")

    fractions = [0.03, 0.05, 0.08]
    records = []
    pooled = {"length": [], "score": [], "calibrated": [], "reward": []}
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        summary = payload["response_summary"]
        length = as_numpy(summary["seq_valid_token_count"])
        score = as_numpy(summary["seq_mismatch"])
        reward = as_numpy(summary["seq_reward"])
        calibrated, median_curve = local_median_mad(length, score, args.neighbors)
        censor_aware = calibrate_censored_max(length, score, calibrated)
        percentile = conditional_percentile(length, score, args.neighbors)
        record = {
            "step": int(payload["metadata"]["global_step"]),
            "count": len(score),
            "correlations": {
                "raw_vs_length": safe_corr(score, length),
                "calibrated_vs_length": safe_corr(calibrated, length),
                "raw_vs_reward": safe_corr(score, reward),
                "calibrated_vs_reward": safe_corr(calibrated, reward),
                "censor_aware_vs_length": safe_corr(censor_aware, length),
                "censor_aware_vs_reward": safe_corr(censor_aware, reward),
                "conditional_percentile_vs_length": safe_corr(percentile, length),
                "conditional_percentile_vs_reward": safe_corr(percentile, reward),
                "median_curve_vs_length": safe_corr(median_curve, length),
            },
            "fractions": {},
        }
        for fraction in fractions:
            key = f"top_{fraction:.0%}"
            raw_mask = select_top(score, fraction)
            calibrated_mask = select_top(calibrated, fraction)
            censor_aware_mask = select_top(censor_aware, fraction)
            percentile_mask = select_top(percentile, fraction)
            intersection = int((raw_mask & calibrated_mask).sum())
            union = int((raw_mask | calibrated_mask).sum())
            record["fractions"][key] = {
                "raw": {
                    **selection_stats(raw_mask, length, reward),
                    "length_deciles": length_decile_rates(raw_mask, length),
                },
                "calibrated": {
                    **selection_stats(calibrated_mask, length, reward),
                    "length_deciles": length_decile_rates(calibrated_mask, length),
                },
                "censor_aware": {
                    **selection_stats(censor_aware_mask, length, reward),
                    "length_deciles": length_decile_rates(censor_aware_mask, length),
                },
                "conditional_percentile": {
                    **selection_stats(percentile_mask, length, reward),
                    "length_deciles": length_decile_rates(percentile_mask, length),
                },
                "selection_jaccard": intersection / union if union else 1.0,
            }
        records.append(record)
        pooled["length"].append(length)
        pooled["score"].append(score)
        pooled["calibrated"].append(calibrated)
        pooled.setdefault("censor_aware", []).append(censor_aware)
        pooled.setdefault("conditional_percentile", []).append(percentile)
        pooled["reward"].append(reward)
        del payload, summary
        gc.collect()

    pooled = {key: np.concatenate(values) for key, values in pooled.items()}
    result = {
        "source": str(args.dump_dir),
        "files": len(paths),
        "responses": int(len(pooled["score"])),
        "calibration": {
            "method": "per-step continuous local median/MAD interpolation",
            "neighbors": args.neighbors,
            "knots": 65,
        },
        "pooled_correlations": {
            "raw_vs_length": safe_corr(pooled["score"], pooled["length"]),
            "calibrated_vs_length": safe_corr(pooled["calibrated"], pooled["length"]),
            "raw_vs_reward": safe_corr(pooled["score"], pooled["reward"]),
            "calibrated_vs_reward": safe_corr(pooled["calibrated"], pooled["reward"]),
            "censor_aware_vs_length": safe_corr(pooled["censor_aware"], pooled["length"]),
            "censor_aware_vs_reward": safe_corr(pooled["censor_aware"], pooled["reward"]),
            "conditional_percentile_vs_length": safe_corr(pooled["conditional_percentile"], pooled["length"]),
            "conditional_percentile_vs_reward": safe_corr(pooled["conditional_percentile"], pooled["reward"]),
        },
        "mean_per_step_correlations": {
            name: {
                metric: float(np.nanmean([record["correlations"][name][metric] for record in records]))
                for metric in ("pearson", "spearman")
            }
            for name in records[0]["correlations"]
        },
        "selection": summarize_records(records, fractions),
        "per_step": records,
        "limitations": [
            "The dumps contain per-response Expert Usage L1 but not per-response expert histograms/counts.",
            "Therefore this analysis evaluates empirical length calibration, not a multinomial-null bootstrap.",
            "Correctness is defined as seq_reward > 0.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    if not args.quiet:
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ("files", "responses", "pooled_correlations", "mean_per_step_correlations", "selection")
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
