#!/usr/bin/env python3
"""Analyze whether router drift leads policy and validation degradation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal.datastore import DataStore


REPO_ROOT = Path(__file__).resolve().parents[2]
WANDB_ROOT = REPO_ROOT / "wandb"
OUT_DIR = Path(__file__).resolve().parent

RUN_GROUPS = {
    "usage_l1_baseline": ["run-20260708_225736-pvna8pyr"],
    "sequence_mismatch_tis_no_rs": [
        "run-20260626_080321-modp2i13",
        "run-20260701_215809-g8haovvm",
    ],
    "sequence_mismatch_threshold025_tis": [
        "run-20260626_073707-sufv5is1",
        "run-20260701_215818-rsmtrlmx",
    ],
    "usage_l1_top8_rs": ["run-20260707_225928-fun7svgs"],
    "usage_l1_lengthbucket_top8_tis": ["run-20260710_004624-64ch1f99"],
    "usage_l1_madz_top8_cap2_tis": ["run-20260711_074148-a9wjqbvn"],
}

ROUTER_KEYS = (
    "router/rollout_vs_train/expert_usage_l1_mean",
    "router/rollout_vs_train/seq_mismatch_mean",
)

TARGET_KEYS = (
    "training/rollout_probs_diff_mean",
    "training/rollout_actor_probs_pearson_corr",
    "rollout_corr/kl",
    "actor/pg_clipfrac",
    "actor/ppo_kl",
    "actor/grad_norm",
    "actor/entropy",
    "critic/rewards/mean",
    "response_length/mean",
)

VAL_KEYS = (
    "val-core/AIME24/acc/mean@1",
    "val-core/AIME25/acc/mean@1",
)


def item_key(item) -> str:
    return item.key or "/".join(item.nested_key)


def load_history(run_dir: str) -> tuple[str, dict[int, dict[str, float]]]:
    run_id = run_dir.rsplit("-", 1)[-1]
    path = WANDB_ROOT / run_dir / f"run-{run_id}.wandb"
    store = DataStore()
    store.open_for_scan(str(path))
    name = run_dir
    rows: dict[int, dict[str, float]] = {}
    while True:
        data = store.scan_data()
        if data is None:
            break
        record = pb.Record()
        record.ParseFromString(data)
        if record.HasField("run") and record.run.display_name:
            name = record.run.display_name
        if not record.HasField("history"):
            continue
        row = {}
        for item in record.history.item:
            key = item_key(item)
            try:
                value = json.loads(item.value_json)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(value, (int, float)) and np.isfinite(value):
                row[key] = float(value)
        if "_step" in row:
            rows.setdefault(int(row["_step"]), {}).update(row)
    return name, rows


def merge_group(run_dirs: list[str]) -> tuple[list[str], dict[int, dict[str, float]]]:
    names = []
    merged: dict[int, dict[str, float]] = {}
    for run_dir in run_dirs:
        name, rows = load_history(run_dir)
        names.append(name)
        for step, row in rows.items():
            merged.setdefault(step, {}).update(row)
    return names, merged


def corr(x: np.ndarray, y: np.ndarray) -> dict[str, float | int] | None:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 6 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return {
        "n": int(x.size),
        "pearson": float(pearsonr(x, y).statistic),
        "pearson_p": float(pearsonr(x, y).pvalue),
        "spearman": float(spearmanr(x, y).statistic),
        "spearman_p": float(spearmanr(x, y).pvalue),
    }


def lagged_pairs(rows, source, target, lag, diff=False, target_sparse=False):
    steps = sorted(rows)
    if diff:
        source_values = {}
        target_values = {}
        for prev, step in zip(steps, steps[1:]):
            if step != prev + 1:
                continue
            if source in rows[step] and source in rows[prev]:
                source_values[step] = rows[step][source] - rows[prev][source]
            if target in rows[step] and target in rows[prev]:
                target_values[step] = rows[step][target] - rows[prev][target]
    else:
        source_values = {s: r[source] for s, r in rows.items() if source in r}
        target_values = {s: r[target] for s, r in rows.items() if target in r}

    xs, ys = [], []
    for source_step, value in source_values.items():
        target_step = source_step + lag
        if target_step in target_values:
            xs.append(value)
            ys.append(target_values[target_step])
        elif target_sparse:
            # Validation is normally logged every 10 steps. Only exact alignment is
            # accepted to avoid silently changing the requested lag.
            continue
    return np.asarray(xs), np.asarray(ys)


def add_validation_mean(rows):
    for row in rows.values():
        vals = [row[k] for k in VAL_KEYS if k in row]
        if vals:
            row["validation/mean_acc"] = float(np.mean(vals))


def analyze_group(run_dirs: list[str]) -> dict:
    names, rows = merge_group(run_dirs)
    add_validation_mean(rows)
    steps = sorted(rows)
    router_key = next(
        (key for key in ROUTER_KEYS if any(key in row for row in rows.values())),
        None,
    )
    result = {
        "run_names": names,
        "step_min": min(steps) if steps else None,
        "step_max": max(steps) if steps else None,
        "history_steps": len(steps),
        "router_key": router_key,
        "lagged": {},
        "delta_lagged": {},
        "validation_lagged": {},
        "validation_points": sum("validation/mean_acc" in r for r in rows.values()),
    }
    if router_key is None:
        return result

    for target in TARGET_KEYS:
        lag_results = {}
        delta_results = {}
        for lag in (0, 1, 2, 5, 10, 20, 30):
            x, y = lagged_pairs(rows, router_key, target, lag)
            value = corr(x, y)
            if value:
                lag_results[str(lag)] = value
            x, y = lagged_pairs(rows, router_key, target, lag, diff=True)
            value = corr(x, y)
            if value:
                delta_results[str(lag)] = value
        if lag_results:
            result["lagged"][target] = lag_results
        if delta_results:
            result["delta_lagged"][target] = delta_results

    for lag in (0, 10, 20, 30, 40, 50):
        x, y = lagged_pairs(
            rows, router_key, "validation/mean_acc", lag, target_sparse=True
        )
        value = corr(x, y)
        if value:
            result["validation_lagged"][str(lag)] = value
    return result


def best_lag(values: dict[str, dict], direction: str) -> tuple[str, dict] | None:
    candidates = [(lag, value) for lag, value in values.items() if value["n"] >= 8]
    if not candidates:
        return None
    key = (lambda pair: pair[1]["spearman"])
    return (min(candidates, key=key) if direction == "negative" else max(candidates, key=key))


def render_markdown(results: dict) -> str:
    lines = [
        "# Router Drift Lead-Lag Analysis",
        "",
        "This is an observational analysis of local W&B histories associated with existing router dumps.",
        "Raw level correlations can be trend-driven; first-difference results are reported separately.",
        "",
        "| Experiment | Steps | Router metric | Val points | Best lead to prob diff | Best lead to validation |",
        "|---|---:|---|---:|---|---|",
    ]
    for name, result in results.items():
        prob = best_lag(result["lagged"].get("training/rollout_probs_diff_mean", {}), "positive")
        val = best_lag(result["validation_lagged"], "negative")
        prob_text = "n/a" if not prob else f"k={prob[0]}, rho={prob[1]['spearman']:.3f}, n={prob[1]['n']}"
        val_text = "n/a" if not val else f"k={val[0]}, rho={val[1]['spearman']:.3f}, n={val[1]['n']}"
        lines.append(
            f"| {name} | {result['step_min']}-{result['step_max']} | "
            f"`{result['router_key']}` | {result['validation_points']} | {prob_text} | {val_text} |"
        )
    lines.extend(["", "## Per-experiment details", ""])
    for name, result in results.items():
        lines.extend([f"### {name}", ""])
        for target in (
            "training/rollout_probs_diff_mean",
            "training/rollout_actor_probs_pearson_corr",
            "actor/pg_clipfrac",
            "actor/grad_norm",
            "actor/entropy",
            "critic/rewards/mean",
            "response_length/mean",
        ):
            level = result["lagged"].get(target, {})
            delta = result["delta_lagged"].get(target, {})
            best_level = best_lag(level, "negative" if target in {"training/rollout_actor_probs_pearson_corr", "actor/entropy"} else "positive")
            best_delta = best_lag(delta, "negative" if target in {"training/rollout_actor_probs_pearson_corr", "actor/entropy"} else "positive")
            if best_level or best_delta:
                level_text = "n/a" if not best_level else f"k={best_level[0]}, rho={best_level[1]['spearman']:.3f}"
                delta_text = "n/a" if not best_delta else f"k={best_delta[0]}, rho={best_delta[1]['spearman']:.3f}"
                lines.append(f"- `{target}`: level {level_text}; first-difference {delta_text}")
        if result["validation_lagged"]:
            vals = ", ".join(
                f"k={lag}: rho={value['spearman']:.3f} (n={value['n']})"
                for lag, value in result["validation_lagged"].items()
            )
            lines.append(f"- validation lead-lag: {vals}")
        lines.append("")
    return "\n".join(lines)


def main():
    results = {name: analyze_group(runs) for name, runs in RUN_GROUPS.items()}
    json_path = OUT_DIR / "router_lead_lag_analysis.json"
    md_path = OUT_DIR / "router_lead_lag_analysis.md"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(results), encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
