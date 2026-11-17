#!/usr/bin/env python3
"""Export FULL training curves (history) for a filtered set of wandb runs.

Unlike run.summary (final values only), run.scan_history() returns every logged
step with no downsampling. One CSV per run in long format (rows = steps,
columns = metrics), plus an index.csv listing what was exported.

Usage examples:
    export WANDB_BASE_URL=https://wandb1.sii.edu.cn/     # if self-hosted
    export WANDB_API_KEY=<key with read access to the entity>

    # (A) explicit run names:
    python3 tools/wandb_export_curves.py \
        --entity gzy1 --project "router drift control" \
        --names coding-baseline coding-TIS coding-R3 coding-top8RDC-TIS

    # (B) name regex (e.g. all coding runs):
    python3 tools/wandb_export_curves.py \
        --entity gzy1 --project "router drift control" \
        --name-regex '^coding-'

    # (C) only a few metrics (faster; omit to grab everything):
    python3 tools/wandb_export_curves.py --entity gzy1 --project "router drift control" \
        --name-regex '^coding-' \
        --keys critic/rewards/mean val-core/livecodebench/acc/mean@1 actor/entropy
"""

import argparse
import pathlib

import pandas as pd
import wandb


def build_filters(names, name_regex):
    if names:
        return {"display_name": {"$in": list(names)}}
    if name_regex:
        return {"display_name": {"$regex": name_regex}}
    return None  # no filter -> all runs in the project


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--names", nargs="*", default=None, help="exact run names to export")
    ap.add_argument("--name-regex", default=None, help="regex on run display name")
    ap.add_argument("--keys", nargs="*", default=None,
                    help="only export these metric keys (faster); default = all")
    ap.add_argument("--out", default="wandb_curves")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(exist_ok=True)

    api = wandb.Api(timeout=120)
    filters = build_filters(args.names, args.name_regex)
    runs = api.runs(f"{args.entity}/{args.project}", filters=filters) if filters \
        else api.runs(f"{args.entity}/{args.project}")

    index_rows = []
    print(f"matched runs: {len(runs)}")
    for r in runs:
        print(f"  exporting: {r.name}  (state={r.state})")
        # scan_history = FULL history, no sampling. keys=None -> every metric.
        rows = list(r.scan_history(keys=args.keys)) if args.keys else list(r.scan_history())
        if not rows:
            print(f"    (no history rows, skipped)")
            continue
        df = pd.DataFrame(rows)
        # put _step first if present
        if "_step" in df.columns:
            cols = ["_step"] + [c for c in df.columns if c != "_step"]
            df = df[cols]
        safe = r.name.replace("/", "_").replace(" ", "_")
        fp = out / f"{safe}.csv"
        df.to_csv(fp, index=False)
        index_rows.append({"name": r.name, "id": r.id, "state": r.state,
                           "rows": len(df), "cols": df.shape[1], "file": fp.name})
        print(f"    -> {fp}  ({len(df)} steps, {df.shape[1]} metrics)")

    if index_rows:
        pd.DataFrame(index_rows).to_csv(out / "index.csv", index=False)
        print(f"\nindex written: {out/'index.csv'}")
    else:
        print("\nNo runs matched / no history exported. Check --entity/--project/--names.")


if __name__ == "__main__":
    main()
