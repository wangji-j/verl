#!/usr/bin/env python3
"""Export all runs of a wandb project to local files (stage 1 of 2).

Usage:
    export WANDB_BASE_URL=https://wandb1.sii.edu.cn/
    export WANDB_API_KEY=<SOURCE account key>
    python3 tools/wandb_export_runs.py --entity <src_entity> --project "router drift control"

Writes wandb_export/<run_id>/{meta.json,history.jsonl,done}.
Re-runnable: runs already marked `done` are skipped.
"""

import argparse
import json
import pathlib

import wandb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True, help="source entity (username/team)")
    ap.add_argument("--project", required=True, help="source project name")
    ap.add_argument("--out", default="wandb_export", help="output dir")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(exist_ok=True)

    api = wandb.Api(timeout=120)
    runs = api.runs(f"{args.entity}/{args.project}")
    print(f"total runs in {args.entity}/{args.project}: {len(runs)}")

    for r in runs:
        d = out / r.id
        if (d / "done").exists():
            print("skip (done):", r.name)
            continue
        d.mkdir(exist_ok=True)
        meta = {
            "id": r.id,
            "name": r.name,
            "state": r.state,
            "config": {k: v for k, v in r.config.items() if not k.startswith("_")},
            "summary": {
                k: v
                for k, v in dict(r.summary).items()
                if not k.startswith("_") and isinstance(v, (int, float, str, bool))
            },
            "tags": list(r.tags),
            "notes": r.notes,
            "group": r.group,
        }
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
        n = 0
        with (d / "history.jsonl").open("w") as f:
            for row in r.scan_history():  # full-resolution history
                slim = {
                    k: v
                    for k, v in row.items()
                    if isinstance(v, (int, float, str, bool, type(None)))
                }
                f.write(json.dumps(slim, ensure_ascii=False) + "\n")
                n += 1
        (d / "done").write_text("ok")
        print(f"exported: {r.name}  rows={n}")


if __name__ == "__main__":
    main()
