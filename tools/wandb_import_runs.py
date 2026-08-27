#!/usr/bin/env python3
"""Import runs exported by wandb_export_runs.py into another entity (stage 2 of 2).

Usage:
    export WANDB_BASE_URL=https://wandb1.sii.edu.cn/   # target instance
    export WANDB_API_KEY=<TARGET account key>
    python3 tools/wandb_import_runs.py --entity <dst_entity> --project "router drift control"

Reads wandb_export/<run_id>/ and creates a new run per exported run,
replaying history with the original step axis. Re-runnable: runs already
marked `imported` are skipped.
"""

import argparse
import json
import math
import pathlib

import wandb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True, help="target entity (username/team)")
    ap.add_argument("--project", required=True, help="target project name")
    ap.add_argument("--src", default="wandb_export", help="export dir from stage 1")
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    dirs = sorted(p for p in src.iterdir() if (p / "done").exists())
    print(f"exported runs found: {len(dirs)}")

    for d in dirs:
        if (d / "imported").exists():
            print("skip (imported):", d.name)
            continue
        meta = json.loads((d / "meta.json").read_text())
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            name=meta["name"],
            config=meta["config"],
            tags=meta.get("tags") or None,
            group=meta.get("group") or None,
            notes=meta.get("notes") or None,
            resume="never",
            reinit=True,
        )
        n = 0
        for line in (d / "history.jsonl").open():
            row = json.loads(line)
            step = row.pop("_step", None)
            row = {
                k: v
                for k, v in row.items()
                if not k.startswith("_")
                and v is not None
                and not (isinstance(v, float) and math.isnan(v))
            }
            if row:
                run.log(row, step=int(step) if step is not None else None)
                n += 1
        for k, v in meta["summary"].items():
            try:
                run.summary[k] = v
            except Exception:
                pass
        run.finish()
        (d / "imported").write_text("ok")
        print(f"imported: {meta['name']}  rows={n}")


if __name__ == "__main__":
    main()
