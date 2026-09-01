# Copyright 2026
# Licensed under the Apache License, Version 2.0.
"""Prepare ZebraLogic using the official public prompts and gated private solutions."""

import argparse
import importlib
import json
import os
import sys

import datasets
import pandas as pd


DATA_SOURCE = "allenai/ZebraLogicBench"


def _load_official_prompt_builder(zeroeval_dir: str):
    source_dir = os.path.join(os.path.expanduser(zeroeval_dir), "src")
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"ZeroEval src directory not found: {source_dir}")
    sys.path.insert(0, source_dir)
    return importlib.import_module("_TEMPLATES").apply_lgp_grid_template


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zeroeval_dir", required=True, help="Checkout of https://github.com/WildEval/ZeroEval")
    parser.add_argument("--local_save_dir", default="~/data/zebra_logic")
    parser.add_argument("--base_validation_file", default=None, help="Optional parquet to merge with ZebraLogic.")
    parser.add_argument("--public_source", default=DATA_SOURCE)
    parser.add_argument("--private_source", default="WildEval/ZebraLogic")
    args = parser.parse_args()

    build_prompt = _load_official_prompt_builder(args.zeroeval_dir)
    public = datasets.load_dataset(args.public_source, "grid_mode", split="test")
    private = datasets.load_dataset(args.private_source, "grid_mode", split="test")
    private_solutions = {item["id"]: item["solution"] for item in private}

    def process(example: dict, index: int) -> dict:
        solution = private_solutions.get(example["id"])
        if solution is None:
            raise KeyError(f"Missing private ZebraLogic solution for {example['id']}")
        return {
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": build_prompt(example)}],
            "ability": "logic",
            "reward_model": {"style": "rule", "ground_truth": json.dumps(solution)},
            "extra_info": {"index": index, "id": example["id"], "size": example["size"]},
        }

    output = public.map(process, with_indices=True, remove_columns=public.column_names)
    output_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(output_dir, exist_ok=True)
    output.to_parquet(os.path.join(output_dir, "test.parquet"))

    if args.base_validation_file:
        base = pd.read_parquet(os.path.expanduser(args.base_validation_file))
        combined = pd.concat([base, output.to_pandas()], ignore_index=True, sort=False)
        combined.to_parquet(os.path.join(output_dir, "math_and_zebralogic.parquet"), index=False)


if __name__ == "__main__":
    main()
