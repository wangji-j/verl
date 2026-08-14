# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Preprocess HuggingFaceH4/MATH-500 into the verl parquet schema."""

import argparse
import json
import os

import datasets
import pandas as pd


DATA_SOURCE = "HuggingFaceH4/MATH-500"
FINAL_ANSWER_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def process_example(example: dict, index: int) -> dict:
    prompt = f"{example['problem']} {FINAL_ANSWER_INSTRUCTION}"
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": example["answer"]},
        "extra_info": {
            "split": "test",
            "index": index,
            "unique_id": example["unique_id"],
            "subject": example["subject"],
            "level": example["level"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_save_dir",
        default="~/data/math500",
        help="Directory where test.parquet and test_example.json are written.",
    )
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help="Optional local Hugging Face dataset path; defaults to HuggingFaceH4/MATH-500.",
    )
    parser.add_argument(
        "--aime_file",
        default=None,
        help="Optional AIME parquet to merge with MATH-500 into one schema-compatible validation parquet.",
    )
    args = parser.parse_args()

    source = args.local_dataset_path or DATA_SOURCE
    dataset = datasets.load_dataset(source, split="test")
    dataset = dataset.map(process_example, with_indices=True, remove_columns=dataset.column_names)

    output_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(os.path.join(output_dir, "test.parquet"))
    with open(os.path.join(output_dir, "test_example.json"), "w", encoding="utf-8") as f:
        json.dump(dataset[0], f, ensure_ascii=True, indent=2)

    if args.aime_file:
        aime = pd.read_parquet(os.path.expanduser(args.aime_file))
        math500 = dataset.to_pandas()
        combined = pd.concat([aime, math500], ignore_index=True, sort=False)
        combined.to_parquet(os.path.join(output_dir, "aime24_aime25_x32_math500.parquet"), index=False)


if __name__ == "__main__":
    main()
