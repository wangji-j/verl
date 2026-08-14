# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Build deterministic STEM and logic transfer benchmarks in verl format."""

import argparse
import json
import os
import random
import string
from pathlib import Path

import datasets
import pandas as pd
from huggingface_hub import snapshot_download


MMLU_SOURCE = "TIGER-Lab/MMLU-Pro-STEM"
CEVAL_SOURCE = "ceval/ceval-exam-STEM"
GPQA_SOURCE = "Idavidrein/gpqa-diamond"
AUTOLOGI_SOURCE = "8188zq/AutoLogi-en"

MMLU_DATASET = "TIGER-Lab/MMLU-Pro"
MMLU_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
MMLU_STEM_CATEGORIES = (
    "biology",
    "chemistry",
    "computer science",
    "engineering",
    "math",
    "physics",
)

CEVAL_DATASET = "ceval/ceval-exam"
CEVAL_REVISION = "617524a00b307ff6f9933702f724131fe12ca7ce"
CEVAL_STEM_SUBJECTS = (
    "advanced_mathematics",
    "college_chemistry",
    "college_physics",
    "college_programming",
    "computer_architecture",
    "computer_network",
    "discrete_mathematics",
    "electrical_engineer",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_mathematics",
    "high_school_physics",
    "logic",
    "middle_school_biology",
    "middle_school_chemistry",
    "middle_school_mathematics",
    "middle_school_physics",
    "operating_system",
    "probability_and_statistics",
)

SCHEMA_COLUMNS = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
EN_ANSWER_INSTRUCTION = (
    "Think step by step. End your response with a separate line in the exact format `Final Answer: X`, "
    "where X is the letter of the correct option."
)
ZH_ANSWER_INSTRUCTION = "请逐步思考。回答末尾必须另起一行，严格使用`最终答案：X`格式，其中X是正确选项的字母。"


def _record(
    *,
    data_source: str,
    content: str,
    ability: str,
    answer: str,
    split: str,
    index: int,
    source_id: str,
    category: str,
    language: str,
) -> dict:
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": content}],
        "ability": ability,
        "reward_model": {"style": "rule", "ground_truth": str(answer)},
        "extra_info": {
            "split": split,
            "index": int(index),
            "source_id": str(source_id),
            "category": category,
            "language": language,
        },
    }


def _format_options(options: list[str], language: str = "en") -> str:
    separator = "、" if language == "zh" else ". "
    return "\n".join(f"{string.ascii_uppercase[index]}{separator}{option}" for index, option in enumerate(options))


def build_mmlu_pro() -> pd.DataFrame:
    dataset = datasets.load_dataset(MMLU_DATASET, revision=MMLU_REVISION, split="test")
    records = []
    for index, example in enumerate(dataset):
        category = example["category"].strip().lower()
        if category not in MMLU_STEM_CATEGORIES:
            continue
        content = (
            f"{example['question'].strip()}\n\nOptions:\n{_format_options(example['options'])}\n\n"
            f"{EN_ANSWER_INSTRUCTION}"
        )
        records.append(
            _record(
                data_source=MMLU_SOURCE,
                content=content,
                ability="stem_reasoning",
                answer=example["answer"],
                split="test",
                index=index,
                source_id=example["question_id"],
                category=category,
                language="en",
            )
        )
    return pd.DataFrame(records, columns=SCHEMA_COLUMNS)


def build_ceval() -> pd.DataFrame:
    records = []
    index = 0
    repo_dir = snapshot_download(
        repo_id=CEVAL_DATASET,
        repo_type="dataset",
        revision=CEVAL_REVISION,
        allow_patterns=[f"{subject}/test-*.parquet" for subject in CEVAL_STEM_SUBJECTS],
        max_workers=8,
    )
    for subject in CEVAL_STEM_SUBJECTS:
        files = sorted(Path(repo_dir, subject).glob("test-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No C-Eval test parquet found for {subject}")
        dataset = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
        for example in dataset.to_dict("records"):
            options = [example[label] for label in "ABCD"]
            content = f"{example['question'].strip()}\n\n选项：\n{_format_options(options, 'zh')}\n\n{ZH_ANSWER_INSTRUCTION}"
            records.append(
                _record(
                    data_source=CEVAL_SOURCE,
                    content=content,
                    ability="stem_reasoning",
                    answer=example["answer"],
                    split="test",
                    index=index,
                    source_id=f"{subject}:{example['id']}",
                    category=subject,
                    language="zh",
                )
            )
            index += 1
    return pd.DataFrame(records, columns=SCHEMA_COLUMNS)


def build_autologi(path: str) -> pd.DataFrame:
    records = []
    with open(os.path.expanduser(path), encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            example = json.loads(line)
            verifier = json.dumps(
                {
                    "inputs_check_code": example["code"]["Inputs_Check_code"],
                    "constraint_list_code": example["code"]["Constraint_List_code"],
                },
                ensure_ascii=True,
            )
            records.append(
                _record(
                    data_source=AUTOLOGI_SOURCE,
                    content=example["prompt"],
                    ability="logic_reasoning",
                    answer=verifier,
                    split="test",
                    index=index,
                    source_id=example["idx"],
                    category=str(example.get("group_id", "autologi")),
                    language="en",
                )
            )
    return pd.DataFrame(records, columns=SCHEMA_COLUMNS)


def build_gpqa(path: str, seed: int) -> pd.DataFrame:
    source = pd.read_csv(os.path.expanduser(path))
    records = []
    rng = random.Random(seed)
    for index, example in source.iterrows():
        choices = [
            (example["Correct Answer"], True),
            (example["Incorrect Answer 1"], False),
            (example["Incorrect Answer 2"], False),
            (example["Incorrect Answer 3"], False),
        ]
        rng.shuffle(choices)
        answer_index = next(i for i, (_, correct) in enumerate(choices) if correct)
        options = [str(choice) for choice, _ in choices]
        content = f"{example['Question'].strip()}\n\nOptions:\n{_format_options(options)}\n\n{EN_ANSWER_INSTRUCTION}"
        records.append(
            _record(
                data_source=GPQA_SOURCE,
                content=content,
                ability="science_reasoning",
                answer=string.ascii_uppercase[answer_index],
                split="test",
                index=index,
                source_id=index,
                category="gpqa_diamond",
                language="en",
            )
        )
    return pd.DataFrame(records, columns=SCHEMA_COLUMNS)


def _stratified_sample(frame: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    if size <= 0 or len(frame) <= size:
        return frame.copy()
    shuffled_groups = []
    categories = frame["extra_info"].map(lambda info: info["category"])
    for _, group in frame.groupby(categories, sort=True):
        shuffled_groups.append(group.sample(frac=1.0, random_state=seed).to_dict("records"))
    sampled = []
    offset = 0
    while len(sampled) < size:
        made_progress = False
        for group in shuffled_groups:
            if offset < len(group) and len(sampled) < size:
                sampled.append(group[offset])
                made_progress = True
        if not made_progress:
            break
        offset += 1
    return pd.DataFrame(sampled, columns=SCHEMA_COLUMNS)


def _write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, index=False)
    print(f"wrote {path}: {len(frame)} rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="data/reasoning_transfer")
    parser.add_argument("--autologi_file", default=None, help="Official AutoLogi_en.jsonl path.")
    parser.add_argument("--gpqa_csv", default=None, help="Official gated gpqa_diamond.csv path.")
    parser.add_argument("--merge_with", default=None, help="Optional existing verl validation parquet.")
    parser.add_argument("--probe_per_source", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(os.path.expanduser(args.local_save_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "mmlu_pro_stem": build_mmlu_pro(),
        "ceval_stem": build_ceval(),
    }
    if args.autologi_file:
        sources["autologi_en"] = build_autologi(args.autologi_file)
    if args.gpqa_csv:
        sources["gpqa_diamond"] = build_gpqa(args.gpqa_csv, args.seed)

    for name, frame in sources.items():
        _write(frame, output_dir / f"{name}.parquet")

    full = pd.concat(sources.values(), ignore_index=True)
    probe = pd.concat(
        [_stratified_sample(frame, args.probe_per_source, args.seed) for frame in sources.values()],
        ignore_index=True,
    )
    _write(full, output_dir / "transfer_full.parquet")
    _write(probe, output_dir / "transfer_probe.parquet")

    if args.merge_with:
        existing = pd.read_parquet(os.path.expanduser(args.merge_with))
        _write(pd.concat([existing, full], ignore_index=True, sort=False), output_dir / "math_and_transfer_full.parquet")
        _write(pd.concat([existing, probe], ignore_index=True, sort=False), output_dir / "math_and_transfer_probe.parquet")

    manifest = {
        "seed": args.seed,
        "probe_per_source": args.probe_per_source,
        "sources": {name: len(frame) for name, frame in sources.items()},
        "full_rows": len(full),
        "probe_rows": len(probe),
        "gpqa_included": bool(args.gpqa_csv),
        "autologi_included": bool(args.autologi_file),
        "mmlu_revision": MMLU_REVISION,
        "ceval_revision": CEVAL_REVISION,
        "autologi_revision": "d67fda13ab3d950403b32ca2242aee426e54472a",
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    main()
