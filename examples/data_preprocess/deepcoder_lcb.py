"""Prepare DeepCoder training data and Qwen3's LiveCodeBench v5 slice.

The schema follows the official OAPL DeepCoder preprocessing recipe: no system
message is emitted, coding and formatting instructions live in the user prompt,
and unit tests are compressed into ``reward_model.ground_truth``.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import pickle
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset


TRAIN_DATASET_NAME = "agentica-org/DeepCoder-Preview-Dataset"
TRAIN_CONFIGS = ("primeintellect", "taco", "lcbv5")
VALIDATION_DATASET_NAME = "livecodebench/code_generation_lite"
VALIDATION_CONFIG = "v5"
VALIDATION_REVISION = "c5ddb3cdc5b0cbedb4554edc7b04f34d284e6413"
VALIDATION_DATA_FILES = tuple(
    "https://huggingface.co/datasets/"
    f"{VALIDATION_DATASET_NAME}/resolve/{VALIDATION_REVISION}/"
    f"{VALIDATION_CONFIG}/test-{shard:05d}-of-00002.parquet"
    for shard in range(2)
)
VALIDATION_WINDOW = "2024.10-2025.02"
EXPECTED_VALIDATION_SIZE = 167
PARQUET_WRITE_BATCH_SIZE = 16

LCB_INSTRUCTION = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests."
)
LCB_WITH_STARTER = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)
LCB_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within "
    "delimiters as follows. Ensure that when the python program runs, it reads "
    "the inputs, runs the algorithm and writes output to STDOUT."
)


def build_prompt(problem: str, starter_code: str = "") -> str:
    prompt = f"{LCB_INSTRUCTION}\n\n{problem}"
    if starter_code:
        prompt += f"### Format: {LCB_WITH_STARTER}\n```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {LCB_WITHOUT_STARTER}\n```python\n# YOUR CODE HERE\n```\n\n"
    return prompt + "### Answer: (use the provided format with backticks)\n\n"


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        return json.loads(value)
    return dict(value)


def _decode_lcb_test_blob(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    if isinstance(value, list):
        return copy.deepcopy(value)
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        decoded = pickle.loads(zlib.decompress(base64.b64decode(value.encode("utf-8"))))
        decoded = json.loads(decoded) if isinstance(decoded, str) else decoded
    if not isinstance(decoded, list):
        raise ValueError(f"expected a list of LiveCodeBench tests, got {type(decoded).__name__}")
    return decoded


def normalize_tests(tests_raw: Any, metadata_raw: Any) -> list[dict[str, Any]]:
    tests = json.loads(tests_raw) if isinstance(tests_raw, str) else copy.deepcopy(tests_raw)
    metadata = _as_dict(metadata_raw)

    if isinstance(tests, dict) and "inputs" in tests and "outputs" in tests:
        normalized = []
        for input_value, output_value in zip(tests["inputs"], tests["outputs"], strict=False):
            normalized.append({"input": input_value, "output": output_value, "testtype": "stdin_stdout"})
        tests = normalized

    if not isinstance(tests, list):
        tests = [tests] if tests else []

    normalized_tests = []
    for test_raw in tests:
        test = dict(test_raw)
        func_name = metadata.get("func_name") if test.get("testtype") == "functional" else None
        test["metadata"] = {"func_name": str(func_name) if func_name is not None else None}
        normalized_tests.append(test)
    if not normalized_tests:
        raise ValueError("coding example has no unit tests")
    return normalized_tests


def normalize_lcb_tests(example: dict[str, Any]) -> list[dict[str, Any]]:
    public_tests = _decode_lcb_test_blob(example.get("public_test_cases"))
    private_tests = _decode_lcb_test_blob(example.get("private_test_cases"))
    return normalize_tests(public_tests + private_tests, example.get("metadata"))


def compress_tests(tests: list[dict[str, Any]]) -> str:
    payload = pickle.dumps(json.dumps(tests))
    return base64.b64encode(zlib.compress(payload)).decode("utf-8")


def make_preprocess_fn(split: str, index_offset: int = 0):
    def preprocess(example: dict[str, Any], index: int) -> dict[str, Any]:
        global_index = index_offset + index
        starter_code = example.get("starter_code") or ""
        metadata = _as_dict(example.get("metadata"))
        if "private_test_cases" in example:
            tests = normalize_lcb_tests(example)
            problem = example["question_content"]
        else:
            tests = normalize_tests(example.get("tests"), metadata)
            problem = example["problem"]
        contest_date = example.get("contest_date")
        return {
            "data_source": "livecodebench",
            "prompt": [{"role": "user", "content": build_prompt(problem, starter_code)}],
            "ability": "code_generation",
            "reward_model": {"style": "rule", "ground_truth": compress_tests(tests)},
            "extra_info": {
                "split": split,
                "uid": f"deepcoder_{split}_{global_index}",
                "index": global_index,
                "starter_code": starter_code,
                "metadata": json.dumps(metadata, sort_keys=True),
                "contest_date": contest_date,
            },
        }

    return preprocess


def _load_source() -> tuple[list[Dataset], Dataset, dict[str, str]]:
    train_parts = [load_dataset(TRAIN_DATASET_NAME, name=name, split="train") for name in TRAIN_CONFIGS]
    validation = load_dataset("parquet", data_files={"test": list(VALIDATION_DATA_FILES)}, split="test")
    fingerprints = {name: dataset._fingerprint for name, dataset in zip(TRAIN_CONFIGS, train_parts, strict=True)}
    fingerprints[f"{VALIDATION_DATASET_NAME}:{VALIDATION_CONFIG}:test"] = validation._fingerprint
    return train_parts, validation, fingerprints


def _validate_schema(dataset: Dataset, split: str) -> None:
    required = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(f"{split} dataset is missing columns: {sorted(missing)}")
    for index in range(min(32, len(dataset))):
        row = dataset[index]
        roles = [message["role"] for message in row["prompt"]]
        if roles != ["user"]:
            raise ValueError(f"{split}[{index}] must contain exactly one user message, got {roles}")
        if row["data_source"] != "livecodebench":
            raise ValueError(f"{split}[{index}] has unexpected data_source={row['data_source']!r}")


def prepare(args: argparse.Namespace) -> None:
    output_dir = Path(args.local_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_parts, validation, fingerprints = _load_source()
    if args.validation_size is not None:
        validation = validation.select(range(min(args.validation_size, len(validation))))

    processed_train_parts = []
    train_index_offset = 0
    for config_name, train_part in zip(TRAIN_CONFIGS, train_parts, strict=True):
        processed = train_part.map(
            make_preprocess_fn("train", index_offset=train_index_offset),
            with_indices=True,
            remove_columns=train_part.column_names,
            num_proc=args.num_proc,
            writer_batch_size=10,
            desc=f"Preparing DeepCoder train ({config_name})",
        )
        train_index_offset += len(train_part)
        processed_train_parts.append(processed)
    train = concatenate_datasets(processed_train_parts)
    if args.train_size is not None:
        train = train.select(range(min(args.train_size, len(train))))
    validation = validation.map(
        make_preprocess_fn("validation"),
        with_indices=True,
        remove_columns=validation.column_names,
        num_proc=args.num_proc,
        writer_batch_size=10,
        desc="Preparing LiveCodeBench v5 validation",
    )

    _validate_schema(train, "train")
    _validate_schema(validation, "validation")
    if args.validation_size is None and len(validation) != args.expected_validation_size:
        raise ValueError(
            f"expected {args.expected_validation_size} LiveCodeBench v5 validation rows, got {len(validation)}"
        )

    train_path = output_dir / "train.parquet"
    validation_path = output_dir / "test.parquet"
    # Some LCB rows contain tens of MiB of compressed tests. Small row groups
    # keep nested Arrow batches below the 2 GiB conversion limit.
    train.to_parquet(train_path, batch_size=PARQUET_WRITE_BATCH_SIZE)
    validation.to_parquet(validation_path, batch_size=PARQUET_WRITE_BATCH_SIZE)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_source": TRAIN_DATASET_NAME,
        "train_configs": list(TRAIN_CONFIGS),
        "validation_source": VALIDATION_DATASET_NAME,
        "validation_config": VALIDATION_CONFIG,
        "validation_revision": VALIDATION_REVISION,
        "validation_window": VALIDATION_WINDOW,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "source_fingerprints": fingerprints,
        "prompt_roles": ["user"],
        "reward": "binary pass-all-tests",
        "files": {"train": str(train_path), "validation": str(validation_path)},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--num-proc", type=int, default=16)
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--validation-size", type=int)
    parser.add_argument("--expected-validation-size", type=int, default=EXPECTED_VALIDATION_SIZE)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
