# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Strict answer extraction for generative multiple-choice evaluation."""

import re


_ANSWER_PATTERNS = (
    re.compile(
        r"(?:final\s+answer|answer|final|最终答案|答案)\s*(?:is|为|是)?\s*[:：]?\s*"
        r"(?:\\boxed\s*\{\s*)?[\(（\[]?\s*([A-J])\s*[\)）\]]?\s*\}?",
        re.IGNORECASE,
    ),
    re.compile(r"\\boxed\s*\{\s*([A-J])\s*\}", re.IGNORECASE),
)


def extract_answer(solution_str: str) -> str | None:
    """Return the last explicitly marked answer choice in a completion."""
    matches: list[tuple[int, str]] = []
    for pattern in _ANSWER_PATTERNS:
        matches.extend((match.start(), match.group(1).upper()) for match in pattern.finditer(solution_str))
    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


def compute_score(solution_str: str, ground_truth: str) -> dict[str, float]:
    prediction = extract_answer(solution_str)
    expected = str(ground_truth).strip().upper()
    valid = prediction is not None
    correct = valid and prediction == expected
    return {
        "score": float(correct),
        "acc": float(correct),
        "format_valid": float(valid),
    }
