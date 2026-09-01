# Copyright 2026
# Licensed under the Apache License, Version 2.0.
"""Programmatic scoring for ZebraLogic grid-mode responses."""

import json
from typing import Any


def _json_candidates(text: str):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def _extract_prediction(solution_str: str) -> dict[str, Any] | None:
    candidates = list(_json_candidates(solution_str))
    for candidate in reversed(candidates):
        solution = candidate.get("solution")
        if isinstance(solution, dict):
            return solution
    return None


def _normalize_cell(value: Any) -> str | None:
    if isinstance(value, list):
        value = value[0] if value else None
    if not isinstance(value, str):
        return None
    return value.strip().lower()


def compute_metrics(solution_str: str, ground_truth: str) -> dict[str, float | bool]:
    """Return puzzle and cell accuracy using the official grid comparison semantics."""
    truth = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    prediction = _extract_prediction(solution_str)

    headers = truth["header"]
    rows = truth["rows"]
    total_cells = len(rows) * (len(headers) - 1)
    correct_cells = 0

    if prediction is not None:
        for row_index, row in enumerate(rows, start=1):
            predicted_row = prediction.get(f"House {row_index}", {})
            if not isinstance(predicted_row, dict):
                continue
            for column_index, column in enumerate(headers[1:], start=1):
                predicted = _normalize_cell(predicted_row.get(column))
                expected = _normalize_cell(row[column_index])
                if predicted is not None and predicted == expected:
                    correct_cells += 1

    cell_acc = correct_cells / total_cells if total_cells else 0.0
    puzzle_acc = float(total_cells > 0 and correct_cells == total_cells)
    return {
        "score": puzzle_acc,
        "acc": puzzle_acc,
        "cell_acc": cell_acc,
        "format_valid": prediction is not None,
    }


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> float:
    """Return exact puzzle accuracy for verl's mixed validation reward pipeline."""
    return float(compute_metrics(solution_str, ground_truth)["score"])
