# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Program-based verifier for the AutoLogi benchmark."""

import ast
import json
import os
import re
import subprocess
import sys
import tempfile


_VERIFY_FUNCTION = """\
def verify_function(inputs, inputs_check, constraint_list):
    if not inputs_check(inputs):
        return False
    return all(constraint(inputs) for constraint in constraint_list)
"""


def _extract_last_balanced(text: str, opening: str, closing: str) -> str | None:
    depth = 0
    start = None
    last = None
    for index, char in enumerate(text):
        if char == opening:
            if depth == 0:
                start = index
            depth += 1
        elif char == closing and depth:
            depth -= 1
            if depth == 0 and start is not None:
                last = text[start : index + 1]
    return last


def _extract_candidate(solution_str: str) -> str | None:
    code_blocks = re.findall(r"```(?:\s*[\w+-]+)?\s*\n(.*?)```", solution_str, flags=re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()
    return _extract_last_balanced(solution_str, "{", "}") or _extract_last_balanced(solution_str, "[", "]")


def _parse_candidate(solution_str: str):
    candidate = _extract_candidate(solution_str)
    if candidate is None:
        raise ValueError("no structured answer found")
    candidate = re.sub(r"//.*?(?:\n|$)", "", candidate).strip()
    if "=" in candidate and not candidate.lstrip().startswith(("{", "[")):
        candidate = candidate.split("=", 1)[1].strip()
    normalized = re.sub(r"\btrue\b", "True", candidate, flags=re.IGNORECASE)
    normalized = re.sub(r"\bfalse\b", "False", normalized, flags=re.IGNORECASE)
    try:
        return ast.literal_eval(normalized)
    except (SyntaxError, ValueError):
        return json.loads(candidate)


def _verify_in_subprocess(params, verifier: dict, timeout: float) -> tuple[bool, bool]:
    program = "\n".join(
        (
            verifier["inputs_check_code"],
            verifier["constraint_list_code"],
            _VERIFY_FUNCTION,
            f"inputs = {params!r}",
            "print(verify_function(inputs, inputs_check, constraint_list))",
        )
    )
    with tempfile.TemporaryDirectory(prefix="verl_autologi_") as temp_dir:
        script_path = os.path.join(temp_dir, "verify.py")
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write(program)
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, True
    return result.returncode == 0 and result.stdout.strip() == "True", False


def compute_score(solution_str: str, ground_truth: str, timeout: float = 10.0) -> dict[str, float]:
    try:
        params = _parse_candidate(solution_str)
    except (ValueError, TypeError, json.JSONDecodeError):
        return {"score": 0.0, "acc": 0.0, "format_valid": 0.0, "verifier_timeout": 0.0}

    try:
        verifier = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
        correct, timed_out = _verify_in_subprocess(params, verifier, timeout)
    except (KeyError, TypeError, json.JSONDecodeError, OSError):
        correct, timed_out = False, False
    return {
        "score": float(correct),
        "acc": float(correct),
        "format_valid": 1.0,
        "verifier_timeout": float(timed_out),
    }
