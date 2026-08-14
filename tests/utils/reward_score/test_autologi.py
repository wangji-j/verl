import json

from verl.utils.reward_score.autologi import compute_score


def _verifier() -> str:
    return json.dumps(
        {
            "inputs_check_code": "def inputs_check(inputs):\n    return isinstance(inputs, list) and len(inputs) == 2",
            "constraint_list_code": (
                "def constraint_1(inputs):\n    return inputs[0] < inputs[1]\n"
                "constraint_list = [constraint_1]"
            ),
        }
    )


def test_accepts_valid_structured_answer() -> None:
    result = compute_score("Reasoning...\n```json\n[1, 2]\n```", _verifier())
    assert result["score"] == 1.0
    assert result["format_valid"] == 1.0


def test_rejects_constraint_violation_and_bad_format() -> None:
    assert compute_score("Final answer: [2, 1]", _verifier())["score"] == 0.0
    assert compute_score("No structured result", _verifier())["format_valid"] == 0.0
