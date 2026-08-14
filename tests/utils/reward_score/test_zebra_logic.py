import json

import pytest

from verl.utils.reward_score.zebra_logic import compute_metrics, compute_score


@pytest.fixture
def truth():
    return json.dumps(
        {
            "header": ["House", "Name", "Drink"],
            "rows": [["1", "Alice", "tea"], ["2", "Bob", "water"]],
        }
    )


def test_zebra_logic_full_match(truth):
    response = 'reasoning\n{"solution":{"House 1":{"Name":"alice","Drink":"tea"},"House 2":{"Name":"Bob","Drink":"water"}}}'
    assert compute_metrics(response, truth) == {
        "score": 1.0,
        "acc": 1.0,
        "cell_acc": 1.0,
        "format_valid": True,
    }
    assert compute_score(response, truth) == 1.0


def test_zebra_logic_partial_match_uses_last_json(truth):
    response = '{"solution": null}\n{"solution":{"House 1":{"Name":"Alice","Drink":"tea"}}}'
    result = compute_metrics(response, truth)
    assert result["score"] == 0.0
    assert result["cell_acc"] == 0.5
    assert result["format_valid"] is True


def test_zebra_logic_invalid_format(truth):
    result = compute_metrics("Alice drinks tea", truth)
    assert result["score"] == 0.0
    assert result["cell_acc"] == 0.0
    assert result["format_valid"] is False
