from __future__ import annotations

import base64
import json
import pickle
import zlib

from examples.data_preprocess.deepcoder_lcb import build_prompt, compress_tests, make_preprocess_fn
from verl.utils.reward_score import default_compute_score


def _tests():
    return [
        {
            "input": "2 3\n",
            "output": "5\n",
            "testtype": "stdin_stdout",
            "metadata": {"func_name": None},
        }
    ]


def test_deepcoder_preprocessor_emits_user_only_prompt_and_decodable_tests():
    preprocess = make_preprocess_fn("train")
    row = preprocess(
        {"problem": "Add two integers.", "tests": _tests(), "starter_code": "", "metadata": {}},
        7,
    )

    assert row["prompt"] == [{"role": "user", "content": build_prompt("Add two integers.")}]
    assert row["data_source"] == "livecodebench"
    compressed = row["reward_model"]["ground_truth"]
    decoded = json.loads(pickle.loads(zlib.decompress(base64.b64decode(compressed))))
    assert decoded == _tests()


def test_livecodebench_reward_is_binary_pass_all_tests():
    ground_truth = compress_tests(_tests())
    correct = "```python\na, b = map(int, input().split())\nprint(a + b)\n```"
    incorrect = "```python\nprint(0)\n```"

    assert default_compute_score("livecodebench", correct, ground_truth) == 1.0
    assert default_compute_score("livecodebench", incorrect, ground_truth) == 0.0


def test_livecodebench_reward_is_zero_when_code_block_is_missing_or_empty():
    ground_truth = compress_tests(_tests())

    assert default_compute_score("livecodebench", "I could not solve this problem.", ground_truth) == 0.0
    assert default_compute_score("livecodebench", "```python\n```", ground_truth) == 0.0


def test_livecodebench_reward_is_zero_when_evaluator_child_exits_without_result():
    ground_truth = compress_tests(_tests())
    abrupt_exit = "```python\nimport os\nos._exit(7)\n```"

    assert default_compute_score("livecodebench", abrupt_exit, ground_truth) == 0.0


def test_livecodebench_reward_is_zero_when_evaluator_exceeds_memory_limit(monkeypatch):
    ground_truth = compress_tests(_tests())
    monkeypatch.setenv("VERL_CODE_EVAL_MAX_MEMORY_GB", "0.25")
    bounded_allocation = "```python\n_buffer = bytearray(32 * 1024 * 1024)\nprint(5)\n```"
    memory_hog = "```python\n_buffer = bytearray(320 * 1024 * 1024)\nprint(5)\n```"

    assert default_compute_score("livecodebench", bounded_allocation, ground_truth) == 1.0
    assert default_compute_score("livecodebench", memory_hog, ground_truth) == 0.0
