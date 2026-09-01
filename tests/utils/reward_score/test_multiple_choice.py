from verl.utils.reward_score.multiple_choice import compute_score, extract_answer


def test_extracts_last_explicit_answer() -> None:
    response = "First I considered answer B. After checking, Final Answer: D"
    assert extract_answer(response) == "D"
    assert compute_score(response, "D")["score"] == 1.0


def test_accepts_boxed_and_chinese_answer_markers() -> None:
    assert extract_answer(r"Therefore, \boxed{C}") == "C"
    assert extract_answer("最终答案：A") == "A"


def test_rejects_unmarked_choice() -> None:
    result = compute_score("I compared A and B and selected C", "C")
    assert result == {"score": 0.0, "acc": 0.0, "format_valid": 0.0}
