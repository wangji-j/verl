"""
This module contains the RewardCode class, which evaluates code datasets answers
and assigns rewards based on their correctness on unit tests.

Adapted from OAPL commit 6153219b7329afa35909e5ce224e401de0c89ae4
under the Apache-2.0 license. Its evaluator is derived from LiveCodeBench.
"""

import ast
import base64
import json
import math
import multiprocessing
import os
import pickle
import re
import zlib
from typing import Any

from verl.utils.reward_score.rllm_code.livecodebench import run_test as lcb_run_test


_CODE_EVAL_MAX_MEMORY_GIB_ENV = "VERL_CODE_EVAL_MAX_MEMORY_GB"
_DEFAULT_CODE_EVAL_MAX_MEMORY_GIB = 16.0


def _configured_code_eval_memory_bytes() -> int | None:
    """Return the additional address-space budget for one evaluator child."""
    raw_value = os.getenv(_CODE_EVAL_MAX_MEMORY_GIB_ENV, str(_DEFAULT_CODE_EVAL_MAX_MEMORY_GIB))
    try:
        limit_gib = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{_CODE_EVAL_MAX_MEMORY_GIB_ENV} must be a non-negative number, got {raw_value!r}") from exc

    if not math.isfinite(limit_gib) or limit_gib < 0:
        raise ValueError(f"{_CODE_EVAL_MAX_MEMORY_GIB_ENV} must be a finite non-negative number, got {raw_value!r}")
    if limit_gib == 0:
        return None
    return int(limit_gib * 1024**3)


def _current_virtual_memory_bytes() -> int:
    """Read this process's current virtual size before tightening RLIMIT_AS."""
    try:
        with open("/proc/self/statm", encoding="ascii") as statm:
            pages = int(statm.read().split()[0])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError("Unable to determine evaluator virtual memory before applying its hard limit") from exc
        try:
            return int(psutil.Process().memory_info().vms)
        except (OSError, psutil.Error) as exc:
            raise RuntimeError("Unable to determine evaluator virtual memory before applying its hard limit") from exc


def _apply_code_eval_memory_limit(additional_bytes: int | None) -> int | None:
    """Irreversibly cap new address-space growth in the forked evaluator child."""
    if additional_bytes is None:
        return None

    try:
        import resource
    except ImportError as exc:
        raise RuntimeError("Code-evaluation memory limits require resource.RLIMIT_AS") from exc

    if not hasattr(resource, "RLIMIT_AS"):
        raise RuntimeError("Code-evaluation memory limits require resource.RLIMIT_AS")

    current_vms = _current_virtual_memory_bytes()
    target_limit = current_vms + additional_bytes
    _, existing_hard_limit = resource.getrlimit(resource.RLIMIT_AS)
    if existing_hard_limit != resource.RLIM_INFINITY:
        target_limit = min(target_limit, existing_hard_limit)

    # Lower both limits so model-generated code cannot raise the cap again.
    resource.setrlimit(resource.RLIMIT_AS, (target_limit, target_limit))
    return target_limit


def extract_code_from_model(model_response: str) -> str:
    """
    Extracts the code from a Markdown-style code block in an LLM output.

    Parameters:
        model_response (str): The text output from the LLM.

    Returns:
        str: The extracted code, or an empty string if no code block is found.
    """
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return ""
    return code_blocks[-1].strip()


def _invalid_generation_details(test_cases: list[dict[str, Any]], error: str) -> dict[str, Any]:
    return {
        "all_passed": False,
        "test_results": [
            {
                "input": test_case.get("input"),
                "expected": test_case.get("output"),
                "passed": False,
                "error": error,
            }
            for test_case in test_cases
        ],
        "total_tests": len(test_cases),
        "passed_tests": 0,
    }


def postprocess_lcb_sample(sample):
    sample_inputs = [sample["input"] for sample in sample]
    sample_outputs = [sample["output"] for sample in sample]
    sample_dict = {
        "inputs": sample_inputs,
        "outputs": sample_outputs,
    }

    if sample[0].get("testtype") == "functional":
        metadata = sample[0].get("metadata", {})
        fn_name = metadata.get("func_name", None)
        assert fn_name is not None, f"Function name is not found, check if your LCB data is preprocessed correctly: {metadata}\nSample: {sample}"
        # Fill in the blank
        sample_dict["fn_name"] = fn_name

    sample = {
        "input_output": json.dumps(sample_dict),
    }
    return sample


def _temp_run(sample, generation, debug, result_pipe, timeout, max_memory_bytes):
    try:
        try:
            _apply_code_eval_memory_limit(max_memory_bytes)
            result = lcb_run_test(sample, test=generation, debug=debug, timeout=timeout)
        except BaseException as exc:
            test_count = len(json.loads(sample["input_output"])["inputs"])
            result = (
                [False] * test_count,
                {
                    "error": "evaluator child exception",
                    "error_message": f"{type(exc).__name__}: {exc}",
                },
            )
        result_pipe.send(result)
    except (BrokenPipeError, EOFError, OSError):
        # The parent may have timed out and closed its end of the pipe.
        pass
    finally:
        result_pipe.close()


def lcb_check_correctness_v2(sample, generation, timeout=6, debug=False, continuous=False):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    assert len(sample) >= 1, "Sample must contain at least one test case"
    sample = postprocess_lcb_sample(sample)
    max_memory_bytes = _configured_code_eval_memory_bytes()

    context = multiprocessing.get_context("fork")
    result_reader, result_writer = context.Pipe(duplex=False)
    p = context.Process(
        target=_temp_run,
        args=(sample, generation, debug, result_writer, timeout, max_memory_bytes),
    )
    p.start()
    result_writer.close()
    p.join(timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5)

    detailed_results = {"all_passed": False, "test_results": [], "total_tests": 0, "passed_tests": 0}

    timed_out = p.is_alive()
    if timed_out:
        p.kill()
        p.join()
    in_outs = json.loads(sample["input_output"])
    if not result_reader.poll():
        detailed_results["total_tests"] = len(in_outs["inputs"])
        error = "global timeout" if timed_out else f"evaluator exited without result (exitcode={p.exitcode})"
        detailed_results["test_results"] = [
            {"input": inp, "expected": out, "passed": False, "error": error}
            for inp, out in zip(in_outs["inputs"], in_outs["outputs"], strict=False)
        ]
        if debug:
            print(error)
        result_reader.close()
        return False, detailed_results

    try:
        result, metadata = result_reader.recv()
    except (EOFError, OSError) as exc:
        result_reader.close()
        error = f"evaluator pipe closed without result (exitcode={p.exitcode}, {type(exc).__name__})"
        return False, _invalid_generation_details(
            [
                {"input": inp, "output": out}
                for inp, out in zip(in_outs["inputs"], in_outs["outputs"], strict=False)
            ],
            error,
        )
    result_reader.close()

    # Create detailed test results
    detailed_results["total_tests"] = len(result)
    detailed_results["test_results"] = [
        {
            "input": inp,
            "expected": out,
            "passed": res == True,
            "error": metadata.get("error", None),
            "error_message": metadata.get("error_message", None),
            "output": metadata.get("output", None),
        }
        for inp, out, res in zip(in_outs["inputs"], in_outs["outputs"], result, strict=False)
    ]
    detailed_results["passed_tests"] = sum(1 for item in result if item == True)
    detailed_results["all_passed"] = all(item == True for item in result)
    if not continuous:
        return all(item == True for item in result), detailed_results
    else:
        # Continuous: return True if at least one test passed
        pass_frac = detailed_results["passed_tests"] / detailed_results["total_tests"]
        return pass_frac, detailed_results


def taco_to_lcb_format(tests):
    """
    Given a dictionary with keys "inputs" and "outputs", returns a list of test cases.
    Each test case is a dictionary with keys "input" and "output". If the lists are unequal,
    missing entries are filled by reusing the first element of the shorter list.

    Args:
        data (dict): A dictionary with keys "inputs" and "outputs", each mapped to a list of strings.

    Returns:
        list of dict: A list where each element is a dict with keys "input" and "output".
    """
    inputs = tests.get("inputs", [])
    outputs = tests.get("outputs", [])

    # Determine the number of test cases to create.
    n = max(len(inputs), len(outputs))

    test_cases = []
    for i in range(n):
        # Use the first element as a fallback if the list is shorter than n.
        inp = inputs[i] if i < len(inputs) else (inputs[0] if inputs else "")
        out = outputs[i] if i < len(outputs) else (outputs[0] if outputs else "")
        out = out[0] if isinstance(out, list) else out
        test_case: dict[str, Any] = {"input": inp, "output": out, "metadata": {}}
        if "fn_name" in tests:
            test_case["testtype"] = "functional"
            test_case["metadata"]["func_name"] = tests["fn_name"]
        test_cases.append(test_case)

    return test_cases


def compute_score_lcb(completion: str, test_cases: dict | str, continuous: bool = False):
    """Evaluate code solutions against ground truth answers

        This function creates a reward function to evaluate code solutions by pass the test_case from groun_truth. It can optionally use a language model
        for more sophisticated answer validation.

        Args:
            data_source: The source/dataset the problem comes from
            llm_solution: The solution string provided by the language model to evaluate
            ground_truth: some tests for this llm_solution
            enable_llm: Whether to enable language model validation for complex cases (default: False)

        Returns:
            tuple: (bool, dict) where:
                - bool: True if the solution passes all the test_case, False otherwise
                - dict: Detailed test results with test cases and pass/fail status

        Example:
                model_response = '''
    import sys
    from itertools import permutations
    def main():
        n,m=map(int, input().split())
        a=sum(list(map(int, input().split())))
        if a+(n-1)*10<=m:
            print(5)
        else:
            print(5)
    if __name__ == "__main__":
        main()
    '''

        print(f"test the code_forces")
        # tests = [ { "input": "3 30\n2 2 1", "output": "5" }, { "input": "3 10\n3 2 1", "output": "5" } ]
        metadata = {
             "tests": tests,
        }
        True, {"all_passed": True, "test_results": [...]}
    """
    if not isinstance(test_cases, dict):
        try:
            test_cases = json.loads(test_cases)
        except Exception:
            test_cases = json.loads(pickle.loads(zlib.decompress(base64.b64decode(test_cases.encode("utf-8")))))
    model_code = extract_code_from_model(completion)
    if not model_code:
        return False, _invalid_generation_details(test_cases, "no code block found")
    is_correct, test_details = lcb_check_correctness_v2(test_cases, model_code, continuous=continuous)
    return is_correct, test_details


def compute_score_taco_apps_codecontests(completion: str, test_cases: dict | str, continuous: bool = False):
    if not isinstance(test_cases, dict):
        test_cases = json.loads(test_cases)
    model_code = extract_code_from_model(completion)
    normalized_tests = taco_to_lcb_format(test_cases)
    if not model_code:
        return False, _invalid_generation_details(normalized_tests, "no code block found")
    is_correct, test_details = lcb_check_correctness_v2(normalized_tests, model_code, continuous=continuous)
    return is_correct, test_details
