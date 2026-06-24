# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from collections import defaultdict

import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("dapo")
class DAPORewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.reward_debug_dir = os.getenv("VERL_REWARD_DEBUG_DIR", "").strip()
        self.reward_debug_steps = int(os.getenv("VERL_REWARD_DEBUG_STEPS", "0") or 0)
        self.reward_debug_samples = int(os.getenv("VERL_REWARD_DEBUG_SAMPLES", "8") or 0)
        self._reward_debug_call_index = 0

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )
            assert not self.overlong_buffer_cfg.enable or self.overlong_buffer_cfg.len > 0, (
                "overlong_buffer.len must be positive when overlong penalty is enabled,"
                f"but got {self.overlong_buffer_cfg.len}."
                "To disable the overlong penalty, set overlong_buffer.enable = False"
            )

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            self._log_rm_scores_reward_debug(data)
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        reward_debug_rows = []
        global_step = int(data.meta_info.get("global_steps", -1)) if data.meta_info is not None else -1
        debug_step = global_step if global_step >= 0 else self._reward_debug_call_index
        self._reward_debug_call_index += 1

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", {})

            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})

            extra_info["rollout_reward_scores"] = rollout_reward_scores

            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                reward_extra_info["acc"].append(score)

            reward = score

            if self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

            if self._should_log_reward_debug(debug_step, len(reward_debug_rows)):
                if isinstance(result, dict):
                    pred = result.get("pred")
                    acc = result.get("acc")
                    raw_score = result.get("score", score)
                else:
                    pred = None
                    acc = score
                    raw_score = score
                reward_debug_rows.append(
                    {
                        "global_step": global_step,
                        "debug_step": debug_step,
                        "sample_index": i,
                        "data_source": str(data_source),
                        "prompt": prompt_str,
                        "response": response_str,
                        "response_tail_300": response_str[-300:],
                        "ground_truth": ground_truth,
                        "score": float(raw_score),
                        "reward_after_overlong": float(reward),
                        "acc": bool(acc) if isinstance(acc, (bool, np.bool_)) else acc,
                        "pred": pred,
                        "response_length": int(valid_response_length),
                        "has_answer_prefix": "answer:" in response_str.lower()[-300:],
                        "has_boxed": "\\boxed" in response_str[-300:],
                    }
                )

        self._write_reward_debug_rows(debug_step, reward_debug_rows)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    def _should_log_reward_debug(self, global_step: int, current_rows: int) -> bool:
        if not self.reward_debug_dir or self.reward_debug_steps <= 0 or self.reward_debug_samples <= 0:
            return False
        if global_step < 0 or global_step >= self.reward_debug_steps:
            return False
        return current_rows < self.reward_debug_samples

    def _log_rm_scores_reward_debug(self, data: DataProto) -> None:
        if "rm_scores" not in data.batch:
            return
        global_step = int(data.meta_info.get("global_steps", -1)) if data.meta_info is not None else -1
        debug_step = global_step if global_step >= 0 else self._reward_debug_call_index
        self._reward_debug_call_index += 1

        rows = []
        for i in range(len(data)):
            if not self._should_log_reward_debug(debug_step, len(rows)):
                break
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            reward_model = data_item.non_tensor_batch.get("reward_model", {})
            ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "unknown")
            rm_score = float(data_item.batch["rm_scores"].sum().item())

            parser_result = None
            parser_error = None
            if ground_truth is not None:
                try:
                    parser_result = self.compute_score(
                        data_source=data_source,
                        solution_str=response_str,
                        ground_truth=ground_truth,
                        extra_info=data_item.non_tensor_batch.get("extra_info", {}),
                    )
                except Exception as exc:
                    parser_error = repr(exc)

            if isinstance(parser_result, dict):
                pred_debug = parser_result.get("pred")
                parser_score = parser_result.get("score")
                parser_acc = parser_result.get("acc")
            else:
                pred_debug = None
                parser_score = parser_result
                parser_acc = parser_result

            reward_extra = {}
            for key in data.meta_info.get("reward_extra_keys", []) if data.meta_info is not None else []:
                value = data_item.non_tensor_batch.get(key)
                if isinstance(value, np.generic):
                    value = value.item()
                reward_extra[key] = value

            rows.append(
                {
                    "global_step": global_step,
                    "debug_step": debug_step,
                    "sample_index": i,
                    "source": "rm_scores_fast_path",
                    "data_source": str(data_source),
                    "prompt": prompt_str,
                    "response": response_str,
                    "response_tail_300": response_str[-300:],
                    "ground_truth": ground_truth,
                    "score": rm_score,
                    "parser_score_debug": parser_score,
                    "parser_acc_debug": bool(parser_acc) if isinstance(parser_acc, (bool, np.bool_)) else parser_acc,
                    "pred": pred_debug,
                    "parser_error": parser_error,
                    "reward_extra": reward_extra,
                    "response_length": int(valid_response_length),
                    "has_answer_prefix": "answer:" in response_str.lower()[-300:],
                    "has_boxed": "\\boxed" in response_str[-300:],
                }
            )

        self._write_reward_debug_rows(debug_step, rows)

    def _write_reward_debug_rows(self, global_step: int, rows: list[dict]) -> None:
        if not rows:
            return
        os.makedirs(self.reward_debug_dir, exist_ok=True)
        path = os.path.join(self.reward_debug_dir, f"train_reward_debug_step{global_step:04d}_pid{os.getpid()}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
