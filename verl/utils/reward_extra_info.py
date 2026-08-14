# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def collate_reward_extra_infos(
    reward_extra_infos: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Collate reward metadata dictionaries with potentially different schemas.

    Reward functions are allowed to return task-specific metadata. For example, a
    math scorer may return ``score``, ``acc`` and ``pred``, while another scorer
    returns only ``acc``. Keep the union of those fields and use ``None`` for a
    field that is not defined for a sample so batch alignment is preserved.
    """
    reward_extra_keys = list(
        dict.fromkeys(key for reward_extra_info in reward_extra_infos for key in reward_extra_info)
    )
    non_tensor_batch = {}
    for key in reward_extra_keys:
        values = [reward_extra_info.get(key) for reward_extra_info in reward_extra_infos]
        if any(value is None for value in values):
            # np.array may infer an unexpected string/float dtype for heterogeneous
            # values. Object dtype preserves None as the missing-value sentinel.
            value_array = np.empty(len(values), dtype=object)
            value_array[:] = values
        else:
            value_array = np.asarray(values)
        non_tensor_batch[key] = value_array

    return non_tensor_batch, reward_extra_keys


def make_missing_reward_extra_info_array(size: int) -> np.ndarray:
    """Create an aligned array for a reward metadata field absent from a chunk."""
    values = np.empty(size, dtype=object)
    values.fill(None)
    return values


def extend_aligned_reward_extra_infos(
    destination: dict[str, list],
    batch_values: Mapping[str, Any],
    *,
    batch_size: int,
    prior_size: int,
) -> None:
    """Append task-specific metadata while preserving one value per sample."""
    normalized_batch = {}
    for key, values in batch_values.items():
        if isinstance(values, np.ndarray):
            values = values.tolist()
        elif not isinstance(values, list):
            values = [values]
        if len(values) != batch_size:
            raise ValueError(f"reward metadata {key!r} has {len(values)} values, expected {batch_size}")
        normalized_batch[key] = values

    for key in set(destination) - set(normalized_batch) - {"reward"}:
        destination[key].extend([None] * batch_size)

    for key, values in normalized_batch.items():
        if key not in destination:
            destination[key] = [None] * prior_size
        destination[key].extend(values)
