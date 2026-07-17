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

import megatron.core
import torch
from megatron.core import dist_checkpointing, mpu
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.strategies.resharding import (
    TensorReformulationMetadata,
    is_nd_flattened_tensor,
    nd_flattened_tensor_reformulated_global_shape,
)
from megatron.core.dist_checkpointing.serialization import (
    get_default_load_sharded_strategy,
    get_default_save_sharded_strategy,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from packaging import version


def _patch_missing_mcore_data_reformulation_metadata():
    """Allow loading torch_dist checkpoints saved without mcore_data metadata.

    Megatron-Core 0.14 load code expects PyTorch DCP metadata to contain
    `mcore_data` for N-D flattened tensors. Some torch_dist checkpoints saved
    by the same stack only contain the native PyTorch `storage_data` metadata.
    When training parallelism is unchanged, the checkpoint formulation matches
    the application formulation, so no N-D reformulation is needed. In that
    case synthesize identity reformulation metadata instead of failing before
    the actual load starts.
    """
    import megatron.core.dist_checkpointing.strategies.torch as torch_strategy

    if getattr(torch_strategy.get_reformulation_metadata, "_verl_missing_mcore_patch", False):
        return

    original_get_reformulation_metadata = torch_strategy.get_reformulation_metadata

    def get_reformulation_metadata_compat(sharded_state_dict, checkpoint_dir):
        fs_reader = torch_strategy._get_filesystem_reader(checkpoint_dir)
        ckpt_metadata = fs_reader.read_metadata()
        if hasattr(ckpt_metadata, "mcore_data"):
            return original_get_reformulation_metadata(sharded_state_dict, checkpoint_dir)

        reformulation_metadata = {}
        for sh_ten in nested_values(sharded_state_dict):
            if not is_nd_flattened_tensor(sh_ten):
                continue
            reformulation_metadata[sh_ten.key] = TensorReformulationMetadata(
                sh_ten.global_shape,
                nd_flattened_tensor_reformulated_global_shape(sh_ten),
            )
        if reformulation_metadata:
            print(
                "WARNING: checkpoint metadata has no mcore_data; "
                "using identity N-D flattened tensor reformulation metadata. "
                "This is only valid when Megatron parallelism matches the saved checkpoint."
            )
        return reformulation_metadata

    get_reformulation_metadata_compat._verl_missing_mcore_patch = True
    torch_strategy.get_reformulation_metadata = get_reformulation_metadata_compat


def save_dist_checkpointing(
    sharded_state_dict,
    ckpt_path,
    async_save=False,
    content_metadata=None,
):
    validate_sharding_integrity = True
    # Get checkpointing strategies
    save_strategy = get_default_save_sharded_strategy("torch_dist")
    save_strategy = FullyParallelSaveStrategyWrapper(
        save_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
    )

    # https://github.com/NVIDIA/Megatron-LM/blob/core_v0.14.0/megatron/core/optimizer/distrib_optimizer.py#L1109-L1123
    mcore_ge_014 = version.parse(megatron.core.__version__) >= version.parse("0.14.0")
    # Save model sharded state dicts
    save_kwargs = dict(
        sharded_strategy=save_strategy,
        async_sharded_save=async_save,
        validate_access_integrity=validate_sharding_integrity,
    )
    if content_metadata is not None:
        if mcore_ge_014:
            save_kwargs["content_metadata"] = content_metadata
    return dist_checkpointing.save(sharded_state_dict, ckpt_path, **save_kwargs)


def load_dist_checkpointing(sharded_state_dict, ckpt_dir):
    _patch_missing_mcore_data_reformulation_metadata()

    # Get checkpointing strategies
    load_strategy = get_default_load_sharded_strategy(ckpt_dir)
    load_strategy = FullyParallelLoadStrategyWrapper(
        load_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
    )

    # Fix torch.load weights only error
    try:
        import transformer_engine as te

        torch.serialization.add_safe_globals([torch.optim.AdamW])
        torch.serialization.add_safe_globals([te.pytorch.optimizers.fused_adam.FusedAdam])
    except Exception:
        pass

    # Load model sharded state dicts
    state_dict = dist_checkpointing.load(sharded_state_dict, ckpt_dir, sharded_strategy=load_strategy)

    return state_dict
