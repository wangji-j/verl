# Copyright 2026
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

from pathlib import Path

import pytest

from verl.utils.checkpoint.local_retention import cleanup_local_global_step_checkpoints


def _create_checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"global_step_{step}"
    checkpoint.mkdir()
    (checkpoint / "data.pt").write_text(str(step))
    return checkpoint


def test_cleanup_keeps_latest_checkpoint(tmp_path):
    old_checkpoint = _create_checkpoint(tmp_path, 30)
    current_checkpoint = _create_checkpoint(tmp_path, 60)

    deleted = cleanup_local_global_step_checkpoints(str(tmp_path), current_step=60, max_ckpt_to_keep=1)

    assert deleted == [str(old_checkpoint)]
    assert not old_checkpoint.exists()
    assert current_checkpoint.exists()


def test_cleanup_keeps_requested_number_and_ignores_other_directories(tmp_path):
    first_checkpoint = _create_checkpoint(tmp_path, 30)
    second_checkpoint = _create_checkpoint(tmp_path, 60)
    current_checkpoint = _create_checkpoint(tmp_path, 90)
    analysis_dir = tmp_path / "router_analysis_dump"
    analysis_dir.mkdir()

    cleanup_local_global_step_checkpoints(str(tmp_path), current_step=90, max_ckpt_to_keep=2)

    assert not first_checkpoint.exists()
    assert second_checkpoint.exists()
    assert current_checkpoint.exists()
    assert analysis_dir.exists()


def test_cleanup_refuses_to_delete_when_newer_step_exists(tmp_path):
    _create_checkpoint(tmp_path, 60)
    newer_checkpoint = _create_checkpoint(tmp_path, 90)

    with pytest.raises(RuntimeError, match="newer than current step"):
        cleanup_local_global_step_checkpoints(str(tmp_path), current_step=60, max_ckpt_to_keep=1)

    assert newer_checkpoint.exists()


def test_cleanup_requires_current_checkpoint(tmp_path):
    old_checkpoint = _create_checkpoint(tmp_path, 30)

    with pytest.raises(RuntimeError, match="incomplete or missing"):
        cleanup_local_global_step_checkpoints(str(tmp_path), current_step=60, max_ckpt_to_keep=1)

    assert old_checkpoint.exists()


@pytest.mark.parametrize("retention", [None, 0, -1])
def test_cleanup_disabled_does_not_delete(tmp_path, retention):
    old_checkpoint = _create_checkpoint(tmp_path, 30)
    current_checkpoint = _create_checkpoint(tmp_path, 60)

    deleted = cleanup_local_global_step_checkpoints(str(tmp_path), current_step=60, max_ckpt_to_keep=retention)

    assert deleted == []
    assert old_checkpoint.exists()
    assert current_checkpoint.exists()
