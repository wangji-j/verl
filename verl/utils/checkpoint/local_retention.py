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

import os
import re
import shutil
import time
from pathlib import Path


_GLOBAL_STEP_PATTERN = re.compile(r"^global_step_(\d+)$")


def cleanup_local_global_step_checkpoints(
    checkpoint_root: str,
    current_step: int,
    max_ckpt_to_keep: int | None,
    *,
    delete_attempts: int = 5,
    retry_delay_seconds: float = 1.0,
) -> list[str]:
    """Delete stale local global-step directories from a single controller process."""
    if max_ckpt_to_keep is None or max_ckpt_to_keep <= 0:
        return []
    if delete_attempts < 1:
        raise ValueError("delete_attempts must be at least 1")

    root = Path(checkpoint_root)
    current_path = root / f"global_step_{current_step}"
    if not current_path.is_dir():
        raise RuntimeError(f"Current checkpoint is incomplete or missing: {current_path}")

    checkpoints: list[tuple[int, Path]] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        match = _GLOBAL_STEP_PATTERN.fullmatch(entry.name)
        if match is not None:
            checkpoints.append((int(match.group(1)), entry))

    future_steps = [step for step, _ in checkpoints if step > current_step]
    if future_steps:
        raise RuntimeError(
            f"Refusing checkpoint cleanup because steps newer than current step {current_step} exist: "
            f"{sorted(future_steps)}"
        )

    checkpoints.sort(key=lambda item: item[0])
    stale_paths = [path for _, path in checkpoints[:-max_ckpt_to_keep]]
    deleted_paths: list[str] = []

    for stale_path in stale_paths:
        last_error: OSError | None = None
        for attempt in range(delete_attempts):
            try:
                shutil.rmtree(stale_path)
            except FileNotFoundError:
                pass
            except OSError as error:
                last_error = error

            if not os.path.lexists(stale_path):
                deleted_paths.append(str(stale_path))
                break

            if attempt + 1 < delete_attempts:
                time.sleep(retry_delay_seconds * (attempt + 1))
        else:
            detail = f": {last_error}" if last_error is not None else ""
            raise RuntimeError(
                f"Failed to delete stale checkpoint after {delete_attempts} attempts: {stale_path}{detail}"
            )

    return deleted_paths
