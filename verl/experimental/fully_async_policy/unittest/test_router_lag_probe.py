# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""CPU tests for router_lag_probe.py. Run: python3 -m verl.experimental.fully_async_policy.unittest.test_router_lag_probe"""

import numpy as np
import torch
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.router_lag_probe import (
    bucket_quotas,
    compare_routes,
    crop_routes_to_response,
    length_buckets,
    masked_mean_abs_diff,
    pick_alignment,
    probe_settings,
    select_probe_indices,
    summarize,
    update_lag,
)
from verl.utils.router_mismatch_metrics import compute_router_mismatch_metrics

B, P, R, L, K, E = 12, 7, 40, 4, 2, 8


def _routes(batch, seq, generator, low=0, high=E):
    """Distinct top-K expert ids per (row, position, layer) drawn from [low, high)."""
    scores = torch.rand(batch, seq, L, high - low, generator=generator)
    return (scores.argsort(dim=-1)[..., :K] + low).to(torch.int32)


def _mask(generator):
    lengths = torch.randint(5, R + 1, (B,), generator=generator)
    return (torch.arange(R)[None, :] < lengths[:, None]).to(torch.int64)


def test_identical_routes_have_zero_drift():
    g = torch.Generator().manual_seed(0)
    ref, mask = _routes(B, R, g), _mask(g)
    out = compare_routes(ref, ref.clone(), mask, alignment=0, num_experts=E, smoothing_tau=16.0, chunk=5)
    assert torch.all(out["d_route"] == 0) and torch.all(out["flip_rate"] == 0)
    assert torch.all(out["overlap_mismatch"] == 0)
    assert torch.equal(out["ref_counts"], out["cur_counts"])
    assert summarize(out)["token_layer_flip_rate"] == 0.0
    assert float(out["token_count"]) == float(mask.sum())


def test_one_layer_flipped_gives_exact_rates():
    g = torch.Generator().manual_seed(1)
    ref, mask = _routes(B, R, g), _mask(g)
    ref[:, :, 0, :] = _routes(B, R, g, low=0, high=4)[:, :, 0, :]
    cur = ref.clone()
    cur[:, :, 0, :] = ref[:, :, 0, :] + 4  # layer 0 now uses a disjoint expert set
    out = compare_routes(ref, cur, mask, alignment=0, num_experts=E, smoothing_tau=16.0, chunk=5)
    assert torch.allclose(out["flip_rate"], torch.full((B,), 1.0 / L))
    assert torch.allclose(out["overlap_mismatch"], torch.full((B,), 1.0 / L))
    assert torch.all(out["d_route"] > 0)
    assert abs(summarize(out)["token_layer_flip_rate"] - 1.0 / L) < 1e-9
    assert out["ref_counts"].shape == (B, L, E)


def test_crop_matches_full_rollout_routes():
    g = torch.Generator().manual_seed(2)
    rollout_full, train, mask = _routes(B, P + R, g), _routes(B, R, g), _mask(g)
    cropped = crop_routes_to_response(rollout_full, R)
    assert cropped.dtype == torch.uint8 and cropped.shape == (B, R, L, K)
    for alignment in (-1, 0, 1):
        kwargs = dict(alignments=(alignment,), metric_mode="expert_usage_l1", expert_usage_num_experts=E)
        full = compute_router_mismatch_metrics(rollout_full, train, mask, **kwargs)
        crop = compute_router_mismatch_metrics(cropped, crop_routes_to_response(train, R), mask, **kwargs)
        assert torch.allclose(full.seq_mismatch, crop.seq_mismatch)


def test_pick_alignment_recovers_shift():
    g = torch.Generator().manual_seed(3)
    rollout, mask = _routes(B, R, g), _mask(g)
    train = _routes(B, R, g)
    train[:, 1:] = rollout[:, :-1]  # alignment +1 compares rollout[t] with train[t+1]
    assert pick_alignment(rollout, train, mask, num_experts=E, smoothing_tau=16.0) == 1
    assert pick_alignment(rollout, rollout.clone(), mask, num_experts=E, smoothing_tau=16.0) == 0


def test_select_probe_indices_quota_backfill_and_multiple():
    edges = (10, 20)
    g = torch.Generator().manual_seed(4)
    lengths = torch.tensor([5] * 5 + [15] * 5 + [25] * 5)
    buckets = length_buckets(lengths, edges)
    eligible = torch.ones(15, dtype=torch.bool)
    idx, taken = select_probe_indices(eligible, buckets, [3, 3, 2], 8, batch_multiple=4, backfill=False, generator=g)
    assert taken == [3, 3, 2] and idx.numel() == 8 and len(set(idx.tolist())) == 8

    no_long = eligible & (buckets != 2)
    idx, taken = select_probe_indices(no_long, buckets, [3, 3, 2], 8, batch_multiple=4, backfill=False, generator=g)
    assert idx.numel() == 4 and taken[2] == 0
    idx, taken = select_probe_indices(no_long, buckets, [3, 3, 2], 8, batch_multiple=4, backfill=True, generator=g)
    assert idx.numel() == 8 and taken[2] == 0

    idx, _ = select_probe_indices(eligible, buckets, [3, 3, 2], 5, batch_multiple=4, backfill=True, generator=g)
    assert idx.numel() == 4
    idx, _ = select_probe_indices(eligible, buckets, [3, 3, 2], 0, batch_multiple=4, backfill=True, generator=g)
    assert idx.numel() == 0


def test_length_buckets_match_rdc_convention():
    lengths = torch.tensor([1, 2047, 2048, 4095, 4096, 9999])
    assert length_buckets(lengths, (2048, 4096)).tolist() == [0, 0, 1, 1, 2, 2]
    assert bucket_quotas(256, 5) == [52, 51, 51, 51, 51]


def test_update_lag():
    gen = np.array([240, 240, 241])
    assert update_lag(1, gen[:1], 240, 4).tolist() == [0]
    assert update_lag(5, gen, 240, 4).tolist() == [4, 4, 0]
    assert update_lag(6, gen, 240, 4).tolist() == [5, 5, 1]


def test_masked_mean_abs_diff():
    a = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, 0.0, 100.0], [1.0, 1.0, 1.0]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    assert torch.allclose(masked_mean_abs_diff(a, b, mask), torch.tensor([1.5, 1.0]))


def test_probe_settings_validation():
    cfg = OmegaConf.create({"router_lag_probe": {"probe_size": 250, "output_dir": "/tmp/x"}})
    s = probe_settings(cfg)
    assert s.probe_size == 240 and s.num_experts == 128 and s.length_bucket_edges == (2048, 4096, 8192, 12288)
    try:
        probe_settings(OmegaConf.create({"router_lag_probe": {"probe_size": 64}}))
    except ValueError:
        pass
    else:
        raise AssertionError("missing output_dir must fail")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
