import torch

from types import SimpleNamespace

import pytest

from verl.utils.router_mismatch_metrics import (
    compute_length_conditional_percentiles,
    compute_router_mismatch_metrics,
    moe_layer_indices_from_hf_config,
)


def test_length_conditional_percentiles_use_midpoint_ranks_on_cpu():
    scores = torch.tensor([1.0, 2.0, 2.0, 4.0])
    lengths = torch.tensor([10.0, 20.0, 30.0, 40.0])

    result = compute_length_conditional_percentiles(scores, lengths, local_window=4)

    torch.testing.assert_close(result, torch.tensor([0.125, 0.5, 0.5, 0.875]))


def test_length_conditional_percentiles_separate_censored_group_on_cpu():
    scores = torch.tensor([1.0, 9.0, 2.0, 8.0, 4.0, 4.0])
    lengths = torch.tensor([10.0, 20.0, 30.0, 100.0, 100.0, 100.0])

    result = compute_length_conditional_percentiles(
        scores,
        lengths,
        local_window=3,
        censored_length=100,
        min_censored_count=3,
    )

    torch.testing.assert_close(result[:3], torch.tensor([1 / 6, 5 / 6, 0.5]))
    torch.testing.assert_close(result[3:], torch.tensor([5 / 6, 1 / 3, 1 / 3]))


def test_length_conditional_percentiles_ignore_invalid_responses_on_cpu():
    scores = torch.tensor([1.0, 2.0, 3.0])
    lengths = torch.tensor([10.0, 0.0, 30.0])

    result = compute_length_conditional_percentiles(scores, lengths, local_window=2)

    assert torch.isneginf(result[1])
    torch.testing.assert_close(result[[0, 2]], torch.tensor([0.25, 0.75]))


def test_router_mismatch_metrics_selects_best_alignment_on_cpu():
    rollout = torch.tensor(
        [
            [
                [[0, 1], [2, 3]],
                [[4, 5], [6, 7]],
                [[8, 9], [10, 11]],
            ]
        ]
    )
    train = torch.tensor(
        [
            [
                [[4, 5], [6, 7]],
                [[8, 9], [10, 11]],
                [[99, 98], [97, 96]],
            ]
        ]
    )
    response_mask = torch.tensor([[1, 1]])

    result = compute_router_mismatch_metrics(rollout, train, response_mask)

    assert result.alignment == -1
    assert "router/best_alignment" not in result.metrics
    assert result.metrics["router/response_token_match_rate"] == 1.0
    assert result.metrics["router/response_token_count"] == 2.0
    assert result.metrics["router/layer_0/response_token_match_rate"] == 1.0
    assert result.metrics["router/layer_1/response_token_match_rate"] == 1.0
    assert "router/response_top1_match_rate" not in result.metrics
    assert "router/response_token_mismatch_rate" not in result.metrics
    assert "router/layer_0/response_token_count" not in result.metrics
    assert "router/layer_1/response_token_count" not in result.metrics


def test_router_mismatch_metrics_counts_token_match_on_cpu():
    rollout = torch.tensor([[[[1, 2]], [[3, 4]]]])
    train = torch.tensor([[[[1, 9]], [[3, 4]]]])
    response_mask = torch.tensor([[1, 1]])

    result = compute_router_mismatch_metrics(rollout, train, response_mask, alignments=(0,))

    assert result.metrics["router/response_token_match_rate"] == 0.5
    assert result.metrics["router/layer_0/response_token_match_rate"] == 0.5
    assert "router/layer_0/response_token_mismatch_rate" not in result.metrics
    assert "router/layer_0/response_top1_match_rate" not in result.metrics
    assert "router/response_token_mismatch_rate" not in result.metrics
    assert "router/layer_0/response_token_count" not in result.metrics


def test_router_mismatch_metrics_treats_topk_as_unordered_set_on_cpu():
    rollout = torch.tensor([[[[1, 2, 3], [7, 8, 9]], [[4, 5, 6], [10, 11, 12]]]])
    train = torch.tensor([[[[3, 1, 2], [9, 7, 8]], [[6, 5, 4], [10, 12, 99]]]])
    response_mask = torch.tensor([[1, 1]])

    result = compute_router_mismatch_metrics(rollout, train, response_mask, alignments=(0,))

    assert result.metrics["router/response_token_match_rate"] == 0.5
    assert result.metrics["router/layer_0/response_token_match_rate"] == 1.0
    assert result.metrics["router/layer_1/response_token_match_rate"] == 0.5
    assert result.seq_mismatch is not None
    torch.testing.assert_close(result.seq_mismatch, torch.tensor([0.25]))


def test_router_mismatch_metrics_accepts_compact_integer_dtype_on_cpu():
    rollout = torch.tensor([[[[1, 2, 3]], [[4, 5, 6]]]], dtype=torch.uint8)
    train = torch.tensor([[[[3, 1, 2]], [[4, 6, 99]]]], dtype=torch.uint8)
    response_mask = torch.tensor([[1, 1]])

    result = compute_router_mismatch_metrics(rollout, train, response_mask, alignments=(0,))

    assert result.metrics["router/response_token_match_rate"] == 0.5
    assert result.metrics["router/layer_0/response_token_match_rate"] == 0.5
    assert result.seq_mismatch is not None
    torch.testing.assert_close(result.seq_mismatch, torch.tensor([0.5]))


def test_router_mismatch_metrics_overlap_fraction_sequence_mismatch_on_cpu():
    rollout = torch.tensor([[[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 3, 4, 5, 6, 7, 8]]]])
    train = torch.tensor([[[[1, 2, 3, 4, 5, 6, 7, 9]], [[1, 2, 3, 4, 9, 10, 11, 12]]]])
    response_mask = torch.tensor([[1, 1]])

    result = compute_router_mismatch_metrics(
        rollout,
        train,
        response_mask,
        alignments=(0,),
        metric_mode="overlap_fraction",
    )

    assert result.metrics["router/response_token_match_rate"] == 0.0
    assert result.seq_mismatch is not None
    torch.testing.assert_close(result.seq_mismatch, torch.tensor([0.3125]))
    assert result.metrics["router/seq_mismatch_mean"] == 0.3125


def test_router_mismatch_metrics_reports_per_layer_match_rates_on_cpu():
    rollout = torch.tensor([[[[1, 2], [7, 8]], [[3, 4], [9, 10]]]])
    train = torch.tensor([[[[1, 2], [7, 99]], [[3, 4], [9, 10]]]])
    response_mask = torch.tensor([[1, 1]])

    result = compute_router_mismatch_metrics(rollout, train, response_mask, alignments=(0,))

    assert result.metrics["router/response_token_match_rate"] == 0.5
    assert result.metrics["router/layer_0/response_token_match_rate"] == 1.0
    assert result.metrics["router/layer_1/response_token_match_rate"] == 0.5
    assert "router/layer_1/response_token_mismatch_rate" not in result.metrics
    assert "router/layer_1/response_token_count" not in result.metrics


def test_router_mismatch_metrics_reports_sequence_mismatch_on_cpu():
    rollout = torch.tensor(
        [
            [
                [[1], [7]],
                [[2], [8]],
                [[3], [9]],
            ]
        ]
    )
    train = torch.tensor(
        [
            [
                [[1], [0]],
                [[2], [8]],
                [[0], [0]],
            ]
        ]
    )
    response_mask = torch.tensor([[1, 1, 0]])

    result = compute_router_mismatch_metrics(rollout, train, response_mask, alignments=(0,))

    assert result.seq_mismatch is not None
    assert result.seq_valid_token_count is not None
    torch.testing.assert_close(result.seq_mismatch, torch.tensor([0.25]))
    torch.testing.assert_close(result.seq_valid_token_count, torch.tensor([2.0]))
    assert result.metrics["router/seq_mismatch_mean"] == 0.25
    assert result.metrics["router/seq_mismatch_max"] == 0.25


def test_expert_usage_probe_reuses_counts_for_candidate_distances_on_cpu():
    rollout = torch.tensor([[[[0, 1]], [[0, 1]]]], dtype=torch.uint8)
    train = torch.tensor([[[[0, 2]], [[0, 2]]]], dtype=torch.uint8)
    response_mask = torch.tensor([[1, 1]])

    result = compute_router_mismatch_metrics(
        rollout,
        train,
        response_mask,
        alignments=(0,),
        metric_mode="expert_usage_l1",
        expert_usage_smoothing_tau=2.0,
        expert_usage_num_experts=3,
        capture_expert_usage_counts=True,
    )

    assert result.expert_usage_distances is not None
    torch.testing.assert_close(result.expert_usage_distances["tv_raw"], torch.tensor([0.5]))
    torch.testing.assert_close(result.expert_usage_distances["tv_smooth"], torch.tensor([1.0 / 3.0]))
    torch.testing.assert_close(result.expert_usage_distances["l2"], torch.tensor([2.0**-0.5]))
    torch.testing.assert_close(result.expert_usage_distances["linf"], torch.tensor([0.5]))
    torch.testing.assert_close(result.expert_usage_distances["hellinger_sq"], torch.tensor([0.5]))
    torch.testing.assert_close(result.expert_usage_distances["js_normalized"], torch.tensor([0.5]))
    torch.testing.assert_close(result.expert_usage_distances["effective_support"], torch.tensor([2.0]))

    assert result.expert_usage_counts is not None
    assert result.expert_usage_counts["rollout"].dtype == torch.uint16
    assert result.expert_usage_counts["rollout"].shape == (1, 1, 3)
    torch.testing.assert_close(result.expert_usage_counts["rollout"], torch.tensor([[[2, 2, 0]]], dtype=torch.uint16))
    torch.testing.assert_close(result.expert_usage_counts["train"], torch.tensor([[[2, 0, 2]]], dtype=torch.uint16))


def test_expert_usage_probe_does_not_retain_counts_unless_requested_on_cpu():
    routes = torch.tensor([[[[0, 1]], [[0, 1]]]], dtype=torch.uint8)
    result = compute_router_mismatch_metrics(
        routes,
        routes,
        torch.tensor([[1, 1]]),
        alignments=(0,),
        metric_mode="expert_usage_l1",
    )

    assert result.expert_usage_counts is None


def _dense_plus_moe_routes():
    # 3 decoder layers, layer 0 dense: vLLM leaves its slot all-zero, Megatron omits it.
    moe_routes = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])  # [1, 2 tokens, 2 MoE layers, topk 2]
    dense_slot = torch.zeros(1, 2, 1, 2, dtype=moe_routes.dtype)
    rollout = torch.cat([dense_slot, moe_routes], dim=2)  # [1, 2, 3, 2]
    return rollout, moe_routes.clone()


def test_router_mismatch_metrics_drops_dense_layer_slots_on_cpu():
    rollout, train = _dense_plus_moe_routes()
    response_mask = torch.tensor([[1, 1]])

    result = compute_router_mismatch_metrics(
        rollout, train, response_mask, alignments=(0,), rollout_moe_layer_indices=[1, 2]
    )

    assert result.metrics["router/response_token_match_rate"] == 1.0
    assert result.metrics["router/layer_0/response_token_match_rate"] == 1.0
    assert result.metrics["router/layer_1/response_token_match_rate"] == 1.0


def test_router_mismatch_metrics_fails_closed_on_layer_count_mismatch_on_cpu():
    rollout, train = _dense_plus_moe_routes()
    response_mask = torch.tensor([[1, 1]])

    with pytest.raises(ValueError, match="rollout_moe_layer_indices"):
        compute_router_mismatch_metrics(rollout, train, response_mask, alignments=(0,))


def test_moe_layer_indices_from_hf_config_on_cpu():
    moonlight = SimpleNamespace(num_hidden_layers=27, first_k_dense_replace=1, moe_layer_freq=1)
    assert moe_layer_indices_from_hf_config(moonlight) == list(range(1, 27))

    qwen3_moe = SimpleNamespace(num_hidden_layers=48, mlp_only_layers=[], decoder_sparse_step=1)
    assert moe_layer_indices_from_hf_config(qwen3_moe) is None


def test_router_mismatch_metrics_gross_counts_cancelled_changes_on_cpu():
    # 1 response, 3 tokens, 1 layer, top-2. Token 0 unchanged, token 1 swaps one expert
    # (1 slot), token 2 is masked out. The usage histogram (net) can hide swaps; gross counts slots.
    rollout = torch.tensor([[[[0, 1]], [[2, 3]], [[4, 5]]]])
    train = torch.tensor([[[[0, 1]], [[2, 4]], [[6, 7]]]])
    response_mask = torch.tensor([[1, 1, 0]])

    result = compute_router_mismatch_metrics(
        rollout, train, response_mask, alignments=(0,), metric_mode="expert_usage_l1", capture_gross=True
    )

    gross = result.gross
    torch.testing.assert_close(gross["seq_flip_rate"], torch.tensor([0.5]))
    torch.testing.assert_close(gross["seq_overlap_mismatch"], torch.tensor([0.25]))
    assert gross["seq_layer_changed_slots"].tolist() == [[1]]
    torch.testing.assert_close(gross["token_flip_rate"], torch.tensor([[0.0, 1.0, 0.0]]))


def test_router_mismatch_metrics_samples_aligned_response_routes_on_cpu():
    # prompt token at position 0, response tokens at 1..2; train side recorded one position later.
    rollout = torch.tensor([[[[9, 9]], [[1, 2]], [[3, 4]]], [[[9, 9]], [[5, 6]], [[7, 8]]]])
    train = torch.tensor([[[[0, 0]], [[0, 0]], [[1, 2]], [[3, 4]]], [[[0, 0]], [[0, 0]], [[5, 6]], [[7, 8]]]])
    response_mask = torch.tensor([[1, 1], [1, 1]])

    result = compute_router_mismatch_metrics(rollout, train, response_mask, alignments=(0,), capture_route_indices=[1])

    sampled = result.sampled_routes
    assert sampled["indices"].tolist() == [1]
    assert sampled["rollout"].dtype == torch.uint8
    assert sampled["rollout"].tolist() == [[[[5, 6]], [[7, 8]]]]
    assert sampled["train"].tolist() == [[[[5, 6]], [[7, 8]]]]
    assert result.metrics["router/response_token_match_rate"] == 1.0


def test_stratified_route_sample_spans_lengths_on_cpu():
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    lengths = torch.arange(1, 101)  # rows sorted short to long, like after balance_batch
    response_mask = (torch.arange(100).unsqueeze(0) < lengths.unsqueeze(1)).long()

    picks = RayPPOTrainer._stratified_route_sample(None, response_mask, 4, seed=0)

    assert len(picks) == 4
    assert max(picks) >= 75  # a prefix slice would only return rows 0..3
    assert picks == RayPPOTrainer._stratified_route_sample(None, response_mask, 4, seed=0)
