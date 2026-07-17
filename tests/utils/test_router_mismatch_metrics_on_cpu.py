import torch

from verl.utils.router_mismatch_metrics import (
    compute_length_conditional_percentiles,
    compute_router_mismatch_metrics,
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
