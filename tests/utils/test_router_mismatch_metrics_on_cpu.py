import torch

from verl.utils.router_mismatch_metrics import compute_router_mismatch_metrics


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
