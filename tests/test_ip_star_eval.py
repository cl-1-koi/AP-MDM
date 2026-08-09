"""Tests for search-free insertion decoding and planning metrics."""

from __future__ import annotations

import torch

from repro.ip_star_data import TEST_SEED, collate_star_graphs, generate_graphs
from repro.ip_star_eval import (
    DecodedBatch,
    greedy_decode,
    is_valid_plan,
    levenshtein_distance,
    planning_metrics,
)
from repro.ip_star_model import DecoderOutput


class _AppendOracle(torch.nn.Module):
    def forward(self, batch, partial, partial_mask):
        lengths = partial_mask.sum(dim=-1)
        slots = int(lengths.max().item()) + 1
        location = torch.full((len(lengths), slots + 1), -100.0)
        content = torch.full((len(lengths), slots, 59), -100.0)
        for row, length_tensor in enumerate(lengths):
            length = int(length_tensor.item())
            if length == int(batch.lengths[row].item()):
                location[row, -1] = 100.0
            else:
                location[row, length] = 100.0
                token = int(batch.targets[row, length].item())
                content[row, length, token] = 100.0
        slot_mask = torch.arange(slots)[None] <= lengths[:, None]
        return DecoderOutput(location, content, slot_mask)


def test_levenshtein_distance_known_cases():
    assert levenshtein_distance((), ()) == 0
    assert levenshtein_distance((1, 2), (1, 2)) == 0
    assert levenshtein_distance((1, 2), (1, 3, 2)) == 1
    assert levenshtein_distance((1, 2), (3, 4)) == 2


def test_ground_truth_is_an_exact_valid_plan():
    graphs = generate_graphs(8, TEST_SEED)
    decoded = DecodedBatch(
        sequences=tuple(graph.target_tokens for graph in graphs),
        terminated=(True,) * len(graphs),
        steps=1,
    )
    assert all(is_valid_plan(graph.target_tokens, graph) for graph in graphs)
    metrics = planning_metrics(decoded, graphs)
    assert metrics["exact_match"] == 1.0
    assert metrics["valid_plan"] == 1.0
    assert metrics["terminated"] == 1.0
    assert metrics["token_accuracy"] == 1.0
    assert metrics["edit_distance"] == 0.0


def test_invalid_disconnected_plan_is_rejected():
    graph = generate_graphs(1, TEST_SEED)[0]
    broken = list(graph.target_tokens)
    broken[2] = (broken[2] + 1) % 56
    assert not is_valid_plan(broken, graph)


def test_insertion_decoder_uses_per_row_term_index_and_halts():
    graphs = generate_graphs(16, TEST_SEED)
    short = min(graphs, key=lambda graph: len(graph.target_tokens))
    long = max(graphs, key=lambda graph: len(graph.target_tokens))
    assert len(short.target_tokens) < len(long.target_tokens)
    selected = [short, long]
    batch = collate_star_graphs(selected)
    decoded = greedy_decode(_AppendOracle(), batch, mode="insertion", max_tokens=32)
    assert decoded.terminated == (True, True)
    assert decoded.sequences == tuple(graph.target_tokens for graph in selected)
