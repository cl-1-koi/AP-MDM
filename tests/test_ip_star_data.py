"""Fidelity tests for the hard star-graph benchmark."""

from __future__ import annotations

from repro.ip_star_data import (
    GRAPH_BOS_ID,
    MAX_PATH_LENGTH,
    MIN_PATH_LENGTH,
    NODE_VOCAB_SIZE,
    SPLIT_SHA256,
    TEST_SEED,
    TRAIN_COUNT,
    TRAIN_SEED,
    VALIDATION_COUNT,
    VALIDATION_SEED,
    collate_star_graphs,
    generate_graphs,
    graphs_sha256,
)


def test_seeded_generator_is_deterministic_and_distribution_valid():
    first = generate_graphs(32, TEST_SEED)
    second = generate_graphs(32, TEST_SEED)
    assert first == second
    assert graphs_sha256(first) == graphs_sha256(second)
    assert len({graphs_sha256(first), graphs_sha256(generate_graphs(32, TEST_SEED + 1))}) == 2
    for graph in first:
        assert MIN_PATH_LENGTH <= len(graph.path) <= MAX_PATH_LENGTH
        assert len(set(graph.path)) == len(graph.path)
        assert graph.condition_tokens[-3:] == (graph.source, graph.goal, GRAPH_BOS_ID)
        assert graph.target_tokens == tuple(
            node for edge in zip(graph.path, graph.path[1:]) for node in edge
        )
        assert all(0 <= token < NODE_VOCAB_SIZE for token in graph.target_tokens)


def test_collator_preserves_variable_lengths_and_padding():
    graphs = generate_graphs(64, TEST_SEED)
    batch = collate_star_graphs(graphs)
    assert batch.conditions.shape[0] == 64
    assert batch.targets.shape[0] == 64
    assert batch.condition_mask.sum(dim=-1).tolist() == [
        len(graph.condition_tokens) for graph in graphs
    ]
    assert batch.lengths.tolist() == [len(graph.target_tokens) for graph in graphs]
    assert int(batch.lengths.min()) == 2 * (MIN_PATH_LENGTH - 1)
    assert int(batch.lengths.max()) == 2 * (MAX_PATH_LENGTH - 1)


def test_declared_validation_split_hash_is_frozen():
    validation = generate_graphs(VALIDATION_COUNT, VALIDATION_SEED)
    assert graphs_sha256(validation) == SPLIT_SHA256["validation"]


def test_declared_train_split_metadata_is_consistent():
    # The full 50k hash is exercised by the training CLI before optimization;
    # keep this unit test fast while guarding accidental metadata edits.
    assert TRAIN_COUNT == 50_000
    assert TRAIN_SEED == 314_159
    assert len(SPLIT_SHA256["train"]) == 64
