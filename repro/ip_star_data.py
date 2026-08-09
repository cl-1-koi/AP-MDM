"""Deterministic hard star-graph data for the Insertion Process gate.

The graph construction is adapted from ``dhruvdcoder/ILM`` commit
``6cc27f104fd926c8256aff28682c3fe66050ce77``.  The dimensions and split sizes
are the ones exposed by its ``vstar_medium_v2`` configuration, which match the
configuration described by Zhang et al. (2026).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


NODE_VOCAB_SIZE = 56
GRAPH_BOS_ID = 56
PAD_ID = 57
EOS_ID = 58
FULL_VOCAB_SIZE = 59

DEGREE = 5
MIN_PATH_LENGTH = 6
MAX_PATH_LENGTH = 12
MIN_PLAN_LENGTH = 5

TRAIN_COUNT = 50_000
VALIDATION_COUNT = 100
TEST_COUNT = 5_000

TRAIN_SEED = 314_159
VALIDATION_SEED = 271_828
TEST_SEED = 161_803

SPLIT_SHA256 = {
    "train": "8fc03eda5cb9a43e9a5f79c6e31035a0235f16e728ac6d4cda70de2b35b13baf",
    "validation": "9a7adaf0bdc860a7bf18db336d51f5238b7ca2abb7f5071bbdeba010e4d01135",
    "test": "1e359d180d726751b495ef248baf626ac13b6928693de27746126f616f991acd",
}


@dataclass(frozen=True)
class StarGraph:
    edge_list: tuple[tuple[int, int], ...]
    source: int
    goal: int
    path: tuple[int, ...]

    @property
    def condition_tokens(self) -> tuple[int, ...]:
        edges = tuple(node for edge in self.edge_list for node in edge)
        return edges + (self.source, self.goal, GRAPH_BOS_ID)

    @property
    def target_tokens(self) -> tuple[int, ...]:
        return tuple(node for edge in zip(self.path, self.path[1:]) for node in edge)

    def canonical_record(self) -> dict:
        return asdict(self)


def generate_star_graph(
    numpy_rng: np.random.RandomState,
    python_rng: random.Random,
    *,
    degree: int = DEGREE,
    min_path_length: int = MIN_PATH_LENGTH,
    max_path_length: int = MAX_PATH_LENGTH,
    min_plan_length: int = MIN_PLAN_LENGTH,
    node_vocab_size: int = NODE_VOCAB_SIZE,
) -> StarGraph:
    """Generate the cited asymmetric variable-arm-length star distribution."""
    if node_vocab_size < max_path_length * degree - degree + 1:
        raise ValueError("node vocabulary is too small for unique arm nodes")
    if min_plan_length > min_path_length:
        raise ValueError("minimum plan length exceeds minimum path length")

    path_lengths = numpy_rng.randint(min_path_length, max_path_length + 1, degree)
    join_locations = np.asarray(
        [numpy_rng.randint(int(length) - min_plan_length) for length in path_lengths],
        dtype=np.int64,
    )
    node_count = int(path_lengths.sum()) - degree + 1
    unique_node_ids = numpy_rng.choice(node_vocab_size, node_count, replace=False)
    center_node = int(unique_node_ids[0])

    arms: list[list[tuple[int, int]]] = []
    next_unique = 1
    for path_length, join_location in zip(path_lengths, join_locations):
        arm: list[tuple[int, int]] = []
        previous: int | None = None
        for position in range(int(path_length)):
            if position == int(join_location):
                node = center_node
            else:
                node = int(unique_node_ids[next_unique])
                next_unique += 1
            if previous is not None:
                arm.append((previous, node))
            previous = node
        arms.append(arm)

    designated = arms[0]
    path = (designated[0][0],) + tuple(edge[1] for edge in designated)
    edges = [edge for arm in arms for edge in arm]
    python_rng.shuffle(edges)
    graph = StarGraph(
        edge_list=tuple(edges), source=path[0], goal=path[-1], path=path
    )
    validate_graph(graph, degree=degree)
    return graph


def validate_graph(graph: StarGraph, *, degree: int = DEGREE) -> None:
    if not MIN_PATH_LENGTH <= len(graph.path) <= MAX_PATH_LENGTH:
        raise ValueError("designated path length is outside the declared range")
    if graph.source != graph.path[0] or graph.goal != graph.path[-1]:
        raise ValueError("source/goal do not match path endpoints")
    if len(set(graph.path)) != len(graph.path):
        raise ValueError("designated path repeats a node")
    if any(not 0 <= node < NODE_VOCAB_SIZE for edge in graph.edge_list for node in edge):
        raise ValueError("edge node is outside the vocabulary")
    if any(edge not in graph.edge_list for edge in zip(graph.path, graph.path[1:])):
        raise ValueError("a designated path edge is missing from the graph")

    nodes = [node for edge in graph.edge_list for node in edge]
    counts = {node: nodes.count(node) for node in set(nodes)}
    centers = [node for node, count in counts.items() if count >= degree]
    if len(centers) != 1:
        raise ValueError(f"expected exactly one shared star center, found {centers}")


def generate_graphs(count: int, seed: int) -> list[StarGraph]:
    numpy_rng = np.random.RandomState(int(seed))
    python_rng = random.Random(int(seed) ^ 0x5EED5EED)
    return [generate_star_graph(numpy_rng, python_rng) for _ in range(int(count))]


def graphs_sha256(graphs: Iterable[StarGraph]) -> str:
    digest = hashlib.sha256()
    for graph in graphs:
        row = json.dumps(
            graph.canonical_record(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest.update(row)
        digest.update(b"\n")
    return digest.hexdigest()


class StarGraphDataset(Dataset[StarGraph]):
    def __init__(self, graphs: Sequence[StarGraph]):
        self.graphs = tuple(graphs)

    @classmethod
    def generated(cls, count: int, seed: int) -> "StarGraphDataset":
        return cls(generate_graphs(count, seed))

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> StarGraph:
        return self.graphs[index]


@dataclass(frozen=True)
class StarBatch:
    conditions: torch.Tensor
    condition_mask: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor
    lengths: torch.Tensor

    def to(self, device: torch.device | str) -> "StarBatch":
        return StarBatch(
            conditions=self.conditions.to(device),
            condition_mask=self.condition_mask.to(device),
            targets=self.targets.to(device),
            target_mask=self.target_mask.to(device),
            lengths=self.lengths.to(device),
        )


def collate_star_graphs(graphs: Sequence[StarGraph]) -> StarBatch:
    if not graphs:
        raise ValueError("cannot collate an empty batch")
    condition_length = max(len(graph.condition_tokens) for graph in graphs)
    target_length = max(len(graph.target_tokens) for graph in graphs)
    conditions = torch.full((len(graphs), condition_length), PAD_ID, dtype=torch.long)
    targets = torch.full((len(graphs), target_length), PAD_ID, dtype=torch.long)
    condition_mask = torch.zeros_like(conditions, dtype=torch.bool)
    target_mask = torch.zeros_like(targets, dtype=torch.bool)
    for row, graph in enumerate(graphs):
        condition = torch.tensor(graph.condition_tokens, dtype=torch.long)
        target = torch.tensor(graph.target_tokens, dtype=torch.long)
        conditions[row, : condition.numel()] = condition
        targets[row, : target.numel()] = target
        condition_mask[row, : condition.numel()] = True
        target_mask[row, : target.numel()] = True
    return StarBatch(
        conditions=conditions,
        condition_mask=condition_mask,
        targets=targets,
        target_mask=target_mask,
        lengths=target_mask.sum(dim=-1),
    )
