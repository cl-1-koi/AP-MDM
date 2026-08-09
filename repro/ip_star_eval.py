"""Greedy decoding and held-out metrics for star-graph planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch

from repro.ip_star_data import EOS_ID, NODE_VOCAB_SIZE, PAD_ID, StarBatch, StarGraph
from repro.ip_star_model import InsertionDecoder


DecodeMode = Literal["fixed", "insertion"]


@dataclass(frozen=True)
class DecodedBatch:
    sequences: tuple[tuple[int, ...], ...]
    terminated: tuple[bool, ...]
    steps: int


@torch.inference_mode()
def greedy_decode(
    decoder: InsertionDecoder,
    batch: StarBatch,
    *,
    mode: DecodeMode,
    max_tokens: int = 32,
) -> DecodedBatch:
    """Decode without search, using either append-only or arbitrary insertion."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    sequences: list[list[int]] = [[] for _ in range(batch.targets.shape[0])]
    terminated = [False] * len(sequences)
    was_training = decoder.training
    decoder.eval()

    steps = 0
    for steps in range(1, max_tokens + 2):
        width = max(1, max((len(sequence) for sequence in sequences), default=0))
        partial = torch.full(
            (len(sequences), width),
            PAD_ID,
            dtype=torch.long,
            device=batch.targets.device,
        )
        partial_mask = torch.zeros_like(partial, dtype=torch.bool)
        for row, sequence in enumerate(sequences):
            if sequence:
                partial[row, : len(sequence)] = torch.tensor(
                    sequence, dtype=torch.long, device=partial.device
                )
                partial_mask[row, : len(sequence)] = True

        output = decoder(batch, partial, partial_mask)
        for row, sequence in enumerate(sequences):
            if terminated[row]:
                continue
            if mode == "fixed":
                slot = len(sequence)
                logits = torch.cat(
                    (
                        output.content_logits[row, slot, :NODE_VOCAB_SIZE],
                        output.content_logits[row, slot, EOS_ID : EOS_ID + 1],
                    )
                )
                choice = int(logits.argmax().item())
                if choice == NODE_VOCAB_SIZE:
                    terminated[row] = True
                elif len(sequence) < max_tokens:
                    sequence.append(choice)
            elif mode == "insertion":
                location_logits = torch.cat(
                    (
                        output.location_logits[row, : len(sequence) + 1],
                        output.location_logits[row, -1:],
                    )
                )
                location = int(
                    location_logits.argmax().item()
                )
                if location == len(sequence) + 1:
                    terminated[row] = True
                elif len(sequence) < max_tokens:
                    token = int(
                        output.content_logits[row, location, :NODE_VOCAB_SIZE]
                        .argmax()
                        .item()
                    )
                    sequence.insert(location, token)
            else:
                raise ValueError(f"unknown decode mode: {mode}")
        if all(terminated):
            break

    decoder.train(was_training)
    return DecodedBatch(
        sequences=tuple(tuple(sequence) for sequence in sequences),
        terminated=tuple(terminated),
        steps=steps,
    )


def levenshtein_distance(left: Sequence[int], right: Sequence[int]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def is_valid_plan(sequence: Sequence[int], graph: StarGraph) -> bool:
    if len(sequence) < 2 or len(sequence) % 2:
        return False
    edges = list(zip(sequence[::2], sequence[1::2]))
    if edges[0][0] != graph.source or edges[-1][1] != graph.goal:
        return False
    if any(left[1] != right[0] for left, right in zip(edges, edges[1:])):
        return False
    available = set(graph.edge_list)
    return all(edge in available for edge in edges)


def planning_metrics(
    decoded: DecodedBatch, graphs: Sequence[StarGraph]
) -> dict[str, float]:
    if len(decoded.sequences) != len(graphs):
        raise ValueError("decoded batch and graph count differ")
    if not graphs:
        raise ValueError("cannot score an empty graph sequence")
    exact = []
    valid = []
    edit = []
    hamming = []
    normalized_edit = []
    positional_matches = 0
    positional_total = 0
    predicted_lengths = []
    target_lengths = []
    for prediction, graph in zip(decoded.sequences, graphs):
        target = graph.target_tokens
        distance = levenshtein_distance(prediction, target)
        overlap = min(len(prediction), len(target))
        hamming_distance = sum(
            prediction[index] != target[index] for index in range(overlap)
        ) + abs(len(prediction) - len(target))
        exact.append(prediction == target)
        valid.append(is_valid_plan(prediction, graph))
        edit.append(distance)
        hamming.append(hamming_distance)
        normalized_edit.append(distance / max(len(prediction), len(target), 1))
        width = max(len(prediction), len(target))
        positional_matches += sum(
            prediction[index] == target[index]
            for index in range(min(len(prediction), len(target)))
        )
        positional_total += width
        predicted_lengths.append(len(prediction))
        target_lengths.append(len(target))
    count = len(graphs)
    return {
        "exact_match": sum(exact) / count,
        "valid_plan": sum(valid) / count,
        "terminated": sum(decoded.terminated) / count,
        "token_accuracy": positional_matches / max(positional_total, 1),
        "edit_distance": sum(edit) / count,
        "hamming_distance": sum(hamming) / count,
        "normalized_edit_distance": sum(normalized_edit) / count,
        "predicted_length": sum(predicted_lengths) / count,
        "target_length": sum(target_lengths) / count,
        "decode_steps": float(decoded.steps),
    }


def teacher_forced_content_accuracy(
    decoder: InsertionDecoder,
    batch: StarBatch,
    *,
    generator: torch.Generator,
) -> float:
    """A cheap diagnostic for content learning under random partial states."""
    prefix_lengths = torch.floor(
        torch.rand(batch.lengths.shape, device=batch.lengths.device, generator=generator)
        * (batch.lengths + 1)
    ).long()
    positions = torch.arange(batch.targets.shape[1], device=batch.targets.device)[None]
    selected = batch.target_mask & (positions < prefix_lengths[:, None])
    partial = torch.full_like(batch.targets, PAD_ID)
    partial_mask = selected.clone()
    partial[selected] = batch.targets[selected]
    output = decoder(batch, partial, partial_mask)
    active = prefix_lengths < batch.lengths
    if not bool(active.any()):
        return 1.0
    rows = torch.arange(batch.targets.shape[0], device=batch.targets.device)[active]
    slot = prefix_lengths[active]
    target = batch.targets[rows, slot]
    prediction = output.content_logits[rows, slot, :NODE_VOCAB_SIZE].argmax(dim=-1)
    return float((prediction == target).float().mean().item())
