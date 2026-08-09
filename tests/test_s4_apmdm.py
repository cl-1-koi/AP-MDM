"""Tests for the S4 four-operation AP-MDM arm."""

from __future__ import annotations

import torch

from repro.model import APMDMEncoder
from repro.s4_apmdm import (
    CONDITION_LENGTH,
    apmdm_loss,
    conditions,
    corrupt_targets,
    generate,
    model_spec,
)
from repro.small_grokking import CELLS, frozen_splits


def _fixture(batch_size: int = 64):
    train, _, _ = frozen_splits(42)
    repeats = (batch_size + train.n - 1) // train.n
    puzzles = torch.as_tensor(train.puzzles).long().repeat(repeats, 1, 1)[:batch_size]
    solutions = torch.as_tensor(train.solutions).long().repeat(repeats, 1, 1)[:batch_size]
    generator = torch.Generator().manual_seed(41)
    condition = conditions(puzzles, solutions, generator)
    return condition, solutions, generator


def test_corruption_batch_exercises_all_four_operations():
    condition, solutions, generator = _fixture()
    batch = corrupt_targets(condition, solutions.reshape(-1, CELLS), generator)
    assert batch.indices.shape == batch.attention_mask.shape == batch.loss_mask.shape
    assert batch.indices.shape[1] > CONDITION_LENGTH
    assert int((batch.indices == 25).sum()) > 0
    assert int(batch.remask.sum()) > 0
    assert int(batch.insert.sum()) > 0
    assert int(batch.delete.sum()) > 0
    assert torch.all(batch.attention_mask[:, :CONDITION_LENGTH])
    assert not torch.any(batch.loss_mask[:, :CONDITION_LENGTH])


def test_apmdm_loss_reaches_every_operation_head():
    condition, solutions, generator = _fixture(batch_size=8)
    batch = corrupt_targets(condition, solutions.reshape(-1, CELLS), generator)
    model = APMDMEncoder(
        model_spec(width=32, blocks=1, heads=4), time_conditioning=True
    )
    loss, metrics = apmdm_loss(model, batch)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    loss.backward()
    head = model.output_layer
    assert head.unmasking_head.weight.grad is not None
    assert head.remasking_head.weight.grad is not None
    assert head.expansion_head.weight.grad is not None
    assert head.contraction_head.weight.grad is not None


def test_untrained_apmdm_generation_is_bounded():
    condition, _, _ = _fixture(batch_size=2)
    model = APMDMEncoder(
        model_spec(width=32, blocks=1, heads=4), time_conditioning=True
    )
    states, terminated, traces = generate(
        model, condition, bf16=False, max_steps=2
    )
    assert len(states) == len(terminated) == len(traces) == 2
    assert all(trace["steps"] >= 1 for trace in traces)
