"""Tests for the variable-length S4 AO-IP and learned IP arms."""

from __future__ import annotations

import torch

from repro.s4_insertion import (
    S4InsertionDecoder,
    S4InsertionPosterior,
    decode_batch,
    decoder_spec,
    insertion_loss,
    make_batch,
    posterior_spec,
)
from repro.small_grokking import frozen_splits


def _fixture(batch_size: int = 4):
    train, _, _ = frozen_splits(42)
    puzzles = torch.as_tensor(train.puzzles[:batch_size]).long()
    solutions = torch.as_tensor(train.solutions[:batch_size]).long()
    generator = torch.Generator().manual_seed(17)
    return make_batch(puzzles, solutions, generator)


def test_s4_insertion_batch_starts_with_six_givens_and_ten_targets():
    batch = _fixture()
    assert batch.conditions.shape == (4, 64)
    assert torch.all(batch.base_mask.sum(-1) == 6)
    assert torch.all(batch.target_mask.sum(-1) == 10)
    assert torch.all(batch.lengths == 10)


def test_ao_ip_objective_is_finite_and_reaches_decoder():
    batch = _fixture()
    decoder = S4InsertionDecoder(decoder_spec(width=32, layers=1, heads=4))
    loss, metrics = insertion_loss(
        decoder, None, batch, "ao_ip", torch.Generator().manual_seed(23)
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    loss.backward()
    assert decoder.content_head.weight.grad is not None
    assert decoder.location_head.weight.grad is not None


def test_learned_ip_objective_reaches_decoder_and_posterior():
    batch = _fixture()
    spec = decoder_spec(width=32, layers=1, heads=4)
    decoder = S4InsertionDecoder(spec)
    posterior = S4InsertionPosterior(posterior_spec(spec))
    loss, metrics = insertion_loss(
        decoder, posterior, batch, "ip", torch.Generator().manual_seed(29)
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    loss.backward()
    assert decoder.content_head.weight.grad is not None
    assert posterior.head.weight.grad is not None


def test_untrained_variable_length_decoder_emits_bounded_sequences():
    batch = _fixture(batch_size=2)
    decoder = S4InsertionDecoder(decoder_spec(width=32, layers=1, heads=4))
    sequences, terminated, steps = decode_batch(
        decoder, batch, bf16=False, max_steps=3
    )
    assert len(sequences) == len(terminated) == 2
    assert 1 <= steps <= 3
    assert all(6 <= len(sequence) <= 9 for sequence in sequences)
