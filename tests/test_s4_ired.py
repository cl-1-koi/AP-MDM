"""Tests for the S4 Transformer IRED comparator."""

from __future__ import annotations

import torch

from repro.model import DDiTBlock, Rotary
from repro.s4_ired import S4Energy, conditions, generate, ired_loss
from repro.small_grokking import frozen_splits


def _fixture(batch_size: int = 4):
    train, _, _ = frozen_splits(42)
    puzzles = torch.as_tensor(train.puzzles[:batch_size]).long()
    solutions = torch.as_tensor(train.solutions[:batch_size]).long()
    generator = torch.Generator().manual_seed(47)
    condition = conditions(puzzles, solutions, generator)
    return condition, puzzles, solutions, generator


def test_manual_attention_matches_sdpa_forward():
    torch.manual_seed(5)
    block = DDiTBlock(32, 4, 32, dropout=0.0).eval()
    hidden = torch.randn(2, 12, 32)
    conditioning = torch.randn(2, 32)
    rotary = Rotary(8)(12, hidden.device, hidden.dtype)
    fused = block(hidden, rotary, conditioning)
    explicit = block(hidden, rotary, conditioning, manual_attention=True)
    torch.testing.assert_close(explicit, fused, rtol=2e-5, atol=2e-6)


def test_ired_loss_supports_mixed_second_derivative_to_parameters():
    condition, _, solutions, generator = _fixture()
    model = S4Energy(width=32, blocks=1, heads=4)
    loss, metrics = ired_loss(model, condition, solutions, generator)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    loss.backward()
    assert model.output_embed.weight.grad is not None
    assert model.blocks[0].attn_qkv.weight.grad is not None
    assert model.residual.weight.grad is not None


def test_ired_generation_is_bounded_and_clamps_given_digits():
    condition, puzzles, _, _ = _fixture(batch_size=2)
    model = S4Energy(width=32, blocks=1, heads=4)
    generated, reasoning = generate(
        model, condition, puzzles, seed=53, inner_steps=1
    )
    assert generated.shape == (2, 16)
    assert reasoning["landscapes"] == 10
    assert reasoning["inner_steps_per_landscape"] == 1
    flat_puzzles = puzzles.reshape(2, 16)
    assert torch.equal(
        generated[flat_puzzles != 0], flat_puzzles[flat_puzzles != 0]
    )
