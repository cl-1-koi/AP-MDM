"""Numerical and gradient tests for the paper-shaped star-graph models."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from repro.ip_star_data import TEST_SEED, collate_star_graphs, generate_graphs
from repro.ip_star_model import (
    InsertionDecoder,
    InsertionPosterior,
    TransformerSpec,
    fixed_order_objective,
    learned_ip_objective,
    random_ip_objective,
)


@pytest.fixture
def tiny_batch():
    return collate_star_graphs(generate_graphs(4, TEST_SEED))


@pytest.fixture
def tiny_spec():
    return TransformerSpec(width=32, layers=1, heads=4, dropout=0.0)


def _generator() -> torch.Generator:
    return torch.Generator().manual_seed(1234)


def _assert_finite_gradients(module: torch.nn.Module) -> None:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    assert any(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
    assert sum(float(gradient.abs().sum()) for gradient in gradients if gradient is not None) > 0


@pytest.mark.parametrize("objective", [fixed_order_objective, random_ip_objective])
def test_decoder_objectives_are_finite_and_differentiable(
    tiny_batch, tiny_spec, objective
):
    decoder = InsertionDecoder(tiny_spec)
    result = objective(decoder, tiny_batch, _generator())
    assert torch.isfinite(result.loss)
    assert all(torch.isfinite(metric) for metric in result.metrics.values())
    result.loss.backward()
    _assert_finite_gradients(decoder)


def test_learned_objective_reaches_decoder_and_posterior(tiny_batch, tiny_spec):
    decoder = InsertionDecoder(tiny_spec)
    posterior = InsertionPosterior(tiny_spec)
    result = learned_ip_objective(decoder, posterior, tiny_batch, _generator())
    assert torch.isfinite(result.loss)
    assert all(torch.isfinite(metric) for metric in result.metrics.values())
    result.loss.backward()
    _assert_finite_gradients(decoder)
    _assert_finite_gradients(posterior)


def test_objective_sampling_is_seed_reproducible(tiny_batch, tiny_spec):
    decoder = InsertionDecoder(tiny_spec)
    posterior = InsertionPosterior(tiny_spec)
    first = learned_ip_objective(decoder, posterior, tiny_batch, _generator())
    second = learned_ip_objective(decoder, posterior, tiny_batch, _generator())
    torch.testing.assert_close(first.loss, second.loss)
    assert first.metrics.keys() == second.metrics.keys()
    for name in first.metrics:
        torch.testing.assert_close(first.metrics[name], second.metrics[name])


def test_policy_location_distribution_is_normalized(tiny_batch, tiny_spec):
    decoder = InsertionDecoder(tiny_spec)
    partial = tiny_batch.targets[:, :3].clone()
    partial_mask = torch.ones_like(partial, dtype=torch.bool)
    output = decoder(tiny_batch, partial, partial_mask)
    probability = F.softmax(output.location_logits, dim=-1)
    torch.testing.assert_close(probability.sum(dim=-1), torch.ones(probability.shape[0]))
    assert torch.isfinite(output.location_logits[:, -1]).all()
