"""Exact-enumeration gates for the paper-native Insertion Process."""

from __future__ import annotations

import itertools

import pytest
import torch

from repro.ip_objective import (
    all_permutations,
    enumerated_expected_rloo_surrogate,
    exact_permutation_elbo,
    permutation_to_slots,
    pl_log_probability,
    slots_to_permutation,
)


@pytest.mark.parametrize("length", range(1, 7))
def test_permutation_slot_mapping_is_bijective(length):
    for permutation in all_permutations(length):
        slots = permutation_to_slots(permutation)
        assert all(0 <= slot <= step for step, slot in enumerate(slots))
        assert slots_to_permutation(slots) == permutation


def test_paper_mapping_example():
    # Paper example sigma=(3,1,2), z=(1,1,2), converted to zero-based indices.
    assert permutation_to_slots((2, 0, 1)) == (0, 0, 1)
    assert slots_to_permutation((0, 0, 1)) == (2, 0, 1)


@pytest.mark.parametrize("length", range(1, 7))
def test_plackett_luce_distribution_normalizes(length):
    logits = torch.linspace(-0.7, 0.9, length, dtype=torch.float64)
    probability = sum(
        pl_log_probability(logits, permutation).exp()
        for permutation in itertools.permutations(range(length))
    )
    assert torch.allclose(probability, torch.ones_like(probability), atol=1e-12)


def test_enumerated_rloo_gradient_matches_exact_elbo():
    dtype = torch.float64
    q_exact = torch.tensor([0.35, -0.8, 0.15], dtype=dtype, requires_grad=True)
    action_exact = (
        torch.linspace(-1.3, 0.9, (1 << 3) * 3, dtype=dtype)
        .reshape(1 << 3, 3)
        .clone()
        .requires_grad_(True)
    )
    termination_exact = torch.tensor(-0.41, dtype=dtype, requires_grad=True)
    exact = exact_permutation_elbo(q_exact, action_exact, termination_exact)
    exact_gradients = torch.autograd.grad(
        exact, (q_exact, action_exact, termination_exact)
    )

    q_rloo = q_exact.detach().clone().requires_grad_(True)
    action_rloo = action_exact.detach().clone().requires_grad_(True)
    termination_rloo = termination_exact.detach().clone().requires_grad_(True)
    surrogate = enumerated_expected_rloo_surrogate(
        q_rloo, action_rloo, termination_rloo
    )
    rloo_gradients = torch.autograd.grad(
        surrogate, (q_rloo, action_rloo, termination_rloo)
    )

    for exact_gradient, rloo_gradient in zip(exact_gradients, rloo_gradients):
        assert torch.allclose(exact_gradient, rloo_gradient, atol=1e-10, rtol=1e-10)
