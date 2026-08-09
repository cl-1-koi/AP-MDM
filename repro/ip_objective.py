"""Auditable finite-state primitives for the paper-native Insertion Process.

The production model will use Transformer logits, but its permutation ELBO and
RLOO estimator must first agree with exact enumeration on short sequences.
Indices and insertion slots in this module are zero-based.
"""

from __future__ import annotations

import bisect
import itertools
from collections.abc import Iterable, Sequence

import torch


def _checked_permutation(permutation: Sequence[int]) -> tuple[int, ...]:
    value = tuple(int(index) for index in permutation)
    if sorted(value) != list(range(len(value))):
        raise ValueError(f"not a permutation of 0..{len(value) - 1}: {value}")
    return value


def permutation_to_slots(permutation: Sequence[int]) -> tuple[int, ...]:
    """Map final-position insertion order to trajectory-relative slots.

    ``permutation[i]`` is the final sequence index inserted at step ``i``.
    The returned slot is its rank among indices inserted up to that step.
    """
    permutation = _checked_permutation(permutation)
    inserted: list[int] = []
    slots = []
    for final_index in permutation:
        slot = bisect.bisect_left(inserted, final_index)
        slots.append(slot)
        inserted.insert(slot, final_index)
    return tuple(slots)


def slots_to_permutation(slots: Sequence[int]) -> tuple[int, ...]:
    """Invert :func:`permutation_to_slots` for a complete trajectory."""
    final_order: list[int] = []
    for step, raw_slot in enumerate(slots):
        slot = int(raw_slot)
        if not 0 <= slot <= step:
            raise ValueError(f"slot {slot} is invalid at insertion step {step}")
        final_order.insert(slot, step)
    permutation = [0] * len(final_order)
    for final_index, insertion_step in enumerate(final_order):
        permutation[insertion_step] = final_index
    return tuple(permutation)


def all_permutations(length: int) -> Iterable[tuple[int, ...]]:
    if length < 0:
        raise ValueError("length must be non-negative")
    return itertools.permutations(range(length))


def prefix_mask_index(prefix: Sequence[int]) -> int:
    """Encode the set of already inserted final indices as a bit mask."""
    value = 0
    for raw_index in prefix:
        index = int(raw_index)
        if index < 0:
            raise ValueError("prefix indices must be non-negative")
        bit = 1 << index
        if value & bit:
            raise ValueError(f"duplicate prefix index {index}")
        value |= bit
    return value


def pl_prefix_log_probability(
    logits: torch.Tensor, prefix: Sequence[int]
) -> torch.Tensor:
    """Log probability of a prefix under a Plackett--Luce distribution."""
    if logits.ndim != 1:
        raise ValueError(f"expected one-dimensional logits, got {tuple(logits.shape)}")
    remaining = torch.ones(logits.numel(), dtype=torch.bool, device=logits.device)
    result = logits.new_zeros(())
    for raw_index in prefix:
        index = int(raw_index)
        if not 0 <= index < logits.numel() or not bool(remaining[index]):
            raise ValueError(f"invalid or repeated prefix index {index}")
        # Boolean masks are saved by autograd's indexing backward; clone before
        # mutating ``remaining`` for the next Plackett--Luce factor.
        result = result + logits[index] - torch.logsumexp(logits[remaining.clone()], dim=0)
        remaining[index] = False
    return result


def pl_log_probability(logits: torch.Tensor, permutation: Sequence[int]) -> torch.Tensor:
    permutation = _checked_permutation(permutation)
    if len(permutation) != logits.numel():
        raise ValueError("permutation length must match logits")
    return pl_prefix_log_probability(logits, permutation)


def exact_next_value(
    q_logits: torch.Tensor,
    prefix: Sequence[int],
    action_log_values: torch.Tensor,
) -> torch.Tensor:
    """Closed-form next-index expectation in Equation 8 of the paper.

    ``action_log_values[mask, j]`` is the differentiable log probability
    assigned by the generative model to inserting target index ``j`` in its
    induced relative slot after the prefix represented by ``mask``.
    """
    length = q_logits.numel()
    if action_log_values.shape != (1 << length, length):
        raise ValueError(
            f"expected action table {(1 << length, length)}, got "
            f"{tuple(action_log_values.shape)}"
        )
    mask_index = prefix_mask_index(prefix)
    remaining = torch.ones(length, dtype=torch.bool, device=q_logits.device)
    for raw_index in prefix:
        index = int(raw_index)
        if not 0 <= index < length or not bool(remaining[index]):
            raise ValueError(f"invalid or repeated prefix index {index}")
        remaining[index] = False
    if not bool(remaining.any()):
        raise ValueError("exact next expectation requires a non-complete prefix")
    log_q = torch.log_softmax(q_logits.masked_fill(~remaining, -torch.inf), dim=0)
    q = log_q.exp()
    bracket = action_log_values[mask_index] - log_q
    return (q[remaining] * bracket[remaining]).sum()


def exact_permutation_elbo(
    q_logits: torch.Tensor,
    action_log_values: torch.Tensor,
    termination_log_probability: torch.Tensor,
) -> torch.Tensor:
    """Enumerate the complete permutation ELBO for one target sequence."""
    length = q_logits.numel()
    result = termination_log_probability
    for permutation in all_permutations(length):
        log_q = pl_log_probability(q_logits, permutation)
        log_p = q_logits.new_zeros(())
        prefix: tuple[int, ...] = ()
        for next_index in permutation:
            log_p = log_p + action_log_values[prefix_mask_index(prefix), next_index]
            prefix = prefix + (next_index,)
        result = result + log_q.exp() * (log_p - log_q)
    return result


def rloo_surrogate_for_pair(
    q_logits: torch.Tensor,
    action_log_values: torch.Tensor,
    termination_log_probability: torch.Tensor,
    permutation_1: Sequence[int],
    permutation_2: Sequence[int],
    prefix_length: int,
) -> torch.Tensor:
    """Two-sample RLOO autodiff surrogate for one sampled time index."""
    permutation_1 = _checked_permutation(permutation_1)
    permutation_2 = _checked_permutation(permutation_2)
    length = q_logits.numel()
    if len(permutation_1) != length or len(permutation_2) != length:
        raise ValueError("permutation length must match logits")
    if not 0 <= prefix_length <= length:
        raise ValueError("prefix_length must lie in 0..length")

    prefix_1 = permutation_1[:prefix_length]
    prefix_2 = permutation_2[:prefix_length]
    log_q_1 = pl_prefix_log_probability(q_logits, prefix_1)
    log_q_2 = pl_prefix_log_probability(q_logits, prefix_2)
    if prefix_length == length:
        f_1 = termination_log_probability
        f_2 = termination_log_probability
    else:
        f_1 = exact_next_value(q_logits, prefix_1, action_log_values)
        f_2 = exact_next_value(q_logits, prefix_2, action_log_values)
    score_term = (log_q_1 - log_q_2) * (f_1 - f_2).detach()
    return 0.5 * (length + 1) * (score_term + f_1 + f_2)


def enumerated_expected_rloo_surrogate(
    q_logits: torch.Tensor,
    action_log_values: torch.Tensor,
    termination_log_probability: torch.Tensor,
) -> torch.Tensor:
    """Expectation of the stochastic surrogate, for gradient auditing only.

    Sampling weights are detached because the RLOO score term is responsible
    for the gradient through the sampled prefixes.
    """
    length = q_logits.numel()
    permutations = list(all_permutations(length))
    probabilities = [pl_log_probability(q_logits, p).exp().detach() for p in permutations]
    result = q_logits.new_zeros(())
    for permutation_1, probability_1 in zip(permutations, probabilities):
        for permutation_2, probability_2 in zip(permutations, probabilities):
            pair_weight = probability_1 * probability_2 / (length + 1)
            for prefix_length in range(length + 1):
                result = result + pair_weight * rloo_surrogate_for_pair(
                    q_logits,
                    action_log_values,
                    termination_log_probability,
                    permutation_1,
                    permutation_2,
                    prefix_length,
                )
    return result
