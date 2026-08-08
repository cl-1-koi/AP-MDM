"""Sampler tests: Algorithm-1 semantics, termination, ceilings, anomalies."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from repro import data as data_mod
from repro import vocab
from repro.sampler import (
    TERMINATED_EMPTY,
    TERMINATED_FIXED_POINT,
    TERMINATED_LENGTH_OVERFLOW,
    TERMINATED_MAX_STEPS,
    SamplerConfig,
    generate,
)

BIG = 20.0


class ScriptedModel(nn.Module):
    """A model whose four heads are driven by a caller-supplied decision rule."""

    def __init__(self, rule, max_length: int = 400, vocab_size: int = vocab.VOCAB_SIZE):
        super().__init__()
        self.rule = rule
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.calls = 0
        self._anchor = nn.Parameter(torch.zeros(1))

    def forward(self, x, sigma=None):
        self.calls += 1
        y, r, e, d = self.rule(x, self.calls)
        batch, seq = x.shape
        unmask = torch.full((batch, seq, self.vocab_size), -BIG)
        unmask.scatter_(2, y[:, :, None], BIG)
        return {
            "unmasking_logits": unmask,
            "remasking_logits": torch.where(r, BIG, -BIG).float()[:, :, None],
            "expansion_logits": torch.where(e, BIG, -BIG).float()[:, :, None],
            "contraction_logits": torch.where(d, BIG, -BIG).float()[:, :, None],
        }


def _identity_rule(x, call):
    zeros = torch.zeros_like(x, dtype=torch.bool)
    return x.clone(), zeros, zeros, zeros


def test_immediate_fixed_point_terminates_after_one_step():
    model = ScriptedModel(_identity_rule)
    state = np.zeros((2, 8), dtype=np.int64)
    result = generate(model, state, SamplerConfig(max_steps=100))
    for trace in result.traces:
        assert trace.status == TERMINATED_FIXED_POINT
        assert trace.steps == 1
        assert trace.forward_passes == 1
    assert np.array_equal(result.states[0], state[0].astype(np.uint8))


def test_unmasking_then_fixed_point():
    target = 7

    def rule(x, call):
        zeros = torch.zeros_like(x, dtype=torch.bool)
        return torch.full_like(x, target), zeros, zeros, zeros

    model = ScriptedModel(rule)
    state = np.full((1, 6), vocab.MASK, dtype=np.int64)
    result = generate(model, state, SamplerConfig(max_steps=10))
    assert (result.states[0] == target).all()
    assert result.traces[0].status == TERMINATED_FIXED_POINT
    assert result.traces[0].steps == 2  # one step to unmask, one to detect the fixed point
    assert result.traces[0].n_unmask == 6


def test_remask_takes_priority_over_unmask():
    def rule(x, call):
        zeros = torch.zeros_like(x, dtype=torch.bool)
        return torch.full_like(x, 5), torch.ones_like(x, dtype=torch.bool), zeros, zeros

    model = ScriptedModel(rule)
    state = np.full((1, 4), vocab.MASK, dtype=np.int64)
    result = generate(model, state, SamplerConfig(max_steps=3))
    assert (result.states[0] == vocab.MASK).all()
    assert result.traces[0].status == TERMINATED_FIXED_POINT
    assert result.traces[0].n_remask >= 4


def test_max_steps_ceiling_is_enforced_and_reported():
    def flipflop(x, call):
        zeros = torch.zeros_like(x, dtype=torch.bool)
        if call % 2 == 1:
            return x.clone(), torch.ones_like(x, dtype=torch.bool), zeros, zeros
        return torch.full_like(x, 3), zeros, zeros, zeros

    model = ScriptedModel(flipflop)
    state = np.full((1, 4), 3, dtype=np.int64)
    result = generate(model, state, SamplerConfig(max_steps=7))
    assert result.traces[0].status == TERMINATED_MAX_STEPS
    assert result.traces[0].steps == 7
    assert result.traces[0].forward_passes == 7


def test_insert_grows_the_sequence_and_is_counted():
    def rule(x, call):
        zeros = torch.zeros_like(x, dtype=torch.bool)
        insert = torch.zeros_like(x, dtype=torch.bool)
        if call == 1:
            insert[:, 0] = True
        return x.clone(), zeros, insert, zeros

    model = ScriptedModel(rule)
    state = np.array([[1, 2, 3]], dtype=np.int64)
    # Stop after the inserting step so the freshly inserted [MASK] is observable;
    # SUBS forbids predicting [MASK], so a later step would necessarily fill it.
    result = generate(model, state, SamplerConfig(max_steps=1))
    assert list(result.states[0]) == [1, vocab.MASK, 2, 3]
    assert result.traces[0].n_insert == 1
    assert result.traces[0].length_changes == 1
    assert result.traces[0].final_length == 4

    later = generate(ScriptedModel(rule), state, SamplerConfig(max_steps=5))
    assert len(later.states[0]) == 4
    assert vocab.MASK not in list(later.states[0])


def test_delete_removes_masks_and_is_counted():
    def rule(x, call):
        zeros = torch.zeros_like(x, dtype=torch.bool)
        return x.clone(), zeros, zeros, torch.ones_like(x, dtype=torch.bool)

    model = ScriptedModel(rule)
    state = np.array([[1, vocab.MASK, 2]], dtype=np.int64)
    result = generate(model, state, SamplerConfig(max_steps=5))
    assert list(result.states[0]) == [1, 2]
    assert result.traces[0].n_delete == 1


def test_length_overflow_is_rejected_rather_than_silently_truncated():
    def rule(x, call):
        zeros = torch.zeros_like(x, dtype=torch.bool)
        return x.clone(), zeros, torch.ones_like(x, dtype=torch.bool), zeros

    model = ScriptedModel(rule, max_length=8)
    state = np.zeros((1, 5), dtype=np.int64)
    result = generate(model, state, SamplerConfig(max_steps=10))
    assert result.traces[0].status == TERMINATED_LENGTH_OVERFLOW


def test_empty_sequence_is_detected():
    def rule(x, call):
        zeros = torch.zeros_like(x, dtype=torch.bool)
        return x.clone(), zeros, zeros, torch.ones_like(x, dtype=torch.bool)

    model = ScriptedModel(rule)
    state = np.full((1, 3), vocab.MASK, dtype=np.int64)
    result = generate(model, state, SamplerConfig(max_steps=10))
    assert result.traces[0].status == TERMINATED_EMPTY


def test_thresholds_are_honoured():
    def rule(x, call):
        zeros = torch.zeros_like(x, dtype=torch.bool)
        return x.clone(), torch.ones_like(x, dtype=torch.bool), zeros, zeros

    model = ScriptedModel(rule)
    state = np.array([[4, 4]], dtype=np.int64)
    # sigmoid(BIG) ~= 1, so a threshold above 1 can never fire.
    result = generate(model, state, SamplerConfig(tau_remask=1.5, max_steps=3))
    assert list(result.states[0]) == [4, 4]
    assert result.traces[0].n_remask == 0


def test_rows_terminate_independently():
    def rule(x, call):
        zeros = torch.zeros_like(x, dtype=torch.bool)
        # Only the row made of 2s keeps changing; rows are addressed by content
        # because finished rows are dropped from the active batch.
        remask = (x == 2) & (call <= 2)
        y = x.clone()
        y[x == vocab.MASK] = 6
        return y, remask, zeros, zeros

    model = ScriptedModel(rule)
    state = np.array([[1, 1], [2, 2]], dtype=np.int64)
    result = generate(model, state, SamplerConfig(max_steps=20))
    assert result.traces[0].steps < result.traces[1].steps
    assert all(t.status == TERMINATED_FIXED_POINT for t in result.traces)


def test_generation_replays_a_real_solver_trajectory(puzzle_transitions, train_split):
    """An oracle model that replays the recorded controls must reach the solution.

    This validates the sampler against the generator end to end: it is a test
    of the inference machinery, never a measurement of model accuracy.
    """
    step_box = {"i": 0}

    def oracle(x, call):
        i = step_box["i"]
        step_box["i"] += 1
        y = torch.as_tensor(puzzle_transitions.y_star[i].astype(np.int64))[None, :]
        r = torch.as_tensor(puzzle_transitions.r_star[i].astype(bool))[None, :]
        e = torch.as_tensor(puzzle_transitions.e_star[i].astype(bool))[None, :]
        d = torch.as_tensor(puzzle_transitions.c_star[i].astype(bool))[None, :]
        return y, r, e, d

    model = ScriptedModel(oracle)
    initial = data_mod.encode_puzzle_state(train_split.puzzles[0])[None, :]
    result = generate(model, initial, SamplerConfig(max_steps=puzzle_transitions.count))
    decoded = data_mod.decode_state(result.states[0])
    assert not decoded.malformed
    assert decoded.complete
    assert np.array_equal(decoded.grid, train_split.solutions[0])


def test_first_complete_step_is_recorded():
    def rule(x, call):
        zeros = torch.zeros_like(x, dtype=torch.bool)
        y = torch.full_like(x, 5)
        return y, zeros, zeros, zeros

    model = ScriptedModel(rule)
    state = np.array(
        [[vocab.MASK, vocab.WHITE, vocab.NORMAL, vocab.SEPARATOR] * 1], dtype=np.int64
    )
    result = generate(model, state, SamplerConfig(max_steps=5))
    assert result.traces[0].first_complete_step == 1
