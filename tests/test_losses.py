"""Supervised objective, per-head gradient and diagnostics tests."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from repro import vocab
from repro.config import ModelSpec
from repro.losses import head_diagnostics, subs_parameterization, supervised_loss
from repro.model import build_model
from repro.smoke import HEAD_PARAMS, four_operation_fixture

SPEC = ModelSpec(
    vocab_size=vocab.VOCAB_SIZE, length=400, hidden_size=32, n_heads=2, n_blocks=2,
    cond_dim=16, dropout=0.0,
)


def test_subs_parameterization_never_predicts_mask_and_pins_unmasked():
    logits = torch.randn(2, 5, vocab.VOCAB_SIZE)
    x = torch.tensor([[vocab.MASK, 3, vocab.MASK, 4, 5], [1, 2, 3, vocab.MASK, 5]])
    out = subs_parameterization(logits.clone(), x)
    assert float(out[:, :, vocab.MASK].max()) < -1000.0
    for b in range(2):
        for i in range(5):
            if x[b, i] != vocab.MASK:
                assert float(out[b, i, x[b, i]]) == 0.0
                others = [j for j in range(vocab.VOCAB_SIZE) if j != int(x[b, i])]
                assert float(out[b, i, others].max()) < -1000.0
            else:
                total = torch.logsumexp(out[b, i], dim=-1)
                assert math.isclose(float(total), 0.0, abs_tol=1e-4)


def test_subs_parameterization_does_not_mutate_its_input():
    logits = torch.randn(1, 3, vocab.VOCAB_SIZE)
    before = logits.clone()
    subs_parameterization(logits, torch.tensor([[vocab.MASK, 1, 2]]))
    assert torch.equal(logits, before)


def _outputs_for(batch, seed=0):
    model = build_model(SPEC, seed=seed)
    return model, model(batch["x_k"])


def test_every_head_receives_a_finite_nonzero_gradient_on_a_four_operation_fixture():
    batch = four_operation_fixture(batch=4, length=24)
    model, outputs = _outputs_for(batch)
    loss = supervised_loss(
        outputs, batch["x_k"], batch["y_star"], batch["r_star"], batch["e_star"], batch["c_star"],
        attention_mask=batch["attention_mask"],
    )
    loss.total.backward()
    named = dict(model.named_parameters())
    for name in HEAD_PARAMS:
        grad = named[name].grad
        assert grad is not None, name
        assert torch.isfinite(grad).all(), name
        assert float(grad.norm()) > 0.0, name


def test_each_component_alone_produces_gradient_for_its_own_head():
    batch = four_operation_fixture(batch=4, length=24)
    for component, head in (
        ("unmask", "output_layer.unmasking_head.weight"),
        ("remask", "output_layer.remasking_head.weight"),
        ("insert", "output_layer.expansion_head.weight"),
        ("delete", "output_layer.contraction_head.weight"),
    ):
        model, outputs = _outputs_for(batch, seed=1)
        loss = supervised_loss(
            outputs, batch["x_k"], batch["y_star"], batch["r_star"], batch["e_star"],
            batch["c_star"], attention_mask=batch["attention_mask"],
        )
        getattr(loss, component).backward()
        named = dict(model.named_parameters())
        grad = named[head].grad
        assert grad is not None and float(grad.norm()) > 0.0, component


def test_unmask_loss_only_counts_masked_positions_with_non_mask_targets():
    batch = four_operation_fixture(batch=2, length=12)
    model, outputs = _outputs_for(batch)
    loss = supervised_loss(
        outputs, batch["x_k"], batch["y_star"], batch["r_star"], batch["e_star"], batch["c_star"],
        attention_mask=batch["attention_mask"],
    )
    expected = int(
        ((batch["x_k"] == vocab.MASK) & (batch["y_star"] != vocab.MASK)).sum()
    )
    assert loss.n_unmask_positions == expected


def test_pure_remask_transition_yields_zero_unmask_loss_but_keeps_the_graph():
    x = torch.full((1, 8), 5, dtype=torch.long)
    batch = {
        "x_k": x,
        "y_star": x.clone(),
        "r_star": torch.ones_like(x),
        "e_star": torch.zeros_like(x),
        "c_star": torch.zeros_like(x),
        "attention_mask": torch.ones_like(x, dtype=torch.float32),
    }
    model, outputs = _outputs_for(batch)
    loss = supervised_loss(
        outputs, batch["x_k"], batch["y_star"], batch["r_star"], batch["e_star"], batch["c_star"],
        attention_mask=batch["attention_mask"],
    )
    assert loss.n_unmask_positions == 0
    assert float(loss.unmask) == 0.0
    assert loss.total.requires_grad
    loss.total.backward()


def test_loss_weights_are_applied():
    batch = four_operation_fixture(batch=2, length=12)
    model, outputs = _outputs_for(batch)
    base = supervised_loss(
        outputs, batch["x_k"], batch["y_star"], batch["r_star"], batch["e_star"], batch["c_star"],
        attention_mask=batch["attention_mask"],
    )
    doubled = supervised_loss(
        outputs, batch["x_k"], batch["y_star"], batch["r_star"], batch["e_star"], batch["c_star"],
        attention_mask=batch["attention_mask"], lambda_remask=2.0,
    )
    assert math.isclose(
        float(doubled.total - base.total), float(base.remask), rel_tol=1e-5, abs_tol=1e-7
    )


def test_attention_mask_excludes_invalid_positions():
    batch = four_operation_fixture(batch=2, length=12)
    mask = batch["attention_mask"].clone()
    mask[:, 6:] = 0.0
    model, outputs = _outputs_for(batch)
    masked = supervised_loss(
        outputs, batch["x_k"], batch["y_star"], batch["r_star"], batch["e_star"], batch["c_star"],
        attention_mask=mask,
    )
    assert masked.n_valid_positions == 12
    assert masked.n_unmask_positions <= int(
        ((batch["x_k"][:, :6] == vocab.MASK) & (batch["y_star"][:, :6] != vocab.MASK)).sum()
    )


def test_missing_head_logits_are_rejected():
    batch = four_operation_fixture(batch=1, length=8)
    model, outputs = _outputs_for(batch)
    del outputs["contraction_logits"]
    with pytest.raises(ValueError):
        supervised_loss(
            outputs, batch["x_k"], batch["y_star"], batch["r_star"], batch["e_star"],
            batch["c_star"], attention_mask=batch["attention_mask"],
        )


def test_head_diagnostics_expose_collapse_and_non_emission():
    batch = four_operation_fixture(batch=4, length=24)
    model, outputs = _outputs_for(batch)
    stats = head_diagnostics(
        outputs, batch["x_k"], batch["y_star"], batch["r_star"], batch["e_star"], batch["c_star"]
    )
    for key in ("remask", "insert", "delete"):
        assert f"{key}_pred_positive_rate" in stats
        assert f"{key}_target_positive_rate" in stats
        assert f"{key}_recall" in stats
    # Heads are initialised with bias -2.0, i.e. deliberately non-emitting.
    assert stats["remask_pred_positive_rate"] == 0.0
    assert stats["remask_target_positive_rate"] > 0.0


def test_losses_are_finite_on_real_sudoku_transitions(puzzle_transitions):
    idx = np.arange(4)
    batch = {
        "x_k": torch.as_tensor(puzzle_transitions.x_k[idx].astype(np.int64)),
        "y_star": torch.as_tensor(puzzle_transitions.y_star[idx].astype(np.int64)),
        "r_star": torch.as_tensor(puzzle_transitions.r_star[idx].astype(np.int64)),
        "e_star": torch.as_tensor(puzzle_transitions.e_star[idx].astype(np.int64)),
        "c_star": torch.as_tensor(puzzle_transitions.c_star[idx].astype(np.int64)),
    }
    batch["attention_mask"] = torch.ones_like(batch["x_k"], dtype=torch.float32)
    model, outputs = _outputs_for(batch)
    loss = supervised_loss(
        outputs, batch["x_k"], batch["y_star"], batch["r_star"], batch["e_star"], batch["c_star"],
        attention_mask=batch["attention_mask"],
    )
    for value in loss.scalars().values():
        assert np.isfinite(value)
    loss.total.backward()


def test_subs_argmax_matches_the_full_subs_parameterization():
    from repro.losses import subs_argmax

    torch.manual_seed(0)
    for _ in range(20):
        logits = torch.randn(3, 7, vocab.VOCAB_SIZE)
        x = torch.randint(0, vocab.VOCAB_SIZE, (3, 7))
        reference = subs_parameterization(logits.clone(), x).argmax(dim=-1)
        assert torch.equal(subs_argmax(logits.clone(), x), reference)
