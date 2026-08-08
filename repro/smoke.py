"""CPU smokes: one-batch forward/backward and a tiny four-operation overfit.

The Sudoku supervision never sets an insert or delete label (measured: zero
across every generated transition), so a fixture that exercises all four
declared operations must be synthetic.  ``four_operation_fixture`` builds one
deterministically; ``cpu_overfit_smoke`` demonstrates that the model can learn
all four targets on it.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from repro import vocab
from repro.config import ModelSpec, SudokuConfig
from repro.losses import head_diagnostics, supervised_loss
from repro.model import build_model

TINY_SPEC = ModelSpec(
    vocab_size=vocab.VOCAB_SIZE,
    length=64,
    hidden_size=64,
    n_heads=2,
    n_blocks=2,
    cond_dim=32,
    dropout=0.0,
)

HEAD_PARAMS = (
    "output_layer.unmasking_head.weight",
    "output_layer.remasking_head.weight",
    "output_layer.expansion_head.weight",
    "output_layer.contraction_head.weight",
)


def four_operation_fixture(
    batch: int = 8, length: int = 24, seed: int = 42
) -> Dict[str, torch.Tensor]:
    """Deterministic fixture in which every operation target fires somewhere."""
    rng = np.random.default_rng(seed)
    x = rng.integers(vocab.DIGIT_MIN, vocab.DIGIT_MAX + 1, size=(batch, length)).astype(np.int64)

    mask_positions = rng.random((batch, length)) < 0.30
    mask_positions[:, 0] = True  # guarantee at least one unmask target per row
    y = x.copy()
    x = np.where(mask_positions, vocab.MASK, x)

    r = (rng.random((batch, length)) < 0.20).astype(np.int64)
    e = (rng.random((batch, length)) < 0.15).astype(np.int64)
    c = ((rng.random((batch, length)) < 0.30) & mask_positions).astype(np.int64)
    r[:, 1] = 1
    e[:, 2] = 1
    c[:, 0] = 1  # position 0 is always a MASK, so delete is applicable there

    return {
        "x_k": torch.as_tensor(x),
        "y_star": torch.as_tensor(y),
        "r_star": torch.as_tensor(r),
        "e_star": torch.as_tensor(e),
        "c_star": torch.as_tensor(c),
        "attention_mask": torch.ones((batch, length), dtype=torch.float32),
    }


def _grad_stats(model) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    named = dict(model.named_parameters())
    for name in HEAD_PARAMS:
        grad = named[name].grad
        if grad is None:
            stats[name] = {"present": 0.0, "norm": 0.0, "finite": 0.0}
            continue
        norm = float(grad.detach().norm())
        stats[name] = {
            "present": 1.0,
            "norm": norm,
            "finite": float(bool(torch.isfinite(grad).all())),
            "nonzero": float(norm > 0.0),
        }
    return stats


def one_batch_smoke(
    config: SudokuConfig,
    device: str = "cpu",
    batch_size: int = 4,
    use_real_transitions: bool = True,
) -> dict:
    """One forward/backward/optimizer step at the paper-faithful architecture."""
    torch_device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = build_model(config.model, time_conditioning=config.time_conditioning, seed=config.seed)
    model.to(torch_device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optim.lr,
        betas=(config.optim.beta1, config.optim.beta2),
        eps=config.optim.eps,
        weight_decay=config.optim.weight_decay,
    )

    source = "synthetic"
    batch: Dict[str, torch.Tensor]
    if use_real_transitions:
        try:
            from repro.paths import transitions_dir
            from repro.trajectories import TransitionDataset

            dataset = TransitionDataset(transitions_dir() / "sudoku-100")
            raw = dataset.batch(np.arange(batch_size))
            batch = {
                key: torch.as_tensor(raw[key], dtype=torch.long)
                for key in ("x_k", "y_star", "r_star", "e_star", "c_star")
            }
            batch["attention_mask"] = torch.ones_like(batch["x_k"], dtype=torch.float32)
            source = "generated Sudoku transitions"
        except Exception:
            batch = four_operation_fixture(batch=batch_size, length=vocab.SEQUENCE_LENGTH)
    else:
        batch = four_operation_fixture(batch=batch_size, length=vocab.SEQUENCE_LENGTH)
    batch = {k: v.to(torch_device) for k, v in batch.items()}

    before = {name: p.detach().clone() for name, p in model.named_parameters() if name in HEAD_PARAMS}
    optimizer.zero_grad(set_to_none=True)
    outputs = model(batch["x_k"])
    loss = supervised_loss(
        outputs,
        batch["x_k"],
        batch["y_star"],
        batch["r_star"],
        batch["e_star"],
        batch["c_star"],
        attention_mask=batch["attention_mask"],
        lambda_remask=config.lambda_remask,
        lambda_insert=config.lambda_expand,
        lambda_delete=config.lambda_contract,
    )
    loss.total.backward()
    grads = _grad_stats(model)
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val))
    optimizer.step()
    named = dict(model.named_parameters())
    moved = {
        name: float((named[name].detach() - before[name]).abs().max()) for name in HEAD_PARAMS
    }

    shapes = {
        key: list(value.shape) for key, value in outputs.items()
    }
    return {
        "device": str(torch_device),
        "batch_source": source,
        "batch_shape": list(batch["x_k"].shape),
        "output_shapes": shapes,
        "losses": loss.scalars(),
        "head_diagnostics": head_diagnostics(
            outputs, batch["x_k"], batch["y_star"], batch["r_star"], batch["e_star"], batch["c_star"]
        ),
        "grad_norm": grad_norm,
        "head_gradients": grads,
        "head_parameter_movement": moved,
        "all_heads_received_gradient": all(
            g.get("present") == 1.0 and g.get("finite") == 1.0 and g.get("nonzero") == 1.0
            for g in grads.values()
        ),
        "all_losses_finite": all(
            np.isfinite(v) for k, v in loss.scalars().items() if isinstance(v, float)
        ),
    }


def cpu_overfit_smoke(
    device: str = "cpu",
    steps: int = 400,
    lr: float = 3e-3,
    seed: int = 42,
    spec: Optional[ModelSpec] = None,
) -> dict:
    """Tiny overfit demonstrating that all four operation targets are learnable."""
    torch_device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(seed)
    model = build_model(spec or TINY_SPEC, time_conditioning=False, seed=seed).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    batch = {k: v.to(torch_device) for k, v in four_operation_fixture(seed=seed).items()}

    history = []
    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["x_k"])
        loss = supervised_loss(
            outputs,
            batch["x_k"],
            batch["y_star"],
            batch["r_star"],
            batch["e_star"],
            batch["c_star"],
            attention_mask=batch["attention_mask"],
        )
        loss.total.backward()
        optimizer.step()
        if step % max(1, steps // 8) == 0:
            history.append({"step": step, **loss.scalars()})

    model.eval()
    with torch.no_grad():
        outputs = model(batch["x_k"])
        diagnostics = head_diagnostics(
            outputs, batch["x_k"], batch["y_star"], batch["r_star"], batch["e_star"], batch["c_star"]
        )
        unmask_positions = (batch["x_k"] == vocab.MASK) & (batch["y_star"] != vocab.MASK)
        from repro.losses import subs_parameterization

        pred = subs_parameterization(outputs["unmasking_logits"], batch["x_k"]).argmax(dim=-1)
        unmask_acc = float((pred[unmask_positions] == batch["y_star"][unmask_positions]).float().mean())
        head_acc = {}
        for key, logits_key, labels in (
            ("remask", "remasking_logits", batch["r_star"]),
            ("insert", "expansion_logits", batch["e_star"]),
            ("delete", "contraction_logits", batch["c_star"]),
        ):
            fired = torch.sigmoid(outputs[logits_key].squeeze(-1).float()) > 0.5
            head_acc[key] = float((fired == labels.bool()).float().mean())

    targets_present = {
        "unmask": int(unmask_positions.sum()),
        "remask": int(batch["r_star"].sum()),
        "insert": int(batch["e_star"].sum()),
        "delete": int(batch["c_star"].sum()),
    }
    passed = (
        unmask_acc >= 0.99
        and all(value >= 0.99 for value in head_acc.values())
        and all(value > 0 for value in targets_present.values())
    )
    return {
        "device": str(torch_device),
        "steps": steps,
        "targets_present": targets_present,
        "final_unmask_accuracy": unmask_acc,
        "final_head_accuracy": head_acc,
        "history": history,
        "diagnostics": diagnostics,
        "passed": bool(passed),
    }


def sequence_roundtrip_fixture() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Small hand-checked ``g`` fixture used by the transition tests."""
    M = vocab.MASK
    x = np.array([1, M, 3, M, 5], dtype=np.int64)
    y = np.array([9, 2, 9, 4, 9], dtype=np.int64)
    r = np.array([1, 0, 0, 0, 0], dtype=np.int64)
    e = np.array([0, 0, 1, 0, 0], dtype=np.int64)
    d = np.array([0, 0, 0, 1, 0], dtype=np.int64)
    # position 0: remask -> M
    # position 1: MASK, unmask -> 2
    # position 2: keep 3, then insert -> M
    # position 3: MASK and delete -> dropped
    # position 4: keep 5
    expected = np.array([M, 2, 3, M, 5], dtype=np.int64)
    return x, y, r, e, d, expected
