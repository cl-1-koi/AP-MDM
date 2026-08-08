"""Trainer determinism/resumability, telemetry, smoke gates and supervisor tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from repro import data as data_mod
from repro import vocab
from repro.config import ModelSpec
from repro.smoke import cpu_overfit_smoke, four_operation_fixture, one_batch_smoke
from repro.supervisor import (
    FORBIDDEN_ENV_MARKERS,
    SMOKE_CEILING_SECONDS,
    GpuState,
    SupervisorError,
    assert_allowed_host,
    assert_gpus_idle,
    assert_no_paid_capacity,
)
from repro.telemetry import TelemetryWriter, read_telemetry
from repro.trainer import (
    EpochPermutationSampler,
    Trainer,
    TrainerSettings,
    constant_schedule_with_warmup,
    seed_everything,
)
from repro.trajectories import TransitionDataset, build_transition_store, deterministic_split


# --------------------------------------------------------------------------
# Deterministic sampling / schedule
# --------------------------------------------------------------------------


def test_batch_composition_is_a_pure_function_of_seed_and_step():
    idx = np.arange(1000)
    a = EpochPermutationSampler(idx, 32, seed=42)
    b = EpochPermutationSampler(idx, 32, seed=42)
    for step in (1, 2, 31, 32, 33, 200):
        assert np.array_equal(a.batch_for_step(step), b.batch_for_step(step))
    c = EpochPermutationSampler(idx, 32, seed=43)
    assert not np.array_equal(a.batch_for_step(1), c.batch_for_step(1))


def test_batches_within_an_epoch_are_disjoint_and_cover_the_split():
    idx = np.arange(256)
    sampler = EpochPermutationSampler(idx, 32, seed=42)
    seen = np.concatenate([sampler.batch_for_step(s) for s in range(1, 9)])
    assert sorted(seen.tolist()) == list(range(256))


def test_sampler_rejects_a_split_smaller_than_one_batch():
    with pytest.raises(ValueError):
        EpochPermutationSampler(np.arange(4), 32, seed=42)


def test_constant_schedule_with_warmup_matches_the_declared_shape():
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([param], lr=1e-4)
    scheduler = constant_schedule_with_warmup(optimizer, 250)
    seen = []
    for _ in range(300):
        seen.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
    assert seen[0] == 0.0
    assert abs(seen[125] - 0.5e-4) < 1e-9
    assert abs(seen[250] - 1e-4) < 1e-12
    assert abs(seen[299] - 1e-4) < 1e-12


def test_seed_everything_reports_every_rng():
    seeds = seed_everything(42)
    assert set(seeds) >= {"seed", "python_random", "numpy", "torch", "torch_cuda", "dataloader_base", "sampling"}
    assert seeds["seed"] == 42


# --------------------------------------------------------------------------
# Telemetry
# --------------------------------------------------------------------------


def test_telemetry_is_append_only_and_readable(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    with TelemetryWriter(path, run_id="r") as writer:
        writer.write("train", step=1, loss=2.0)
        writer.write("train", step=2, loss=1.0)
        writer.write("checkpoint", step=2)
    records = list(read_telemetry(path))
    assert len(records) == 3
    assert [r["kind"] for r in records] == ["train", "train", "checkpoint"]
    assert [r["step"] for r in read_telemetry(path, kind="train")] == [1, 2]
    with TelemetryWriter(path, run_id="r") as writer:
        writer.write("resume", step=2)
    assert len(list(read_telemetry(path))) == 4


# --------------------------------------------------------------------------
# End-to-end CPU training smoke
# --------------------------------------------------------------------------


TINY_TRAIN_SPEC = ModelSpec(
    vocab_size=vocab.VOCAB_SIZE, length=400, hidden_size=32, n_heads=2, n_blocks=2,
    cond_dim=16, dropout=0.0,
)


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory, train_split, paper_config):
    """A real (tiny) training run over genuine Sudoku transitions."""
    import dataclasses

    root = tmp_path_factory.mktemp("run")
    store = root / "transitions"
    puzzles = np.asarray(train_split.puzzles[:1])
    build_transition_store(
        puzzles, out_dir=store, tag="one", validate=False,
        initial_states=data_mod.encode_puzzle_states(puzzles),
    )
    dataset = TransitionDataset(store)
    cfg = dataclasses.replace(paper_config, model=TINY_TRAIN_SPEC, global_batch_size=8, warmup_steps=2)
    train_idx, monitor_idx = deterministic_split(len(dataset), 0.99, cfg.seed)
    return root, cfg, dataset, train_idx, monitor_idx


def _make_trainer(run_dir, cfg, dataset, train_idx, monitor_idx, **overrides):
    settings = TrainerSettings(
        max_steps=overrides.pop("max_steps", 6), checkpoint_every=overrides.pop("checkpoint_every", 3),
        log_every=1, monitor_every=0, monitor_batches=1, device="cpu", **overrides,
    )
    return Trainer(
        config=cfg, dataset=dataset, train_indices=train_idx, monitor_indices=monitor_idx,
        run_dir=run_dir, settings=settings, manifest_sha256="0" * 64, run_id="tiny",
    )


def test_training_runs_logs_and_checkpoints(tiny_run):
    root, cfg, dataset, train_idx, monitor_idx = tiny_run
    run_dir = root / "run_a"
    trainer = _make_trainer(run_dir, cfg, dataset, train_idx, monitor_idx)
    try:
        summary = trainer.train()
    finally:
        trainer.close()
    assert summary["end_step"] == 6
    assert (run_dir / "checkpoints" / "latest.pt").exists()
    records = list(read_telemetry(run_dir / "telemetry.jsonl", kind="train"))
    assert len(records) == 6
    for record in records:
        for key in ("loss", "loss_unmask", "loss_remask", "loss_insert", "loss_delete", "grad_norm"):
            assert np.isfinite(record[key])
        assert "remask_pred_positive_rate" in record
        assert "samples_per_second" in record


def test_resume_continues_from_the_checkpointed_step(tiny_run):
    root, cfg, dataset, train_idx, monitor_idx = tiny_run
    run_dir = root / "run_b"
    first = _make_trainer(run_dir, cfg, dataset, train_idx, monitor_idx, max_steps=4)
    try:
        first.train()
        weights_before = {k: v.clone() for k, v in first.model.state_dict().items()}
    finally:
        first.close()

    second = _make_trainer(run_dir, cfg, dataset, train_idx, monitor_idx, max_steps=8)
    try:
        assert second.maybe_resume() is True
        assert second.step == 4
        for key, value in second.model.state_dict().items():
            assert torch.allclose(value, weights_before[key])
        summary = second.train()
    finally:
        second.close()
    assert summary["start_step"] == 4
    assert summary["end_step"] == 8
    assert any(r["kind"] == "resume" for r in read_telemetry(run_dir / "telemetry.jsonl"))


def test_resume_rejects_a_foreign_manifest(tiny_run):
    root, cfg, dataset, train_idx, monitor_idx = tiny_run
    run_dir = root / "run_c"
    trainer = _make_trainer(run_dir, cfg, dataset, train_idx, monitor_idx, max_steps=2)
    try:
        trainer.train()
    finally:
        trainer.close()

    settings = TrainerSettings(max_steps=2, checkpoint_every=2, log_every=1, monitor_every=0, device="cpu")
    other = Trainer(
        config=cfg, dataset=dataset, train_indices=train_idx, monitor_indices=monitor_idx,
        run_dir=run_dir, settings=settings, manifest_sha256="1" * 64, run_id="tiny",
    )
    try:
        with pytest.raises(RuntimeError):
            other.maybe_resume()
    finally:
        other.close()


def test_checkpoint_binding_is_written(tiny_run):
    root, cfg, dataset, train_idx, monitor_idx = tiny_run
    run_dir = root / "run_d"
    trainer = _make_trainer(run_dir, cfg, dataset, train_idx, monitor_idx, max_steps=2)
    try:
        trainer.train()
    finally:
        trainer.close()
    payload = torch.load(run_dir / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
    binding = payload["binding"]
    assert binding["architecture_signature"] == cfg.model.signature()
    assert binding["vocabulary_size"] == vocab.VOCAB_SIZE
    assert binding["manifest_sha256"] == "0" * 64
    assert binding["step"] == 2
    assert binding["config_sha256"] == cfg.sha256


def test_training_loss_decreases_on_a_single_repeated_batch(tiny_run):
    """A minimal optimisation sanity check, not a scientific claim."""
    root, cfg, dataset, train_idx, monitor_idx = tiny_run
    run_dir = root / "run_e"
    trainer = _make_trainer(run_dir, cfg, dataset, train_idx, monitor_idx, max_steps=1)
    try:
        batch_idx = trainer.sampler.batch_for_step(1)
        from repro.trainer import to_torch_batch

        batch = to_torch_batch(dataset.batch(batch_idx), trainer.device)
        first = float(trainer._forward_backward(batch)[0].total)
        for _ in range(40):
            last = float(trainer._forward_backward(batch)[0].total)
    finally:
        trainer.close()
    assert last < first


# --------------------------------------------------------------------------
# Smoke gates
# --------------------------------------------------------------------------


def test_one_batch_smoke_gives_every_head_gradient(paper_config):
    result = one_batch_smoke(paper_config, device="cpu", batch_size=2, use_real_transitions=False)
    assert result["all_heads_received_gradient"]
    assert result["all_losses_finite"]
    assert result["output_shapes"]["unmasking_logits"] == [2, vocab.SEQUENCE_LENGTH, 31]
    assert all(v > 0 for v in result["head_parameter_movement"].values())


def test_tiny_overfit_learns_all_four_operations():
    result = cpu_overfit_smoke(device="cpu", steps=400)
    assert all(v > 0 for v in result["targets_present"].values())
    assert result["final_unmask_accuracy"] >= 0.99
    for key, value in result["final_head_accuracy"].items():
        assert value >= 0.99, (key, value)
    assert result["passed"]


def test_fixture_exercises_all_four_operations():
    batch = four_operation_fixture()
    assert int(batch["r_star"].sum()) > 0
    assert int(batch["e_star"].sum()) > 0
    assert int(batch["c_star"].sum()) > 0
    assert int(((batch["x_k"] == vocab.MASK) & (batch["y_star"] != vocab.MASK)).sum()) > 0


# --------------------------------------------------------------------------
# Supervisor (fail-closed)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("marker", ["RUNPOD_POD_ID", "RUNPOD_API_KEY", "VAST_CONTAINERLABEL"])
def test_paid_capacity_markers_abort(monkeypatch, marker):
    assert marker in FORBIDDEN_ENV_MARKERS
    monkeypatch.setenv(marker, "1")
    with pytest.raises(SupervisorError, match="paid-capacity"):
        assert_no_paid_capacity()


def test_paid_capacity_hostname_aborts(monkeypatch):
    monkeypatch.setattr("platform.node", lambda: "runpod-abc123")
    with pytest.raises(SupervisorError):
        assert_no_paid_capacity()


def test_only_a10_gpus_or_the_allowed_host_are_permitted(monkeypatch):
    monkeypatch.setattr("platform.node", lambda: "some-random-box")
    a10 = GpuState(0, "NVIDIA A10", 23028, 0, 0)
    h100 = GpuState(0, "NVIDIA H100 80GB HBM3", 81559, 0, 0)
    assert assert_allowed_host([a10]) == "some-random-box"
    with pytest.raises(SupervisorError):
        assert_allowed_host([h100])
    with pytest.raises(SupervisorError):
        assert_allowed_host([])
    monkeypatch.setattr("platform.node", lambda: "a10-220")
    assert assert_allowed_host([h100]) == "a10-220"


def test_busy_gpus_are_refused():
    idle = GpuState(0, "NVIDIA A10", 23028, 10, 0)
    assert_gpus_idle([idle])
    with pytest.raises(SupervisorError):
        assert_gpus_idle([GpuState(0, "NVIDIA A10", 23028, 10, 0, ["pid=1 mem=100MiB"])])
    with pytest.raises(SupervisorError):
        assert_gpus_idle([GpuState(0, "NVIDIA A10", 23028, 8000, 0)])
    with pytest.raises(SupervisorError):
        assert_gpus_idle([GpuState(0, "NVIDIA A10", 23028, 10, 80)])
    with pytest.raises(SupervisorError):
        assert_gpus_idle([], None)


def test_smoke_ceiling_is_fifteen_minutes():
    assert SMOKE_CEILING_SECONDS == 15 * 60
