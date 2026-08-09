"""Trainer/manifest tests for the monotone Sudoku ladder."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from repro import data as data_mod
from repro.insertion import (
    SudokuInsertionPolicy,
    condition_model_spec,
    effective_model_spec,
)
from repro.insertion_trainer import (
    InsertionTrainer,
    InsertionTrainerSettings,
    build_insertion_manifest,
    verify_insertion_manifest,
)
from test_insertion import tiny_spec


def tiny_config(paper_config):
    return replace(
        paper_config,
        model=tiny_spec(),
        global_batch_size=2,
        warmup_steps=2,
    )


def test_insertion_trainer_checkpoint_resume(tmp_path, paper_config, train_split, test_split):
    config = tiny_config(paper_config)
    run_dir = tmp_path / "run"
    first_settings = InsertionTrainerSettings(
        arm="fixed_ar",
        run_id="tiny-fixed",
        max_steps=2,
        batch_size=2,
        checkpoint_every=1,
        log_every=1,
        eval_every=0,
        device="cpu",
        seed=31,
        bf16=False,
    )
    first = InsertionTrainer(
        config, first_settings, train_split, test_split, run_dir, "sealed-tiny"
    )
    try:
        result = first.train()
        assert result["end_step"] == 2
    finally:
        first.close()

    second_settings = replace(first_settings, max_steps=4)
    second = InsertionTrainer(
        config, second_settings, train_split, test_split, run_dir, "sealed-tiny"
    )
    try:
        assert second.maybe_resume() is True
        assert second.step == 2
        result = second.train()
        assert result["end_step"] == 4
    finally:
        second.close()


def test_trainer_evaluates_non_cadence_terminal_step(
    tmp_path, paper_config, train_split, test_split
):
    settings = InsertionTrainerSettings(
        arm="random_insertion",
        run_id="terminal-eval",
        max_steps=3,
        batch_size=2,
        checkpoint_every=2,
        log_every=1,
        eval_every=2,
        eval_limit=1,
        device="cpu",
        seed=7,
        bf16=False,
        condition_mode="transposed_solution_hint",
    )
    run_dir = tmp_path / "terminal-eval"
    trainer = InsertionTrainer(
        tiny_config(paper_config),
        settings,
        train_split,
        test_split,
        run_dir,
        "sealed-terminal",
    )
    try:
        trainer.train()
    finally:
        trainer.close()
    records = [
        json.loads(line) for line in (run_dir / "telemetry.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in records if row["kind"] == "eval"] == [2, 3]


def test_insertion_manifest_seal_detects_mutation(
    tmp_path, paper_config, train_split, test_split
):
    config = tiny_config(paper_config)
    settings = InsertionTrainerSettings(
        arm="fixed_ar",
        run_id="manifest-test",
        max_steps=10,
        batch_size=2,
        device="cpu",
        bf16=False,
    )
    policy = SudokuInsertionPolicy(effective_model_spec(config.model))
    overlap = data_mod.measure_overlap(train_split, test_split)
    manifest = build_insertion_manifest(
        config=config,
        settings=settings,
        train=train_split,
        test=test_split,
        overlap=overlap.as_dict(),
        policy=policy,
        posterior=None,
        run_dir=tmp_path / "manifest-run",
    )
    assert verify_insertion_manifest(manifest) == manifest["seal"]["manifest_sha256"]
    manifest["training"]["max_steps"] = 11
    with pytest.raises(RuntimeError, match="seal mismatch"):
        verify_insertion_manifest(manifest)


def test_manifest_seals_oracle_hint_as_an_explicit_control(
    tmp_path, paper_config, train_split, test_split
):
    config = tiny_config(paper_config)
    settings = InsertionTrainerSettings(
        arm="random_insertion",
        run_id="oracle-control",
        max_steps=10,
        batch_size=2,
        device="cpu",
        bf16=False,
        condition_mode="transposed_solution_hint",
    )
    policy = SudokuInsertionPolicy(effective_model_spec(config.model))
    overlap = data_mod.measure_overlap(train_split, test_split)
    manifest = build_insertion_manifest(
        config=config,
        settings=settings,
        train=train_split,
        test=test_split,
        overlap=overlap.as_dict(),
        policy=policy,
        posterior=None,
        run_dir=tmp_path / "oracle-control",
    )
    boundary = manifest["mechanism_boundary"]
    assert boundary["condition_mode"] == "transposed_solution_hint"
    assert "complete solution" in boundary["oracle_information"]
    assert verify_insertion_manifest(manifest) == manifest["seal"]["manifest_sha256"]


def test_manifest_seals_keyed_shuffle_and_expanded_vocabulary(
    tmp_path, paper_config, train_split, test_split
):
    config = tiny_config(paper_config)
    settings = InsertionTrainerSettings(
        arm="random_insertion",
        run_id="keyed-shuffle-control",
        max_steps=10,
        batch_size=2,
        device="cpu",
        bf16=False,
        condition_mode="keyed_shuffled_solution_hint",
    )
    policy = SudokuInsertionPolicy(
        condition_model_spec(config.model, settings.condition_mode)
    )
    overlap = data_mod.measure_overlap(train_split, test_split)
    manifest = build_insertion_manifest(
        config=config,
        settings=settings,
        train=train_split,
        test=test_split,
        overlap=overlap.as_dict(),
        policy=policy,
        posterior=None,
        run_dir=tmp_path / "keyed-shuffle-control",
    )
    boundary = manifest["mechanism_boundary"]
    assert boundary["condition_mode"] == "keyed_shuffled_solution_hint"
    assert "CELL_KEY_0..CELL_KEY_80" in boundary["oracle_information"]
    assert manifest["architecture"]["policy_spec"]["vocab_size"] == 115
    assert verify_insertion_manifest(manifest) == manifest["seal"]["manifest_sha256"]


def test_insertion_manifest_records_explicit_paid_capacity_authorization(
    tmp_path, paper_config, train_split, test_split, monkeypatch
):
    authorization = "owner_authorized_test_value"
    monkeypatch.setenv("APMDM_INSERTION_PAID_CAPACITY_AUTHORIZATION", authorization)
    config = tiny_config(paper_config)
    settings = InsertionTrainerSettings(
        arm="fixed_ar",
        run_id="paid-capacity-test",
        max_steps=10,
        batch_size=2,
        device="cpu",
        bf16=False,
    )
    policy = SudokuInsertionPolicy(effective_model_spec(config.model))
    overlap = data_mod.measure_overlap(train_split, test_split)
    manifest = build_insertion_manifest(
        config=config,
        settings=settings,
        train=train_split,
        test=test_split,
        overlap=overlap.as_dict(),
        policy=policy,
        posterior=None,
        run_dir=tmp_path / "paid-capacity-run",
    )
    assert manifest["compute"]["paid_capacity"] == authorization
    assert "RunPod A40 R0" in manifest["compute"]["initial_execution"]
    assert verify_insertion_manifest(manifest) == manifest["seal"]["manifest_sha256"]
