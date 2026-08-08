"""Trainer/manifest tests for the monotone Sudoku ladder."""

from __future__ import annotations

from dataclasses import replace

import pytest

from repro import data as data_mod
from repro.insertion import SudokuInsertionPolicy, effective_model_spec
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
