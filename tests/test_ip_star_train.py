"""End-to-end smoke tests for resumable star-graph training."""

from __future__ import annotations

import json
from pathlib import Path

from repro.ip_star_model import TransformerSpec
from repro.ip_star_train import TrainSettings, train


def test_tiny_fixed_training_emits_metrics_and_checkpoint(tmp_path: Path):
    output = tmp_path / "fixed"
    settings = TrainSettings(
        arm="fixed",
        output_dir=str(output),
        epochs=1,
        batch_size=4,
        max_steps=2,
        warmup_steps=1,
        device="cpu",
        bf16=False,
        log_every=1,
        eval_every=2,
        checkpoint_every=2,
        eval_limit=2,
        train_eval_limit=2,
        train_count=8,
        validation_count=2,
        test_count=2,
        decoder_spec=TransformerSpec(width=16, layers=1, heads=2, dropout=0.0),
        posterior_spec=TransformerSpec(width=16, layers=1, heads=2, dropout=0.0),
    )
    result = train(settings)
    assert result["step"] == 2
    assert Path(result["checkpoint"]).is_file()
    assert len(result["checkpoint_sha256"]) == 64
    manifest = json.loads((output / "manifest.json").read_text())
    assert not manifest["fidelity_mode"]
    records = [json.loads(line) for line in (output / "telemetry.jsonl").read_text().splitlines()]
    assert {record["kind"] for record in records} >= {
        "start",
        "train",
        "evaluation",
        "checkpoint",
        "complete",
    }
    evaluation = [record for record in records if record["kind"] == "evaluation"]
    assert {(record["split"], record["weights"]) for record in evaluation} == {
        ("validation", "raw"),
        ("train", "raw"),
    }


def test_tiny_learned_training_reaches_checkpoint(tmp_path: Path):
    output = tmp_path / "learned"
    settings = TrainSettings(
        arm="learned",
        output_dir=str(output),
        epochs=1,
        batch_size=2,
        max_steps=1,
        warmup_steps=1,
        device="cpu",
        bf16=False,
        log_every=1,
        eval_every=1,
        checkpoint_every=1,
        eval_limit=1,
        train_count=2,
        validation_count=1,
        test_count=1,
        decoder_spec=TransformerSpec(width=16, layers=1, heads=2, dropout=0.0),
        posterior_spec=TransformerSpec(width=16, layers=1, heads=2, dropout=0.0),
    )
    result = train(settings)
    assert result["step"] == 1
    assert Path(result["checkpoint"]).is_file()

    resumed = TrainSettings(
        **{
            **settings.__dict__,
            "epochs": 2,
            "max_steps": 2,
            "resume": result["checkpoint"],
        }
    )
    resumed_result = train(resumed)
    assert resumed_result["step"] == 2
    assert Path(resumed_result["checkpoint"]).is_file()
