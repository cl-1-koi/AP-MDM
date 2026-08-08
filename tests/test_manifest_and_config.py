"""Manifest sealing/immutability, config integrity and portability tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from repro import vocab
from repro.config import load_historical_config, load_paper_config
from repro.hashing import sha256_file
from repro.manifest import (
    ManifestError,
    build_manifest,
    load_manifest,
    source_hashes,
    verify_manifest,
    write_manifest,
)
from repro.model import count_parameters
from repro.paths import assert_outside_repo, data_root, repo_root

#: SHA-256 of the historical config as inherited from upstream commit 836b79b3.
HISTORICAL_CONFIG_SHA256 = "e7e0da4cebfd9733033ed7bc65cc7d090abc418cd0f3eadf263bb12e69b9d273"


# --------------------------------------------------------------------------
# Historical config integrity
# --------------------------------------------------------------------------


def test_historical_config_file_is_byte_identical_to_upstream():
    path = repo_root() / "train" / "configs" / "sudoku.yaml"
    assert sha256_file(path) == HISTORICAL_CONFIG_SHA256


def test_historical_config_meaning_is_unchanged(historical_config):
    spec = historical_config.model
    assert (spec.n_blocks, spec.n_heads, spec.hidden_size, spec.cond_dim) == (4, 2, 128, 64)
    assert spec.vocab_size == 31
    assert spec.length == 400
    assert historical_config.optim.lr == 1e-4
    assert historical_config.optim.weight_decay == 0.01
    assert historical_config.global_batch_size == 256
    assert historical_config.warmup_steps == 250
    assert historical_config.max_steps == 1_000_000
    assert historical_config.seed == 42
    assert historical_config.time_conditioning is False
    assert historical_config.train_ratio == 0.99
    assert (
        historical_config.lambda_remask
        == historical_config.lambda_expand
        == historical_config.lambda_contract
        == 1.0
    )


def test_paper_config_is_a_separate_file_and_differs_only_where_intended(
    paper_config, historical_config
):
    assert paper_config.path.name == "sudoku_paper.yaml"
    assert paper_config.path != historical_config.path
    # Scientific settings that must agree with the historical config.
    assert paper_config.optim.as_dict() == historical_config.optim.as_dict()
    assert paper_config.global_batch_size == historical_config.global_batch_size
    assert paper_config.warmup_steps == historical_config.warmup_steps
    assert paper_config.max_steps == historical_config.max_steps
    assert paper_config.seed == historical_config.seed
    assert paper_config.time_conditioning == historical_config.time_conditioning
    assert paper_config.gradient_clip_val == historical_config.gradient_clip_val
    assert paper_config.precision == historical_config.precision
    assert paper_config.train_ratio == historical_config.train_ratio
    for field in ("lambda_remask", "lambda_expand", "lambda_contract"):
        assert getattr(paper_config, field) == getattr(historical_config, field)
    # Architecture is where they intentionally diverge.
    assert paper_config.model.signature() != historical_config.model.signature()


def test_paper_config_does_not_carry_absolute_workspace_paths():
    text = (repo_root() / "train" / "configs" / "sudoku_paper.yaml").read_text()
    assert "/workspace/" not in text


def test_historical_config_still_contains_its_stale_paths_unmodified():
    """We document the stale paths in the audit; we do not silently rewrite them."""
    text = (repo_root() / "train" / "configs" / "sudoku.yaml").read_text()
    assert "/workspace/data" in text


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def _manifest(tmp_path, paper_config):
    return build_manifest(
        config=paper_config,
        parameter_count=count_parameters(paper_config.model),
        vocabulary_report=vocab.authenticate_vocabulary(),
        train_provenance={"name": "train", "n": 100},
        test_provenance={"name": "test", "n": 10000},
        overlap={"clean": True},
        transition_index={"accounting": {"n_source_puzzles": 100}, "index_sha256": "x" * 64},
        split_hashes={"train_indices_sha256": "a" * 64},
        sampler_config={"tau_remask": 0.5, "max_steps": 65536},
        run_paths={"run_dir": str(tmp_path)},
        compute_ceilings={"smoke_ceiling_combined_a10_minutes": 15},
        checkpoint_every_n_steps=10_000,
        eval_every_n_steps=3_000,
        seeds={"seed": 42},
        notes=["unit test"],
    )


def test_manifest_seals_and_verifies(tmp_path, paper_config):
    manifest = _manifest(tmp_path, paper_config)
    assert len(manifest["seal"]["manifest_sha256"]) == 64
    assert verify_manifest(manifest) == manifest["seal"]["manifest_sha256"]


def test_manifest_seal_detects_tampering(tmp_path, paper_config):
    manifest = _manifest(tmp_path, paper_config)
    manifest["config"]["optim"]["lr"] = 3e-4
    with pytest.raises(ManifestError):
        verify_manifest(manifest)


def test_manifest_is_immutable_on_disk(tmp_path, paper_config):
    manifest = _manifest(tmp_path, paper_config)
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    with pytest.raises(ManifestError):
        write_manifest(path, manifest)
    write_manifest(path, manifest, allow_overwrite=True)
    assert load_manifest(path)["seal"]["manifest_sha256"] == manifest["seal"]["manifest_sha256"]


def test_manifest_binds_everything_the_contract_requires(tmp_path, paper_config):
    manifest = _manifest(tmp_path, paper_config)
    for key in (
        "paper", "code", "config", "architecture", "vocabulary", "data", "transitions",
        "splits", "optimizer", "losses", "seeds", "inference", "cadence", "environment",
        "paths", "compute_ceilings", "seal",
    ):
        assert key in manifest, key
    assert manifest["architecture"]["parameter_count"]["total"] == 6_050_594
    assert manifest["architecture"]["output_functions"] == ["unmask", "remask", "insert", "delete"]
    assert manifest["architecture"]["time_conditioning"] is False
    assert manifest["vocabulary"]["effective_size"] == 31
    assert manifest["optimizer"]["optimizer"] == "AdamW"
    assert manifest["compute_ceilings"]["paid_capacity"].startswith("forbidden")


def test_manifest_records_paper_artifact_digests(tmp_path, paper_config):
    manifest = _manifest(tmp_path, paper_config)
    artifacts = manifest["paper"]["artifacts"]
    assert set(artifacts) == {"2510.06190v2.pdf", "2510.06190v2-source.tar"}
    for entry in artifacts.values():
        assert len(entry["expected_sha256"]) == 64
        if entry["sha256"] is not None:
            assert entry["matches"] is True


def test_source_hashes_cover_every_repro_module():
    hashes = source_hashes()
    for rel, digest in hashes.items():
        assert digest is not None, rel
        assert len(digest) == 64
    for module in sorted((repo_root() / "repro").glob("*.py")):
        assert f"repro/{module.name}" in hashes, module.name


# --------------------------------------------------------------------------
# Portability and import safety
# --------------------------------------------------------------------------


def test_data_root_is_outside_the_repository():
    assert repo_root() not in data_root().parents
    assert data_root() != repo_root()


def test_assert_outside_repo_blocks_in_tree_paths():
    with pytest.raises(ValueError):
        assert_outside_repo(repo_root() / "artifacts")
    with pytest.raises(ValueError):
        assert_outside_repo(repo_root())
    assert assert_outside_repo(data_root() / "ok")


def _non_docstring_constants(tree):
    """Yield every string constant that is not a module/class/function docstring."""
    import ast

    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    docstrings.add(id(body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            yield node.value


def test_no_absolute_workspace_paths_in_the_repro_package():
    """Stale ``/workspace/...`` literals may be *documented* but never *used*."""
    import ast

    for module in sorted((repo_root() / "repro").glob("*.py")):
        tree = ast.parse(module.read_text())
        for value in _non_docstring_constants(tree):
            assert "/workspace/" not in value, (module.name, value)


def test_every_repro_module_imports_cleanly_in_a_fresh_interpreter():
    modules = [
        f"repro.{m.stem}" for m in sorted((repo_root() / "repro").glob("*.py")) if m.stem != "__init__"
    ]
    code = "import importlib\n" + "\n".join(f"importlib.import_module({m!r})" for m in modules)
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=str(repo_root()), capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr


def test_cli_help_works_without_a_gpu():
    result = subprocess.run(
        [sys.executable, "-m", "repro.cli", "--help"],
        cwd=str(repo_root()), capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0
    for command in ("authenticate", "generate", "manifest", "train", "evaluate", "preflight", "smoke"):
        assert command in result.stdout
