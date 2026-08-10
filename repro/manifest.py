"""Sealed experiment manifest.

The manifest binds, in one hashed document: paper and code provenance, data
hashes and split hashes, the architecture signature and exact parameter count,
the authenticated vocabulary, measured generator counts, optimiser and loss
settings, every seed, the frozen inference thresholds, checkpoint/evaluation
cadence, environment versions, hardware, output paths and compute ceilings.

Sealing is a SHA-256 over the canonical JSON of everything except the ``seal``
block itself, so any later edit is detectable.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from repro import __version__
from repro.hashing import sha256_file, sha256_json
from repro.paths import repo_root

MANIFEST_SCHEMA_VERSION = 2

#: Paper artefacts declared by the replication specification.
PAPER_ARTIFACTS = {
    "2510.06190v2.pdf": "efec4ac49e7359659426b48e21c5e6b0076eae66bbbe89eee964572ca14eba9c",
    "2510.06190v2-source.tar": "891ef81fb986e31ecafb7803ef5eed3ed78d7f6a4cbd9dd186caedfc94ec085c",
}
DEFAULT_PAPER_DIR = Path("/home/ubuntu/papers/2510.06190")

#: Upstream commit this replication is pinned to.
UPSTREAM_COMMIT = "836b79b30286301ee1cca975db7e8468bc2a653e"

#: Repository files whose content is bound into the manifest.
HASHED_SOURCES: List[str] = [
    "dataset/sudoku/sudoku_generator.py",
    "dataset/sudoku/sudoku_solver.py",
    "dataset/sudoku/sudoku_loader.py",
    "dataset/sudoku/sudoku_verifier.py",
    "train/configs/sudoku.yaml",
    "train/configs/sudoku_paper.yaml",
    "train/apmdm_dataloader.py",
    "train/dataloader.py",
    "train/diffusion.py",
    "train/models/__init__.py",
    "train/models/dit.py",
    "repro/__init__.py",
    "repro/apmdm_teacher_forced_audit.py",
    "repro/cli.py",
    "repro/config.py",
    "repro/data.py",
    "repro/evaluator.py",
    "repro/hashing.py",
    "repro/insertion.py",
    "repro/insertion_trainer.py",
    "repro/ip_objective.py",
    "repro/ip_chem_data.py",
    "repro/ip_chem_eval.py",
    "repro/ip_chem_model.py",
    "repro/ip_chem_train.py",
    "ops/run_ip_chem_continue_runpod.sh",
    "repro/ip_star_data.py",
    "repro/ip_star_eval.py",
    "repro/ip_star_eval_checkpoint.py",
    "repro/ip_star_model.py",
    "repro/ip_star_train.py",
    "repro/keyed_diagnostics.py",
    "repro/losses.py",
    "repro/manifest.py",
    "repro/model.py",
    "repro/official.py",
    "repro/paths.py",
    "repro/sampler.py",
    "repro/s4_apmdm.py",
    "repro/s4_insertion.py",
    "repro/s4_ired.py",
    "repro/small_grokking.py",
    "repro/smoke.py",
    "repro/supervisor.py",
    "repro/telemetry.py",
    "repro/trainer.py",
    "repro/trajectories.py",
    "repro/transition.py",
    "repro/upstream_adapter.py",
    "repro/vocab.py",
]


class ManifestError(RuntimeError):
    """Raised when a manifest fails authentication."""


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root()), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception:  # pragma: no cover - git absent
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_provenance() -> dict:
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "dirty_paths": [line[3:] for line in (status or "").splitlines()][:50],
        "upstream_commit_pinned": UPSTREAM_COMMIT,
        "remotes": _git("remote", "-v"),
    }


def environment_provenance() -> dict:
    versions: Dict[str, Optional[str]] = {}
    for module_name in ("torch", "numpy", "yaml", "pytest", "flash_attn"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[module_name] = None

    cuda: Dict[str, object] = {"available": False}
    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda_version": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
            "capability": [
                list(torch.cuda.get_device_capability(i))
                for i in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        }
    except Exception:
        pass

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
        "package_versions": versions,
        "cuda": cuda,
        "repro_version": __version__,
        "attention_backend": "torch.nn.functional.scaled_dot_product_attention"
        if versions.get("flash_attn") is None
        else "flash_attn available (repro still uses SDPA)",
    }


def paper_provenance(paper_dir: Path | None = None) -> dict:
    directory = Path(paper_dir) if paper_dir is not None else DEFAULT_PAPER_DIR
    entries = {}
    for name, expected in PAPER_ARTIFACTS.items():
        path = directory / name
        if path.exists():
            actual = sha256_file(path)
            entries[name] = {
                "path": str(path),
                "sha256": actual,
                "expected_sha256": expected,
                "matches": actual == expected,
            }
        else:
            entries[name] = {
                "path": str(path),
                "sha256": None,
                "expected_sha256": expected,
                "matches": None,
            }
    return {"arxiv_id": "2510.06190v2", "artifacts": entries}


def source_hashes() -> Dict[str, Optional[str]]:
    root = repo_root()
    out: Dict[str, Optional[str]] = {}
    for rel in HASHED_SOURCES:
        path = root / rel
        out[rel] = sha256_file(path) if path.exists() else None
    return out


def build_manifest(
    *,
    config,
    parameter_count,
    vocabulary_report,
    train_provenance: dict,
    test_provenance: dict,
    overlap: dict,
    transition_index: dict,
    split_hashes: dict,
    sampler_config: dict,
    run_paths: dict,
    compute_ceilings: dict,
    checkpoint_every_n_steps: int,
    eval_every_n_steps: Optional[int],
    seeds: dict,
    notes: Optional[List[str]] = None,
    paper_dir: Path | None = None,
) -> dict:
    """Assemble and seal a manifest."""
    content = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment": "AP-MDM Sudoku replication (arXiv:2510.06190v2)",
        "paper": paper_provenance(paper_dir),
        "code": {"git": git_provenance(), "source_sha256": source_hashes()},
        "config": config.as_dict(),
        "architecture": {
            "signature": config.model.signature(),
            "fields": config.model.as_dict(),
            "parameter_count": parameter_count.as_dict(),
            "time_conditioning": config.time_conditioning,
            "output_functions": ["unmask", "remask", "insert", "delete"],
        },
        "vocabulary": vocabulary_report.as_dict(),
        "data": {
            "train": train_provenance,
            "test": test_provenance,
            "overlap": overlap,
            "test_set_use": "held out for evaluation only; never used for training or threshold selection",
        },
        "transitions": {
            "accounting": transition_index.get("accounting", {}),
            "index_sha256": transition_index.get("index_sha256"),
            "file_sha256": transition_index.get("file_sha256", {}),
            "validated": transition_index.get("validated"),
            "validation_summary": transition_index.get("validation_summary", {}),
        },
        "splits": split_hashes,
        "optimizer": {
            **config.optim.as_dict(),
            "optimizer": "AdamW",
            "schedule": "constant with warmup",
            "warmup_steps": config.warmup_steps,
            "global_batch_size": config.global_batch_size,
            "gradient_clip_val": config.gradient_clip_val,
            "precision": config.precision,
        },
        "losses": {
            "objective": "L_unmask + lr*L_remask + li*L_insert + ld*L_delete",
            "lambda_remask": config.lambda_remask,
            "lambda_insert": config.lambda_expand,
            "lambda_delete": config.lambda_contract,
        },
        "seeds": seeds,
        "inference": sampler_config,
        "cadence": {
            "checkpoint_every_n_steps": checkpoint_every_n_steps,
            "eval_every_n_steps": eval_every_n_steps,
            "max_steps": config.max_steps,
        },
        "environment": environment_provenance(),
        "paths": run_paths,
        # The paid-capacity prohibition is part of the contract, so it is
        # injected here rather than trusted to every caller.
        "compute_ceilings": {
            "paid_capacity": "forbidden (no RunPod or other rented capacity)",
            **compute_ceilings,
        },
        "notes": list(notes or []),
    }
    digest = sha256_json(content)
    content["seal"] = {
        "manifest_sha256": digest,
        "sealed_over": "canonical JSON of every field except 'seal'",
    }
    return content


def write_manifest(path: str | Path, manifest: dict, allow_overwrite: bool = False) -> str:
    """Write a sealed manifest; refuses to overwrite unless explicitly allowed."""
    target = Path(path)
    if target.exists() and not allow_overwrite:
        raise ManifestError(f"refusing to overwrite existing manifest: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest["seal"]["manifest_sha256"]


def verify_manifest(manifest: dict) -> str:
    """Re-derive and check the seal; returns the verified digest."""
    if "seal" not in manifest:
        raise ManifestError("manifest has no seal")
    content = {k: v for k, v in manifest.items() if k != "seal"}
    digest = sha256_json(content)
    recorded = manifest["seal"]["manifest_sha256"]
    if digest != recorded:
        raise ManifestError(
            f"manifest seal mismatch: recomputed {digest} != recorded {recorded}"
        )
    return digest


def load_manifest(path: str | Path, verify: bool = True) -> dict:
    """Load a manifest from disk, verifying its seal by default."""
    manifest = json.loads(Path(path).read_text())
    if verify:
        verify_manifest(manifest)
    return manifest


@dataclass(frozen=True)
class CheckpointBinding:
    """Fields a checkpoint must carry so the evaluator can bind it to a manifest."""

    architecture_signature: str
    vocabulary_size: int
    manifest_sha256: str
    step: int
    config_sha256: str

    def as_dict(self) -> dict:
        return {
            "architecture_signature": self.architecture_signature,
            "vocabulary_size": self.vocabulary_size,
            "manifest_sha256": self.manifest_sha256,
            "step": self.step,
            "config_sha256": self.config_sha256,
        }
