"""Resumable trainer and sealed manifest for the monotone Sudoku ladder."""

from __future__ import annotations

import json
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from repro import data as data_mod
from repro.config import SudokuConfig
from repro.hashing import sha256_file, sha256_json
from repro.insertion import (
    ConditionMode,
    InsertionArm,
    SudokuInsertionPolicy,
    SudokuOrderPosterior,
    default_posterior_spec,
    effective_model_spec,
    fixed_or_random_loss,
    learned_insertion_loss,
    solve_monotone,
    validate_arm,
    validate_condition_mode,
)
from repro.manifest import environment_provenance
from repro.paths import repo_root
from repro.telemetry import TelemetryWriter, gpu_snapshot
from repro.trainer import constant_schedule_with_warmup, seed_everything

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class InsertionTrainerSettings:
    arm: InsertionArm
    run_id: str
    max_steps: int
    batch_size: int
    checkpoint_every: int = 10_000
    log_every: int = 100
    eval_every: int = 5_000
    eval_limit: int = 256
    device: str = "cuda"
    seed: int = 42
    q_lr_ratio: float = 0.01
    bf16: bool = True
    time_budget_seconds: Optional[float] = None
    keep_last_checkpoints: int = 3
    condition_mode: ConditionMode = "puzzle_only"

    def __post_init__(self):
        validate_arm(self.arm)
        validate_condition_mode(self.condition_mode)
        if self.max_steps <= 0 or self.batch_size <= 0:
            raise ValueError("max_steps and batch_size must be positive")
        if not (0 < self.q_lr_ratio <= 1):
            raise ValueError("q_lr_ratio must be in (0,1]")


def _git(*args: str) -> Optional[str]:
    result = subprocess.run(
        ["git", "-C", str(repo_root()), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _parameter_count(module: torch.nn.Module) -> dict:
    return {
        "total": int(sum(p.numel() for p in module.parameters())),
        "trainable": int(sum(p.numel() for p in module.parameters() if p.requires_grad)),
    }


def build_insertion_manifest(
    *,
    config: SudokuConfig,
    settings: InsertionTrainerSettings,
    train: data_mod.SudokuSplit,
    test: data_mod.SudokuSplit,
    overlap: dict,
    policy: SudokuInsertionPolicy,
    posterior: Optional[SudokuOrderPosterior],
    run_dir: Path,
) -> dict:
    paper_path = Path("/home/ubuntu/papers/2606.02133/2606.02133v3.pdf")
    source_path = Path("/home/ubuntu/papers/2606.02133/2606.02133v3-source.tar")
    status = _git("status", "--porcelain") or ""
    paid_authorization = os.environ.get(
        "APMDM_INSERTION_PAID_CAPACITY_AUTHORIZATION"
    )
    content = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "Sudoku monotone generation ladder",
        "arm": settings.arm,
        "mechanism_boundary": {
            "monotone": True,
            "can_remask": False,
            "can_replace": False,
            "can_delete": False,
            "known_fixed_canvas": True,
            "termination": "deterministic when all 81 cells are filled",
            "condition_mode": settings.condition_mode,
            "oracle_information": (
                "the complete solution is exposed through COLOR_1..COLOR_9 at the same cell; this is an intentional local visible-information control"
                if settings.condition_mode == "aligned_solution_hint"
                else "the complete solution is exposed through COLOR_1..COLOR_9 on the transposed board; this is an intentional retrieval/reordering control"
                if settings.condition_mode == "transposed_solution_hint"
                else None
            ),
            "learned_insertion": (
                "fixed-canvas permutation ELBO with exact next-cell expectation and M=2 RLOO"
                if settings.arm == "learned_insertion"
                else None
            ),
        },
        "papers": {
            "apmdm": {
                "arxiv": "2510.06190v2",
                "role": "full remask/insert/delete comparison",
            },
            "insertion_process": {
                "arxiv": "2606.02133v3",
                "role": "permutation-trajectory bijection and variational learned order",
                "pdf": str(paper_path),
                "pdf_sha256": sha256_file(paper_path) if paper_path.exists() else None,
                "source": str(source_path),
                "source_sha256": sha256_file(source_path) if source_path.exists() else None,
                "official_code_release_found": False,
            },
        },
        "code": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status),
            "dirty_paths": [line[3:] for line in status.splitlines()],
            "sources": {
                name: sha256_file(repo_root() / name)
                for name in (
                    "repro/insertion.py",
                    "repro/insertion_trainer.py",
                    "repro/cli.py",
                    "repro/model.py",
                    "repro/data.py",
                    "repro/telemetry.py",
                    "train/configs/sudoku_paper.yaml",
                    "tests/test_insertion.py",
                    "tests/test_insertion_trainer.py",
                    "SUDOKU_MONOTONE_LADDER_20260808.md",
                    "SUDOKU_MONOTONE_RUNPOD_R0_20260808.md",
                    "SUDOKU_MONOTONE_RUNPOD_R1_20260808.md",
                    "SUDOKU_VISIBLE_COMPONENT_GATE_20260809.md",
                    "ops/run_insertion_r0_runpod.sh",
                    "ops/run_insertion_r1_runpod.sh",
                    "ops/run_sudoku_vc1_runpod.sh",
                )
                if (repo_root() / name).exists()
            },
        },
        "data": {
            "train": train.provenance(),
            "test": test.provenance(),
            "overlap": overlap,
            "raw_training_instances": train.n,
            "derived_prefixes": "sampled online; no solver/backtracking transitions consumed",
        },
        "architecture": {
            "policy_signature": policy.signature(),
            "policy_spec": policy.spec.as_dict(),
            "policy_parameters": _parameter_count(policy),
            "posterior_spec": posterior.spec.as_dict() if posterior is not None else None,
            "posterior_parameters": _parameter_count(posterior) if posterior is not None else None,
            "shared_across_arms": "policy trunk and cell/digit heads; non-learned arms do not optimize cell head",
        },
        "optimizer": {
            **config.optim.as_dict(),
            "policy_lr": config.optim.lr,
            "posterior_lr": config.optim.lr * settings.q_lr_ratio if posterior is not None else None,
            "q_lr_ratio": settings.q_lr_ratio,
            "schedule": "constant with linear warmup",
            "warmup_steps": config.warmup_steps,
            "gradient_clip": config.gradient_clip_val,
        },
        "training": asdict(settings),
        "evaluation": {
            "primary": "exact held-out valid Sudoku solve rate consistent with givens",
            "digit_decoding": "argmax",
            "cell_decoding": {
                "fixed_ar": "first empty row-major cell",
                "random_insertion": "uniform random empty cell",
                "learned_insertion": "argmax learned cell logit among empty cells",
            }[settings.arm],
            "heldout_limit": settings.eval_limit,
        },
        "environment": environment_provenance(),
        "run_dir": str(run_dir.resolve()),
        "compute": {
            "paid_capacity": (
                paid_authorization
                if paid_authorization
                else "forbidden until separate explicit owner authorization"
            ),
            "initial_execution": (
                "parallel RunPod A40 R0 smoke after explicit owner authorization"
                if paid_authorization
                else "local A10 only after the active AP-MDM run releases it"
            ),
        },
    }
    digest = sha256_json(content)
    return {
        **content,
        "seal": {
            "manifest_sha256": digest,
            "sealed_over": "canonical JSON excluding seal",
        },
    }


def verify_insertion_manifest(manifest: dict) -> str:
    recorded = manifest.get("seal", {}).get("manifest_sha256")
    actual = sha256_json({k: v for k, v in manifest.items() if k != "seal"})
    if actual != recorded:
        raise RuntimeError(f"manifest seal mismatch: {actual} != {recorded}")
    return actual


class InsertionTrainer:
    def __init__(
        self,
        config: SudokuConfig,
        settings: InsertionTrainerSettings,
        train: data_mod.SudokuSplit,
        test: data_mod.SudokuSplit,
        run_dir: Path,
        manifest_sha256: str,
    ):
        self.config = config
        self.settings = settings
        self.train_split = train
        self.test_split = test
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_sha256 = manifest_sha256
        self.device = torch.device(
            settings.device
            if settings.device != "cuda" or torch.cuda.is_available()
            else "cpu"
        )
        self.seeds = seed_everything(settings.seed)
        self.generator = torch.Generator(device=self.device).manual_seed(settings.seed + 17)
        policy_spec = effective_model_spec(config.model)
        self.policy = SudokuInsertionPolicy(policy_spec).to(self.device)
        self.posterior = (
            SudokuOrderPosterior(default_posterior_spec(policy_spec)).to(self.device)
            if settings.arm == "learned_insertion"
            else None
        )
        groups = [{"params": self.policy.parameters(), "lr": config.optim.lr}]
        if self.posterior is not None:
            groups.append(
                {
                    "params": self.posterior.parameters(),
                    "lr": config.optim.lr * settings.q_lr_ratio,
                }
            )
        self.optimizer = torch.optim.AdamW(
            groups,
            lr=config.optim.lr,
            betas=(config.optim.beta1, config.optim.beta2),
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay,
        )
        self.scheduler = constant_schedule_with_warmup(
            self.optimizer, config.warmup_steps
        )
        self.step = 0
        self.autocast_dtype = (
            torch.bfloat16
            if self.device.type == "cuda" and settings.bf16
            else None
        )
        # Authenticated splits are deliberately read-only.  Materialize owned
        # tensors so PyTorch never aliases an immutable NumPy allocation.
        self.train_puzzles = torch.tensor(
            np.array(train.puzzles, copy=True), dtype=torch.long, device=self.device
        )
        self.train_solutions = torch.tensor(
            np.array(train.solutions, copy=True), dtype=torch.long, device=self.device
        )
        self.telemetry = TelemetryWriter(
            self.run_dir / "telemetry.jsonl", run_id=settings.run_id
        )

    @property
    def architecture_signature(self) -> str:
        posterior = (
            f"|q={self.posterior.spec.signature()}" if self.posterior is not None else "|q=none"
        )
        return self.policy.signature() + posterior

    def _batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        index = torch.randint(
            0,
            self.train_puzzles.shape[0],
            (self.settings.batch_size,),
            device=self.device,
            generator=self.generator,
        )
        return self.train_puzzles[index], self.train_solutions[index]

    def _loss(self, puzzles: torch.Tensor, solutions: torch.Tensor):
        if self.settings.arm == "learned_insertion":
            assert self.posterior is not None
            return learned_insertion_loss(
                self.policy,
                self.posterior,
                puzzles,
                solutions,
                self.generator,
                condition_mode=self.settings.condition_mode,
            )
        return fixed_or_random_loss(
            self.policy,
            puzzles,
            solutions,
            self.settings.arm,
            self.generator,
            self.settings.condition_mode,
        )

    def train_step(self) -> tuple[Dict[str, float], float]:
        self.policy.train()
        if self.posterior is not None:
            self.posterior.train()
        puzzles, solutions = self._batch()
        self.optimizer.zero_grad(set_to_none=True)
        if self.autocast_dtype is not None:
            with torch.autocast("cuda", dtype=self.autocast_dtype):
                result = self._loss(puzzles, solutions)
        else:
            result = self._loss(puzzles, solutions)
        result.loss.backward()
        parameters = list(self.policy.parameters())
        if self.posterior is not None:
            parameters.extend(self.posterior.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, self.config.gradient_clip_val
        )
        self.optimizer.step()
        self.scheduler.step()
        metrics = {"loss": float(result.loss.detach())}
        metrics.update({k: float(v) for k, v in result.metrics.items()})
        return metrics, float(grad_norm)

    @torch.no_grad()
    def evaluate(self, limit: Optional[int] = None) -> dict:
        n = min(limit or self.settings.eval_limit, self.test_split.n)
        self.policy.eval()
        context = (
            torch.autocast("cuda", dtype=self.autocast_dtype)
            if self.autocast_dtype is not None
            else torch.autocast("cpu", enabled=False)
        )
        with context:
            result = solve_monotone(
                self.policy,
                np.asarray(self.test_split.puzzles[:n]),
                self.settings.arm,
                device=self.device,
                seed=self.settings.seed + self.step,
                condition_mode=self.settings.condition_mode,
                solutions=np.asarray(self.test_split.solutions[:n]),
            )
        aggregate = result.aggregate()
        aggregate.update(
            {
                "step": self.step,
                "arm": self.settings.arm,
                "condition_mode": self.settings.condition_mode,
            }
        )
        rows_path = self.run_dir / f"eval_rows_step_{self.step:09d}.jsonl"
        tmp = rows_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            for i in range(n):
                row = {
                    "index": i,
                    "solved": bool(result.solved[i]),
                    "valid": bool(result.valid[i]),
                    "consistent": bool(result.consistent[i]),
                    "steps": int(result.steps[i]),
                    "grid": result.grids[i].tolist(),
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        os.replace(tmp, rows_path)
        aggregate["rows_path"] = str(rows_path)
        aggregate["rows_sha256"] = sha256_file(rows_path)
        self.telemetry.write("eval", **aggregate)
        return aggregate

    def save_checkpoint(self) -> Path:
        payload = {
            "version": 1,
            "step": self.step,
            "arm": self.settings.arm,
            "architecture_signature": self.architecture_signature,
            "manifest_sha256": self.manifest_sha256,
            "policy": self.policy.state_dict(),
            "posterior": self.posterior.state_dict() if self.posterior is not None else None,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "generator_state": self.generator.get_state(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        path = self.ckpt_dir / f"checkpoint_step_{self.step:09d}.pt"
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        latest = self.ckpt_dir / "latest.pt"
        latest_tmp = latest.with_suffix(".tmp")
        torch.save(payload, latest_tmp)
        os.replace(latest_tmp, latest)
        files = sorted(self.ckpt_dir.glob("checkpoint_step_*.pt"))
        for stale in files[: -self.settings.keep_last_checkpoints]:
            stale.unlink()
        self.telemetry.write(
            "checkpoint", step=self.step, path=str(path), sha256=sha256_file(path)
        )
        return path

    def load_checkpoint(self, path: Path) -> int:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("arm") != self.settings.arm:
            raise RuntimeError("checkpoint arm mismatch")
        if payload.get("architecture_signature") != self.architecture_signature:
            raise RuntimeError("checkpoint architecture mismatch")
        if payload.get("manifest_sha256") != self.manifest_sha256:
            raise RuntimeError("checkpoint manifest mismatch")
        self.policy.load_state_dict(payload["policy"])
        if self.posterior is not None:
            if payload.get("posterior") is None:
                raise RuntimeError("learned-insertion checkpoint has no posterior")
            self.posterior.load_state_dict(payload["posterior"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        # Generator states are serialized as CPU ByteTensors for both CPU and
        # CUDA generators; ``set_state`` performs the backend-specific copy.
        self.generator.set_state(payload["generator_state"].cpu())
        self.step = int(payload["step"])
        rng = payload.get("rng", {})
        if rng.get("python") is not None:
            random.setstate(rng["python"])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"].cpu())
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        self.telemetry.write("resume", step=self.step, path=str(path))
        return self.step

    def maybe_resume(self) -> bool:
        path = self.ckpt_dir / "latest.pt"
        if not path.exists():
            return False
        self.load_checkpoint(path)
        return True

    def train(self) -> dict:
        started = time.time()
        window_started = started
        window_examples = 0
        start_step = self.step
        while self.step < self.settings.max_steps:
            self.step += 1
            metrics, grad_norm = self.train_step()
            window_examples += self.settings.batch_size
            if self.step % self.settings.log_every == 0 or self.step == self.settings.max_steps:
                now = time.time()
                elapsed = max(now - window_started, 1e-9)
                self.telemetry.write(
                    "train",
                    step=self.step,
                    grad_norm=grad_norm,
                    lr=[float(x) for x in self.scheduler.get_last_lr()],
                    steps_per_second=self.settings.log_every / elapsed,
                    examples_per_second=window_examples / elapsed,
                    **metrics,
                    **{f"gpu_{k}": v for k, v in gpu_snapshot(self.device).items()},
                )
                window_started = now
                window_examples = 0
            if self.settings.eval_every and self.step % self.settings.eval_every == 0:
                self.evaluate()
            if self.settings.checkpoint_every and self.step % self.settings.checkpoint_every == 0:
                self.save_checkpoint()
            if (
                self.settings.time_budget_seconds is not None
                and time.time() - started >= self.settings.time_budget_seconds
            ):
                break
        if self.settings.eval_every and self.step % self.settings.eval_every:
            self.evaluate()
        self.save_checkpoint()
        summary = {
            "arm": self.settings.arm,
            "start_step": start_step,
            "end_step": self.step,
            "target_step": self.settings.max_steps,
            "wall_seconds": time.time() - started,
            "device": str(self.device),
            "completed": self.step >= self.settings.max_steps,
        }
        self.telemetry.write("train_summary", **summary)
        (self.run_dir / "train_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True)
        )
        return summary

    def close(self) -> None:
        self.telemetry.close()
