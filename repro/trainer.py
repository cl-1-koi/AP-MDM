"""Resumable supervised trainer for the paper-faithful AP-MDM Sudoku model.

Scientific settings are taken from the config and are not tuned here:
AdamW(lr 1e-4, betas (0.9, 0.999), weight decay 0.01, eps 1e-8), global batch
size 256, constant schedule with 250 linear-warmup steps, bf16 autocast,
gradient clipping 1.0, all operation-loss weights 1.0, up to 1,000,000
optimizer steps.

Determinism and resumability: batch composition for step *s* is a pure function
of ``(seed, s)`` via per-epoch permutations, so a resumed run consumes exactly
the batches the interrupted run would have consumed.  Python/NumPy/Torch RNG
states are additionally checkpointed.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from repro.config import SudokuConfig
from repro.losses import head_diagnostics, supervised_loss
from repro.manifest import CheckpointBinding
from repro.model import APMDMEncoder, build_model
from repro.telemetry import TelemetryWriter, gpu_snapshot
from repro.trajectories import TransitionDataset

CHECKPOINT_NAME = "checkpoint_step_{step:09d}.pt"
LATEST_NAME = "latest.pt"


@dataclass
class TrainerSettings:
    """Run-level settings that are not part of the scientific contract."""

    max_steps: int
    checkpoint_every: int = 10_000
    log_every: int = 100
    monitor_every: int = 3_000
    monitor_batches: int = 8
    device: str = "cuda"
    dataloader_seed_salt: int = 1_000_003
    time_budget_seconds: Optional[float] = None
    keep_last_checkpoints: int = 3


def seed_everything(seed: int) -> Dict[str, int]:
    """Seed every RNG this run touches and report the seeds used."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return {
        "seed": seed,
        "python_random": seed,
        "numpy": seed % (2**32),
        "torch": seed,
        "torch_cuda": seed,
        "dataloader_base": seed,
        "sampling": seed,
    }


def constant_schedule_with_warmup(optimizer, num_warmup_steps: int):
    """Match ``transformers.get_constant_schedule_with_warmup``."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EpochPermutationSampler:
    """Deterministic, resumable batch index generator."""

    def __init__(self, indices: np.ndarray, batch_size: int, seed: int, salt: int = 1_000_003):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.salt = int(salt)
        self.n = int(self.indices.size)
        if self.n < self.batch_size:
            raise ValueError(
                f"dataset has {self.n} transitions, fewer than one batch of {self.batch_size}"
            )
        self.batches_per_epoch = self.n // self.batch_size
        self._epoch = -1
        self._perm: Optional[np.ndarray] = None

    def _permutation(self, epoch: int) -> np.ndarray:
        if self._epoch != epoch:
            rng = np.random.default_rng(self.seed * self.salt + epoch)
            self._perm = self.indices[rng.permutation(self.n)]
            self._epoch = epoch
        assert self._perm is not None
        return self._perm

    def batch_for_step(self, step: int) -> np.ndarray:
        """Return the index batch consumed by 1-based optimizer step ``step``."""
        offset = (step - 1) % self.batches_per_epoch
        epoch = (step - 1) // self.batches_per_epoch
        perm = self._permutation(epoch)
        start = offset * self.batch_size
        return perm[start : start + self.batch_size]


def to_torch_batch(raw: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for key in ("x_k", "y_star", "r_star", "e_star", "c_star"):
        out[key] = torch.as_tensor(raw[key], dtype=torch.long, device=device)
    out["attention_mask"] = torch.ones_like(out["x_k"], dtype=torch.float32)
    return out


class Trainer:
    """Supervised AP-MDM trainer with checkpointing, telemetry and resume."""

    def __init__(
        self,
        config: SudokuConfig,
        dataset: TransitionDataset,
        train_indices: np.ndarray,
        monitor_indices: np.ndarray,
        run_dir: Path,
        settings: TrainerSettings,
        manifest_sha256: str,
        run_id: str,
    ):
        self.config = config
        self.dataset = dataset
        self.train_indices = np.asarray(train_indices, dtype=np.int64)
        self.monitor_indices = np.asarray(monitor_indices, dtype=np.int64)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self.manifest_sha256 = manifest_sha256
        self.run_id = run_id

        self.device = torch.device(
            settings.device if (settings.device != "cuda" or torch.cuda.is_available()) else "cpu"
        )
        self.seeds = seed_everything(config.seed)
        self.model: APMDMEncoder = build_model(
            config.model, time_conditioning=config.time_conditioning, seed=config.seed
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.optim.lr,
            betas=(config.optim.beta1, config.optim.beta2),
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay,
        )
        self.scheduler = constant_schedule_with_warmup(self.optimizer, config.warmup_steps)
        self.sampler = EpochPermutationSampler(
            self.train_indices,
            config.global_batch_size,
            config.seed,
            settings.dataloader_seed_salt,
        )
        self.step = 0
        self.autocast_dtype = (
            torch.bfloat16 if (self.device.type == "cuda" and config.precision.startswith("bf16")) else None
        )
        self.telemetry = TelemetryWriter(self.run_dir / "telemetry.jsonl", run_id=run_id)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def binding(self) -> CheckpointBinding:
        return CheckpointBinding(
            architecture_signature=self.config.model.signature(),
            vocabulary_size=self.config.model.vocab_size,
            manifest_sha256=self.manifest_sha256,
            step=self.step,
            config_sha256=self.config.sha256,
        )

    def save_checkpoint(self, tag: Optional[str] = None) -> Path:
        payload = {
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "binding": self.binding().as_dict(),
            "seeds": self.seeds,
            "model_spec": self.config.model.as_dict(),
            "time_conditioning": self.config.time_conditioning,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        name = tag or CHECKPOINT_NAME.format(step=self.step)
        path = self.ckpt_dir / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        latest = self.ckpt_dir / LATEST_NAME
        tmp_latest = latest.with_suffix(".tmp")
        torch.save(payload, tmp_latest)
        os.replace(tmp_latest, latest)
        self._prune_checkpoints()
        self.telemetry.write("checkpoint", step=self.step, path=str(path))
        return path

    def _prune_checkpoints(self) -> None:
        keep = self.settings.keep_last_checkpoints
        if keep <= 0:
            return
        files = sorted(self.ckpt_dir.glob("checkpoint_step_*.pt"))
        for stale in files[:-keep]:
            try:
                stale.unlink()
            except OSError:  # pragma: no cover
                pass

    def load_checkpoint(self, path: Path) -> int:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        binding = payload.get("binding", {})
        if binding.get("architecture_signature") != self.config.model.signature():
            raise RuntimeError(
                "checkpoint architecture signature does not match the current config"
            )
        if binding.get("manifest_sha256") not in (None, self.manifest_sha256):
            raise RuntimeError("checkpoint was produced under a different sealed manifest")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.step = int(payload["step"])
        rng = payload.get("rng") or {}
        try:
            if rng.get("python"):
                random.setstate(tuple(rng["python"]) if isinstance(rng["python"], list) else rng["python"])
            if rng.get("numpy") is not None:
                np.random.set_state(rng["numpy"])
            if rng.get("torch") is not None:
                torch.set_rng_state(rng["torch"].cpu() if hasattr(rng["torch"], "cpu") else rng["torch"])
            if rng.get("torch_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
        except Exception as exc:  # pragma: no cover - RNG restore is best effort
            self.telemetry.write("warning", message=f"RNG restore failed: {exc}")
        self.telemetry.write("resume", step=self.step, path=str(path))
        return self.step

    def maybe_resume(self) -> bool:
        latest = self.ckpt_dir / LATEST_NAME
        if latest.exists():
            self.load_checkpoint(latest)
            return True
        return False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _forward_backward(self, batch: Dict[str, torch.Tensor]):
        self.optimizer.zero_grad(set_to_none=True)
        if self.autocast_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                outputs = self.model(batch["x_k"])
        else:
            outputs = self.model(batch["x_k"])
        loss = supervised_loss(
            outputs,
            batch["x_k"],
            batch["y_star"],
            batch["r_star"],
            batch["e_star"],
            batch["c_star"],
            attention_mask=batch["attention_mask"],
            lambda_remask=self.config.lambda_remask,
            lambda_insert=self.config.lambda_expand,
            lambda_delete=self.config.lambda_contract,
        )
        loss.total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clip_val
        )
        self.optimizer.step()
        self.scheduler.step()
        return loss, outputs, float(grad_norm)

    @torch.no_grad()
    def monitor(self) -> Dict[str, float]:
        """In-distribution transition monitor.

        This is drawn from the same 100 source puzzles as training; it measures
        optimisation progress, not held-out generalisation.
        """
        self.model.eval()
        totals: Dict[str, float] = {}
        count = 0
        batch_size = self.config.global_batch_size
        for i in range(self.settings.monitor_batches):
            start = i * batch_size
            idx = self.monitor_indices[start : start + batch_size]
            if idx.size == 0:
                break
            batch = to_torch_batch(self.dataset.batch(idx), self.device)
            if self.autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    outputs = self.model(batch["x_k"])
            else:
                outputs = self.model(batch["x_k"])
            loss = supervised_loss(
                outputs,
                batch["x_k"],
                batch["y_star"],
                batch["r_star"],
                batch["e_star"],
                batch["c_star"],
                attention_mask=batch["attention_mask"],
                lambda_remask=self.config.lambda_remask,
                lambda_insert=self.config.lambda_expand,
                lambda_delete=self.config.lambda_contract,
            )
            stats = {**loss.scalars(), **head_diagnostics(
                outputs, batch["x_k"], batch["y_star"], batch["r_star"], batch["e_star"], batch["c_star"]
            )}
            for key, value in stats.items():
                if isinstance(value, (int, float)) and not math.isnan(float(value)):
                    totals[key] = totals.get(key, 0.0) + float(value)
            count += 1
        self.model.train()
        return {key: value / max(1, count) for key, value in totals.items()}

    def train(self, max_steps: Optional[int] = None) -> Dict[str, object]:
        target = int(max_steps if max_steps is not None else self.settings.max_steps)
        self.model.train()
        started = time.time()
        window_start = started
        window_samples = 0
        summary = {"start_step": self.step, "target_step": target}

        while self.step < target:
            self.step += 1
            idx = self.sampler.batch_for_step(self.step)
            batch = to_torch_batch(self.dataset.batch(idx), self.device)
            loss, outputs, grad_norm = self._forward_backward(batch)
            window_samples += int(batch["x_k"].shape[0])

            if self.step % self.settings.log_every == 0 or self.step == target:
                now = time.time()
                elapsed = max(1e-9, now - window_start)
                record = {
                    "step": self.step,
                    "lr": float(self.scheduler.get_last_lr()[0]),
                    "grad_norm": grad_norm,
                    "steps_per_second": self.settings.log_every / elapsed
                    if self.step % self.settings.log_every == 0
                    else float("nan"),
                    "samples_per_second": window_samples / elapsed,
                    **loss.scalars(),
                    **head_diagnostics(
                        outputs,
                        batch["x_k"],
                        batch["y_star"],
                        batch["r_star"],
                        batch["e_star"],
                        batch["c_star"],
                    ),
                    **{f"gpu_{k}": v for k, v in gpu_snapshot(self.device).items()},
                }
                self.telemetry.write("train", **record)
                window_start = now
                window_samples = 0

            if self.settings.monitor_every and self.step % self.settings.monitor_every == 0:
                self.telemetry.write("monitor", step=self.step, **self.monitor())

            if self.settings.checkpoint_every and self.step % self.settings.checkpoint_every == 0:
                self.save_checkpoint()

            if (
                self.settings.time_budget_seconds is not None
                and time.time() - started > self.settings.time_budget_seconds
            ):
                summary["stopped_early"] = "time_budget_exhausted"
                break

        self.save_checkpoint()
        summary.update(
            {
                "end_step": self.step,
                "wall_seconds": time.time() - started,
                "device": str(self.device),
            }
        )
        self.telemetry.write("train_summary", **summary)
        (self.run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
        return summary

    def close(self) -> None:
        self.telemetry.close()
