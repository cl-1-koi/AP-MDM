"""Resumable paper-native training for the hard star-graph IP gate."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import torch

from repro.ip_star_data import (
    SPLIT_SHA256,
    TEST_COUNT,
    TEST_SEED,
    TRAIN_COUNT,
    TRAIN_SEED,
    VALIDATION_COUNT,
    VALIDATION_SEED,
    StarBatch,
    StarGraph,
    collate_star_graphs,
    generate_graphs,
    graphs_sha256,
)
from repro.ip_star_eval import greedy_decode, planning_metrics
from repro.ip_star_model import (
    PAPER_DECODER_SPEC,
    PAPER_POSTERIOR_SPEC,
    InsertionDecoder,
    InsertionPosterior,
    ObjectiveOutput,
    TransformerSpec,
    fixed_order_objective,
    learned_ip_objective,
    random_ip_objective,
)
from repro.telemetry import TelemetryWriter, gpu_snapshot


Arm = Literal["fixed", "random", "learned"]


@dataclass(frozen=True)
class TrainSettings:
    arm: Arm
    output_dir: str
    epochs: int = 100
    batch_size: int = 64
    max_steps: int | None = None
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 1_000
    ema_decay: float = 0.999
    ema_start: int = 200
    seed: int = 42
    device: str = "cuda"
    bf16: bool = True
    log_every: int = 10
    eval_every: int = 100
    checkpoint_every: int = 100
    eval_limit: int = 100
    train_eval_limit: int = 0
    train_count: int = TRAIN_COUNT
    validation_count: int = VALIDATION_COUNT
    test_count: int = TEST_COUNT
    decoder_spec: TransformerSpec = PAPER_DECODER_SPEC
    posterior_spec: TransformerSpec = PAPER_POSTERIOR_SPEC
    resume: str | None = None

    @property
    def steps_per_epoch(self) -> int:
        return self.train_count // self.batch_size

    @property
    def paper_steps(self) -> int:
        return self.epochs * self.steps_per_epoch

    @property
    def stop_step(self) -> int:
        return min(self.max_steps or self.paper_steps, self.paper_steps)

    @property
    def fidelity_mode(self) -> bool:
        return self.paper_configuration and self.stop_step == self.paper_steps

    @property
    def paper_configuration(self) -> bool:
        return (
            self.epochs == 100
            and self.batch_size == 64
            and self.learning_rate == 1e-4
            and self.warmup_steps == 1_000
            and self.ema_decay == 0.999
            and self.ema_start == 200
            and self.train_count == TRAIN_COUNT
            and self.validation_count == VALIDATION_COUNT
            and self.test_count == TEST_COUNT
            and self.decoder_spec == PAPER_DECODER_SPEC
            and self.posterior_spec == PAPER_POSTERIOR_SPEC
        )

    def __post_init__(self) -> None:
        if self.arm not in ("fixed", "random", "learned"):
            raise ValueError(f"unknown arm: {self.arm}")
        if self.epochs < 1 or self.batch_size < 1 or self.train_count < self.batch_size:
            raise ValueError("epochs, batch size, and train count are inconsistent")
        if self.stop_step < 1:
            raise ValueError("training must include at least one step")


class ExponentialMovingAverage:
    def __init__(self, module: torch.nn.Module, decay: float, start_step: int):
        self.decay = float(decay)
        self.start_step = int(start_step)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in module.named_parameters()
        }
        self.initialized = False

    @torch.no_grad()
    def update(self, module: torch.nn.Module, step: int) -> None:
        if step < self.start_step:
            return
        if not self.initialized:
            for name, parameter in module.named_parameters():
                self.shadow[name].copy_(parameter)
            self.initialized = True
            return
        for name, parameter in module.named_parameters():
            self.shadow[name].lerp_(parameter, 1.0 - self.decay)

    @contextlib.contextmanager
    def average_parameters(self, module: torch.nn.Module):
        if not self.initialized:
            yield False
            return
        original = {
            name: parameter.detach().clone()
            for name, parameter in module.named_parameters()
        }
        try:
            with torch.no_grad():
                for name, parameter in module.named_parameters():
                    parameter.copy_(self.shadow[name])
            yield True
        finally:
            with torch.no_grad():
                for name, parameter in module.named_parameters():
                    parameter.copy_(original[name])

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "start_step": self.start_step,
            "initialized": self.initialized,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: dict) -> None:
        self.initialized = bool(state["initialized"])
        for name, value in state["shadow"].items():
            self.shadow[name].copy_(value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git(*arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments], check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _parameter_count(module: torch.nn.Module | None) -> int:
    return 0 if module is None else sum(parameter.numel() for parameter in module.parameters())


def _manifest(
    settings: TrainSettings,
    split_hashes: dict[str, str],
    decoder: InsertionDecoder,
    posterior: InsertionPosterior | None,
) -> dict:
    paper = Path("/home/ubuntu/papers/2606.02133/2606.02133v3.pdf")
    source = Path("/home/ubuntu/papers/2606.02133/2606.02133v3-source.tar")
    return {
        "schema_version": 1,
        "experiment": "paper-native insertion process star-graph planning",
        "arm": settings.arm,
        "paper": {
            "arxiv": "2606.02133v3",
            "pdf_sha256": _file_sha256(paper) if paper.exists() else None,
            "source_sha256": _file_sha256(source) if source.exists() else None,
            "reported_exact_match_percent": {"fixed": 24.0, "random": None, "learned": 83.0}[settings.arm],
            "arm_note": (
                "random is variable-length AO-IP; the paper's 26.5% star-graph baseline is fixed-canvas AO-ARM"
                if settings.arm == "random"
                else None
            ),
        },
        "upstream_task_source": {
            "repository": "https://github.com/dhruvdcoder/ILM",
            "commit": "6cc27f104fd926c8256aff28682c3fe66050ce77",
            "generator": "src/pcdd/datamodule/star_v2.py:asymmetric_variable_armlength_star_graph",
            "configuration": "configs/lightning_train/dataset/vstar_medium_v2.yaml",
        },
        "code": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        },
        "settings": asdict(settings),
        "paper_configuration": settings.paper_configuration,
        "fidelity_mode": settings.fidelity_mode,
        "data": {
            "split_hashes": split_hashes,
            "declared_full_split_hashes": SPLIT_SHA256,
            "train_seed": TRAIN_SEED,
            "validation_seed": VALIDATION_SEED,
            "test_seed": TEST_SEED,
            "drop_last": True,
        },
        "architecture": {
            "decoder_parameters": _parameter_count(decoder),
            "posterior_parameters": _parameter_count(posterior),
            "termination": "policy TERM location represented by appended EOS embedding",
            "order_posterior": "Plackett-Luce with Gumbel top-k and M=2 RLOO" if posterior else None,
            "attention": "bidirectional SDPA with RoPE",
        },
        "evaluation": {
            "decoding": "search-free greedy",
            "maximum_generated_tokens": 32,
            "primary": "held-out exact sequence match",
        },
        "environment": {
            "python": os.sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": settings.device,
        },
    }


def _batch_iterator(
    graphs: list[StarGraph], batch_size: int, seed: int, epoch: int
) -> Iterator[tuple[int, StarBatch]]:
    generator = torch.Generator().manual_seed(seed + 1_000_003 * epoch)
    permutation = torch.randperm(len(graphs), generator=generator)
    full_batches = len(graphs) // batch_size
    for batch_index in range(full_batches):
        indices = permutation[batch_index * batch_size : (batch_index + 1) * batch_size]
        yield batch_index, collate_star_graphs([graphs[int(index)] for index in indices])


def _lr_multiplier(step: int, warmup_steps: int, total_steps: int) -> float:
    if step <= warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _objective(
    arm: Arm,
    decoder: InsertionDecoder,
    posterior: InsertionPosterior | None,
    batch: StarBatch,
    generator: torch.Generator,
) -> ObjectiveOutput:
    if arm == "fixed":
        return fixed_order_objective(decoder, batch, generator)
    if arm == "random":
        return random_ip_objective(decoder, batch, generator)
    if posterior is None:
        raise RuntimeError("learned arm requires a posterior")
    return learned_ip_objective(decoder, posterior, batch, generator)


@torch.inference_mode()
def evaluate(
    decoder: InsertionDecoder,
    graphs: list[StarGraph],
    *,
    arm: Arm,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    count = 0
    started = time.perf_counter()
    for offset in range(0, len(graphs), batch_size):
        selected = graphs[offset : offset + batch_size]
        batch = collate_star_graphs(selected).to(device)
        decoded = greedy_decode(
            decoder,
            batch,
            mode="fixed" if arm == "fixed" else "insertion",
            max_tokens=32,
        )
        metrics = planning_metrics(decoded, selected)
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value * len(selected)
        count += len(selected)
    result = {name: value / count for name, value in totals.items()}
    result["examples"] = float(count)
    result["elapsed_s"] = time.perf_counter() - started
    result["examples_per_second"] = count / max(result["elapsed_s"], 1e-9)
    return result


def _save_checkpoint(
    path: Path,
    *,
    settings: TrainSettings,
    step: int,
    decoder: InsertionDecoder,
    posterior: InsertionPosterior | None,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage,
    objective_generator: torch.Generator,
) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "step": step,
            "settings": asdict(settings),
            "decoder": decoder.state_dict(),
            "posterior": posterior.state_dict() if posterior else None,
            "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict(),
            "objective_generator_rng": objective_generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        temporary,
    )
    os.replace(temporary, path)
    return _file_sha256(path)


def train(settings: TrainSettings) -> dict:
    seed_everything(settings.seed)
    output_dir = Path(settings.output_dir).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        settings.device
        if settings.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    train_graphs = generate_graphs(settings.train_count, TRAIN_SEED)
    validation_graphs = generate_graphs(settings.validation_count, VALIDATION_SEED)
    test_graphs = generate_graphs(settings.test_count, TEST_SEED)
    split_hashes = {
        "train": graphs_sha256(train_graphs),
        "validation": graphs_sha256(validation_graphs),
        "test": graphs_sha256(test_graphs),
    }
    if settings.paper_configuration and split_hashes != SPLIT_SHA256:
        raise RuntimeError(f"frozen split hash mismatch: {split_hashes} != {SPLIT_SHA256}")

    decoder = InsertionDecoder(settings.decoder_spec).to(device)
    posterior = (
        InsertionPosterior(settings.posterior_spec).to(device)
        if settings.arm == "learned"
        else None
    )
    parameters = list(decoder.parameters()) + (
        list(posterior.parameters()) if posterior else []
    )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    ema = ExponentialMovingAverage(decoder, settings.ema_decay, settings.ema_start)
    objective_generator = torch.Generator(device=device).manual_seed(settings.seed + 17)
    step = 0
    if settings.resume:
        state = torch.load(settings.resume, map_location=device, weights_only=False)
        decoder.load_state_dict(state["decoder"])
        if posterior is not None:
            posterior.load_state_dict(state["posterior"])
        optimizer.load_state_dict(state["optimizer"])
        ema.load_state_dict(state["ema"])
        objective_generator.set_state(state["objective_generator_rng"].cpu())
        step = int(state["step"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["cuda_rng"]])

    manifest = _manifest(settings, split_hashes, decoder, posterior)
    _atomic_json(output_dir / "manifest.json", manifest)
    telemetry_path = output_dir / "telemetry.jsonl"
    autocast_enabled = device.type == "cuda" and settings.bf16
    train_started = time.perf_counter()
    last_log_time = train_started
    last_log_step = step
    final_checkpoint: Path | None = None

    with TelemetryWriter(telemetry_path, output_dir.name) as telemetry:
        telemetry.write(
            "start",
            step=step,
            stop_step=settings.stop_step,
            paper_steps=settings.paper_steps,
            fidelity_mode=settings.fidelity_mode,
            split_hashes=split_hashes,
            gpu=gpu_snapshot(device),
        )
        start_epoch = step // settings.steps_per_epoch
        start_batch = step % settings.steps_per_epoch
        for epoch in range(start_epoch, settings.epochs):
            for batch_index, cpu_batch in _batch_iterator(
                train_graphs, settings.batch_size, settings.seed, epoch
            ):
                if epoch == start_epoch and batch_index < start_batch:
                    continue
                if step >= settings.stop_step:
                    break
                batch = cpu_batch.to(device)
                next_step = step + 1
                multiplier = _lr_multiplier(
                    next_step, settings.warmup_steps, settings.paper_steps
                )
                for group in optimizer.param_groups:
                    group["lr"] = settings.learning_rate * multiplier
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ):
                    output = _objective(
                        settings.arm, decoder, posterior, batch, objective_generator
                    )
                if not torch.isfinite(output.loss):
                    telemetry.write(
                        "fatal", step=step, reason="non_finite_loss", loss=float(output.loss)
                    )
                    raise FloatingPointError(f"non-finite loss at step {step}: {output.loss}")
                output.loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, float("inf"))
                if not torch.isfinite(gradient_norm):
                    telemetry.write(
                        "fatal", step=step, reason="non_finite_gradient", gradient_norm=float(gradient_norm)
                    )
                    raise FloatingPointError(f"non-finite gradient at step {step}")
                optimizer.step()
                step = next_step
                ema.update(decoder, step)

                if step == 1 or step % settings.log_every == 0:
                    now = time.perf_counter()
                    delta_steps = step - last_log_step
                    delta_time = now - last_log_time
                    telemetry.write(
                        "train",
                        step=step,
                        epoch=epoch,
                        batch_index=batch_index,
                        loss=float(output.loss.detach()),
                        learning_rate=optimizer.param_groups[0]["lr"],
                        gradient_norm=float(gradient_norm),
                        steps_per_second=delta_steps / max(delta_time, 1e-9),
                        examples_per_second=delta_steps * settings.batch_size / max(delta_time, 1e-9),
                        metrics={name: float(value) for name, value in output.metrics.items()},
                        gpu=gpu_snapshot(device),
                    )
                    last_log_time = now
                    last_log_step = step

                if step % settings.eval_every == 0 or step == settings.stop_step:
                    selected_validation = validation_graphs[: settings.eval_limit]
                    raw_validation = evaluate(
                        decoder,
                        selected_validation,
                        arm=settings.arm,
                        device=device,
                        batch_size=settings.batch_size,
                    )
                    telemetry.write(
                        "evaluation",
                        step=step,
                        split="validation",
                        weights="raw",
                        **raw_validation,
                    )
                    if settings.train_eval_limit:
                        raw_train = evaluate(
                            decoder,
                            train_graphs[: settings.train_eval_limit],
                            arm=settings.arm,
                            device=device,
                            batch_size=settings.batch_size,
                        )
                        telemetry.write(
                            "evaluation",
                            step=step,
                            split="train",
                            weights="raw",
                            **raw_train,
                        )
                    with ema.average_parameters(decoder) as using_ema:
                        if using_ema:
                            ema_validation = evaluate(
                                decoder,
                                selected_validation,
                                arm=settings.arm,
                                device=device,
                                batch_size=settings.batch_size,
                            )
                            telemetry.write(
                                "evaluation",
                                step=step,
                                split="validation",
                                weights="ema",
                                **ema_validation,
                            )
                            if settings.train_eval_limit:
                                ema_train = evaluate(
                                    decoder,
                                    train_graphs[: settings.train_eval_limit],
                                    arm=settings.arm,
                                    device=device,
                                    batch_size=settings.batch_size,
                                )
                                telemetry.write(
                                    "evaluation",
                                    step=step,
                                    split="train",
                                    weights="ema",
                                    **ema_train,
                                )

                if step % settings.checkpoint_every == 0 or step == settings.stop_step:
                    final_checkpoint = checkpoint_dir / f"step_{step:08d}.pt"
                    checksum = _save_checkpoint(
                        final_checkpoint,
                        settings=settings,
                        step=step,
                        decoder=decoder,
                        posterior=posterior,
                        optimizer=optimizer,
                        ema=ema,
                        objective_generator=objective_generator,
                    )
                    telemetry.write(
                        "checkpoint", step=step, path=str(final_checkpoint), sha256=checksum
                    )
            if step >= settings.stop_step:
                break

        elapsed = time.perf_counter() - train_started
        result = {
            "step": step,
            "elapsed_s": elapsed,
            "examples_per_second": step * settings.batch_size / max(elapsed, 1e-9),
            "checkpoint": str(final_checkpoint) if final_checkpoint else None,
            "checkpoint_sha256": _file_sha256(final_checkpoint) if final_checkpoint else None,
        }
        telemetry.write("complete", **result)
        _atomic_json(output_dir / "result.json", result)
    return result


def parse_args() -> TrainSettings:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("fixed", "random", "learned"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--eval-limit", type=int, default=100)
    parser.add_argument("--train-eval-limit", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=TRAIN_COUNT)
    parser.add_argument("--validation-count", type=int, default=VALIDATION_COUNT)
    parser.add_argument("--test-count", type=int, default=TEST_COUNT)
    parser.add_argument("--decoder-width", type=int, default=PAPER_DECODER_SPEC.width)
    parser.add_argument("--decoder-layers", type=int, default=PAPER_DECODER_SPEC.layers)
    parser.add_argument("--decoder-heads", type=int, default=PAPER_DECODER_SPEC.heads)
    parser.add_argument("--posterior-width", type=int, default=PAPER_POSTERIOR_SPEC.width)
    parser.add_argument("--posterior-layers", type=int, default=PAPER_POSTERIOR_SPEC.layers)
    parser.add_argument("--posterior-heads", type=int, default=PAPER_POSTERIOR_SPEC.heads)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--resume")
    arguments = parser.parse_args()
    return TrainSettings(
        arm=arguments.arm,
        output_dir=arguments.output_dir,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        max_steps=arguments.max_steps,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        warmup_steps=arguments.warmup_steps,
        seed=arguments.seed,
        device=arguments.device,
        bf16=not arguments.no_bf16,
        log_every=arguments.log_every,
        eval_every=arguments.eval_every,
        checkpoint_every=arguments.checkpoint_every,
        eval_limit=arguments.eval_limit,
        train_eval_limit=arguments.train_eval_limit,
        train_count=arguments.train_count,
        validation_count=arguments.validation_count,
        test_count=arguments.test_count,
        decoder_spec=TransformerSpec(
            width=arguments.decoder_width,
            layers=arguments.decoder_layers,
            heads=arguments.decoder_heads,
            dropout=arguments.dropout,
        ),
        posterior_spec=TransformerSpec(
            width=arguments.posterior_width,
            layers=arguments.posterior_layers,
            heads=arguments.posterior_heads,
            dropout=arguments.dropout,
        ),
        resume=arguments.resume,
    )


def main() -> None:
    result = train(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
