"""Resumable, sealed GuacaMol training for FO-ARM, AO-IP and learned IP."""

from __future__ import annotations

import argparse
import contextlib
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

from repro.ip_chem_data import (
    OFFICIAL_SPLITS,
    SmilesTokenizer,
    authenticate_split,
    collate_smiles,
    file_digest,
    read_smiles,
)
from repro.ip_chem_eval import ChemDecodedBatch, generation_metrics, sample_decode
from repro.ip_chem_model import (
    CHEM_DECODER_SPEC,
    CHEM_POSTERIOR_SPEC,
    ChemInsertionDecoder,
    ChemInsertionPosterior,
    TransformerSpec,
    fixed_order_objective,
    learned_ip_objective,
    random_ip_objective,
)
from repro.ip_star_train import ExponentialMovingAverage
from repro.telemetry import TelemetryWriter, gpu_snapshot


Arm = Literal["fixed", "random", "learned"]


@dataclass(frozen=True)
class ChemTrainSettings:
    arm: Arm
    data_dir: str
    output_dir: str
    max_steps: int = 20_000
    batch_size: int = 64
    decoder_learning_rate: float = 5e-4
    posterior_learning_rate: float = 5e-6
    weight_decay: float = 1e-12
    warmup_steps: int = 1_000
    ema_decay: float = 0.9999
    seed: int = 42
    device: str = "cuda"
    bf16: bool = True
    log_every: int = 10
    eval_every: int = 1_000
    checkpoint_every: int = 1_000
    eval_samples: int = 256
    final_eval_samples: int = 1_000
    decoder_spec: TransformerSpec = CHEM_DECODER_SPEC
    posterior_spec: TransformerSpec = CHEM_POSTERIOR_SPEC
    resume: str | None = None

    def __post_init__(self) -> None:
        if self.arm not in ("fixed", "random", "learned"):
            raise ValueError(f"unknown chemistry arm: {self.arm}")
        if min(self.max_steps, self.batch_size, self.eval_samples) < 1:
            raise ValueError("step, batch, and evaluation counts must be positive")


def _git(*arguments: str) -> str | None:
    result = subprocess.run(["git", *arguments], check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return file_digest(path)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parameter_count(module: torch.nn.Module | None) -> int:
    return 0 if module is None else sum(parameter.numel() for parameter in module.parameters())


def _manifest(
    settings: ChemTrainSettings,
    tokenizer: SmilesTokenizer,
    split_records: dict[str, dict[str, object]],
    decoder: ChemInsertionDecoder,
    posterior: ChemInsertionPosterior | None,
) -> dict:
    paper_pdf = Path("/home/ubuntu/papers/2606.02133/2606.02133v3.pdf")
    paper_source = Path("/home/ubuntu/papers/2606.02133/2606.02133v3-source.tar")
    return {
        "schema": "ip/guacamol-reconstruction-v1",
        "experiment": "GuacaMol classifier-terminated insertion-process reconstruction",
        "arm": settings.arm,
        "paper": {
            "arxiv": "2606.02133v3",
            "pdf_sha256": file_digest(paper_pdf) if paper_pdf.exists() else None,
            "source_sha256": file_digest(paper_source) if paper_source.exists() else None,
            "reported_percent": {
                "fixed": {"validity": 98.5, "valid_unique": 98.3, "valid_unique_novel": 80.4, "kl": 99.4, "fcd": 90.3},
                "random": {"validity": 88.2, "valid_unique": 88.2, "valid_unique_novel": 87.7, "kl": 95.3, "fcd": 79.1},
                "learned": {"validity": 97.4, "valid_unique": 97.3, "valid_unique_novel": 95.6, "kl": 97.2, "fcd": 89.2},
            }[settings.arm],
            "omitted_configuration": [
                "decoder width", "decoder heads", "posterior width", "posterior heads",
                "batch size", "tokenizer", "maximum sequence length", "steps/epochs",
                "random seeds", "hardware",
            ],
        },
        "code": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
            "source_files": {
                name: file_digest(Path(name))
                for name in (
                    "repro/ip_chem_data.py", "repro/ip_chem_model.py",
                    "repro/ip_chem_eval.py", "repro/ip_chem_train.py",
                    "ops/run_ip_chem_runpod.sh",
                )
            },
        },
        "settings": asdict(settings),
        "data": {
            "upstream": "BenevolentAI/guacamol@60ebe1f6a396f16e08b834dce448e9343d259feb",
            "splits": split_records,
            "official": OFFICIAL_SPLITS,
            "tokenizer_tokens": list(tokenizer.tokens),
            "tokenizer_sha256": tokenizer.sha256,
            "maximum_observed_token_length": 100,
        },
        "architecture": {
            "decoder_parameters": _parameter_count(decoder),
            "posterior_parameters": _parameter_count(posterior),
            "termination": "classifier EOS content forced to final insertion slot",
            "objective": "permutation ELBO; M=2 RLOO for learned posterior",
            "declared_dimensions_not_from_paper": True,
        },
        "evaluation": {
            "decoding": "search-free ancestral sampling, temperature 1",
            "smoke_samples": settings.eval_samples,
            "final_samples": settings.final_eval_samples,
            "full_gate_samples": 10_000,
        },
        "environment": {
            "python": os.sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": settings.device,
            "rdkit": __import__("rdkit").__version__,
        },
    }


def _batches(rows: list[str], batch_size: int, seed: int, epoch: int) -> Iterator[list[str]]:
    generator = torch.Generator().manual_seed(seed + 1_000_003 * epoch)
    permutation = torch.randperm(len(rows), generator=generator)
    for offset in range(0, len(rows) - batch_size + 1, batch_size):
        yield [rows[int(index)] for index in permutation[offset : offset + batch_size]]


def _lr_multiplier(step: int, warmup: int, total: int) -> float:
    if step <= warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def _objective(
    settings: ChemTrainSettings,
    decoder: ChemInsertionDecoder,
    posterior: ChemInsertionPosterior | None,
    batch,
    generator: torch.Generator,
):
    if settings.arm == "fixed":
        return fixed_order_objective(decoder, batch, generator)
    if settings.arm == "random":
        return random_ip_objective(decoder, batch, generator)
    if posterior is None:
        raise RuntimeError("learned chemistry arm requires its order posterior")
    return learned_ip_objective(decoder, posterior, batch, generator)


@torch.inference_mode()
def evaluate(
    decoder: ChemInsertionDecoder,
    *,
    arm: Arm,
    samples: int,
    batch_size: int,
    seed: int,
    training_smiles: set[str],
) -> tuple[dict[str, float], list[dict[str, object]]]:
    device = next(decoder.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    sequences: list[tuple[int, ...]] = []
    traces: list[tuple[int, ...]] = []
    terminated: list[bool] = []
    steps = 0
    started = time.perf_counter()
    while len(sequences) < samples:
        count = min(batch_size, samples - len(sequences))
        decoded = sample_decode(
            decoder,
            count,
            mode="fixed" if arm == "fixed" else "insertion",
            generator=generator,
            max_tokens=100,
        )
        sequences.extend(decoded.token_sequences)
        traces.extend(decoded.insertion_traces)
        terminated.extend(decoded.terminated)
        steps = max(steps, decoded.steps)
    combined = ChemDecodedBatch(tuple(sequences), tuple(traces), tuple(terminated), steps)
    metrics, rows = generation_metrics(combined, decoder.tokenizer, training_smiles)
    metrics["wall_seconds"] = time.perf_counter() - started
    metrics["examples_per_second"] = samples / max(metrics["wall_seconds"], 1e-9)
    return metrics, rows


def _save_checkpoint(
    path: Path,
    *,
    settings: ChemTrainSettings,
    step: int,
    decoder: ChemInsertionDecoder,
    posterior: ChemInsertionPosterior | None,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage,
    objective_generator: torch.Generator,
) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema": "ip/guacamol-checkpoint-v1",
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
    return file_digest(path)


def train(settings: ChemTrainSettings) -> dict:
    _seed(settings.seed)
    output = Path(settings.output_dir).resolve()
    checkpoints = output / "checkpoints"
    evaluations = output / "evaluations"
    checkpoints.mkdir(parents=True, exist_ok=True)
    evaluations.mkdir(parents=True, exist_ok=True)
    data_dir = Path(settings.data_dir)
    split_paths = {
        split: data_dir / record["filename"] for split, record in OFFICIAL_SPLITS.items()
    }
    split_records = {
        split: authenticate_split(path, split) for split, path in split_paths.items()
    }
    tokenizer = SmilesTokenizer.from_training_file(split_paths["train"])
    train_rows = read_smiles(split_paths["train"])
    training_set = set(train_rows)
    device = torch.device(
        settings.device if settings.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    decoder = ChemInsertionDecoder(tokenizer, settings.decoder_spec).to(device)
    posterior = (
        ChemInsertionPosterior(tokenizer, settings.posterior_spec).to(device)
        if settings.arm == "learned"
        else None
    )
    groups = [{"params": decoder.parameters(), "lr": settings.decoder_learning_rate}]
    if posterior is not None:
        groups.append({"params": posterior.parameters(), "lr": settings.posterior_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=settings.weight_decay)
    ema = ExponentialMovingAverage(decoder, settings.ema_decay, 0)
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
    manifest = _manifest(settings, tokenizer, split_records, decoder, posterior)
    _atomic_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    telemetry_path = output / "telemetry.jsonl"
    with TelemetryWriter(telemetry_path, output.name) as telemetry:
        epoch = step // max(len(train_rows) // settings.batch_size, 1)
        while step < settings.max_steps:
            for selected in _batches(train_rows, settings.batch_size, settings.seed, epoch):
                if step >= settings.max_steps:
                    break
                batch = collate_smiles(selected, tokenizer).to(device)
                autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if settings.bf16 and device.type == "cuda"
                    else contextlib.nullcontext()
                )
                with autocast:
                    objective = _objective(settings, decoder, posterior, batch, objective_generator)
                objective.loss.backward()
                parameters = list(decoder.parameters()) + (
                    list(posterior.parameters()) if posterior is not None else []
                )
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                step += 1
                multiplier = _lr_multiplier(step, settings.warmup_steps, settings.max_steps)
                for index, group in enumerate(optimizer.param_groups):
                    base = settings.decoder_learning_rate if index == 0 else settings.posterior_learning_rate
                    group["lr"] = base * multiplier
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(decoder, step)
                if not torch.isfinite(objective.loss):
                    raise FloatingPointError(f"non-finite chemistry loss at step {step}")
                if step % settings.log_every == 0 or step == 1:
                    elapsed = time.perf_counter() - started
                    telemetry.write(
                        "train",
                        step=step,
                        epoch=epoch,
                        loss=float(objective.loss.detach()),
                        grad_norm=float(grad_norm),
                        learning_rates=[group["lr"] for group in optimizer.param_groups],
                        steps_per_second=step / max(elapsed, 1e-9),
                        examples_per_second=step * settings.batch_size / max(elapsed, 1e-9),
                        metrics={name: float(value) for name, value in objective.metrics.items()},
                        gpu=gpu_snapshot(device),
                    )
                if step % settings.eval_every == 0 or step == settings.max_steps:
                    with ema.average_parameters(decoder):
                        metrics, rows = evaluate(
                            decoder,
                            arm=settings.arm,
                            samples=settings.eval_samples,
                            batch_size=min(settings.batch_size, settings.eval_samples),
                            seed=settings.seed + 100_000 + step,
                            training_smiles=training_set,
                        )
                    rows_path = evaluations / f"samples_step_{step:08d}.jsonl"
                    rows_sha256 = _write_jsonl(rows_path, rows)
                    telemetry.write(
                        "evaluation", step=step, weights="ema", rows_path=str(rows_path),
                        rows_sha256=rows_sha256, **metrics
                    )
                if step % settings.checkpoint_every == 0 or step == settings.max_steps:
                    path = checkpoints / f"step_{step:08d}.pt"
                    digest = _save_checkpoint(
                        path,
                        settings=settings,
                        step=step,
                        decoder=decoder,
                        posterior=posterior,
                        optimizer=optimizer,
                        ema=ema,
                        objective_generator=objective_generator,
                    )
                    telemetry.write("checkpoint", step=step, path=str(path), sha256=digest)
            epoch += 1
        with ema.average_parameters(decoder):
            final_metrics, final_rows = evaluate(
                decoder,
                arm=settings.arm,
                samples=settings.final_eval_samples,
                batch_size=min(settings.batch_size, settings.final_eval_samples),
                seed=settings.seed + 2_000_000,
                training_smiles=training_set,
            )
        final_rows_path = evaluations / "final_samples.jsonl"
        final_rows_sha256 = _write_jsonl(final_rows_path, final_rows)
        completion = {
            "schema": "ip/guacamol-completion-v1",
            "completed": True,
            "arm": settings.arm,
            "step": step,
            "metrics": final_metrics,
            "samples": {"path": str(final_rows_path), "sha256": final_rows_sha256},
            "checkpoint": {
                "path": str(checkpoints / f"step_{step:08d}.pt"),
                "sha256": file_digest(checkpoints / f"step_{step:08d}.pt"),
            },
            "manifest": {"path": str(output / "manifest.json"), "sha256": file_digest(output / "manifest.json")},
            "telemetry": {"path": str(telemetry_path)},
            "elapsed_s": time.perf_counter() - started,
        }
        _atomic_json(output / "completion.json", completion)
        telemetry.write("complete", **completion)
    return completion


def _spec(prefix: str, args: argparse.Namespace) -> TransformerSpec:
    return TransformerSpec(
        width=getattr(args, f"{prefix}_width"),
        layers=getattr(args, f"{prefix}_layers"),
        heads=getattr(args, f"{prefix}_heads"),
        dropout=getattr(args, f"{prefix}_dropout"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=("fixed", "random", "learned"))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--checkpoint-every", type=int, default=1_000)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--final-eval-samples", type=int, default=1_000)
    parser.add_argument("--resume")
    for prefix, spec in (("decoder", CHEM_DECODER_SPEC), ("posterior", CHEM_POSTERIOR_SPEC)):
        parser.add_argument(f"--{prefix}-width", type=int, default=spec.width)
        parser.add_argument(f"--{prefix}-layers", type=int, default=spec.layers)
        parser.add_argument(f"--{prefix}-heads", type=int, default=spec.heads)
        parser.add_argument(f"--{prefix}-dropout", type=float, default=spec.dropout)
    args = parser.parse_args()
    settings = ChemTrainSettings(
        arm=args.arm,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        bf16=not args.no_bf16,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        eval_samples=args.eval_samples,
        final_eval_samples=args.final_eval_samples,
        decoder_spec=_spec("decoder", args),
        posterior_spec=_spec("posterior", args),
        resume=args.resume,
    )
    print(json.dumps(train(settings), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
