"""Loading of the checked-in Hydra Sudoku configs, without Hydra.

Only the plain, interpolation-free blocks are read (``model``, ``optim``,
``apmdm``, ``lr_scheduler``, ``sampling``, ``seed``, ``time_conditioning``,
``trainer.max_steps``, ``loader.global_batch_size``), which is everything the
reproduction needs.  The historical ``train/configs/sudoku.yaml`` is read but
never written: the paper-faithful settings live in a separately named file.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict

import yaml

from repro.hashing import sha256_file
from repro.paths import train_config_dir

HISTORICAL_CONFIG = "sudoku.yaml"
PAPER_CONFIG = "sudoku_paper.yaml"


@dataclass(frozen=True)
class ModelSpec:
    """Architecture fields that determine the parameter count and shapes."""

    vocab_size: int
    length: int
    hidden_size: int
    n_heads: int
    n_blocks: int
    cond_dim: int
    dropout: float
    mlp_ratio: int = 4
    tie_word_embeddings: bool = False
    scale_by_sigma: bool = True

    def signature(self) -> str:
        return (
            f"apmdm-encoder|L={self.n_blocks}|H={self.n_heads}|d={self.hidden_size}"
            f"|ffn={self.mlp_ratio * self.hidden_size}|cond={self.cond_dim}"
            f"|V={self.vocab_size}|maxlen={self.length}|heads=4(unmask,remask,insert,delete)"
        )

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OptimSpec:
    lr: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SudokuConfig:
    """A loaded Sudoku configuration."""

    name: str
    path: Path
    sha256: str
    raw: Dict[str, Any]
    model: ModelSpec
    optim: OptimSpec
    lambda_remask: float
    lambda_expand: float
    lambda_contract: float
    remasking_threshold: float
    expansion_threshold: float
    contraction_threshold: float
    global_batch_size: int
    max_steps: int
    warmup_steps: int
    gradient_clip_val: float
    precision: str
    seed: int
    time_conditioning: bool
    sampling_max_steps: int
    train_ratio: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path.name,
            "sha256": self.sha256,
            "model": self.model.as_dict(),
            "model_signature": self.model.signature(),
            "optim": self.optim.as_dict(),
            "loss_weights": {
                "lambda_remask": self.lambda_remask,
                "lambda_expand": self.lambda_expand,
                "lambda_contract": self.lambda_contract,
            },
            "thresholds": {
                "remask": self.remasking_threshold,
                "insert": self.expansion_threshold,
                "delete": self.contraction_threshold,
            },
            "global_batch_size": self.global_batch_size,
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "gradient_clip_val": self.gradient_clip_val,
            "precision": self.precision,
            "seed": self.seed,
            "time_conditioning": self.time_conditioning,
            "sampling_max_steps": self.sampling_max_steps,
            "train_ratio": self.train_ratio,
        }


def _get(mapping: Dict[str, Any], path: str, default: Any = None) -> Any:
    node: Any = mapping
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def load_sudoku_config(filename: str = PAPER_CONFIG, directory: Path | None = None) -> SudokuConfig:
    """Load and normalise one of the checked-in Sudoku YAML configs."""
    root = Path(directory) if directory is not None else train_config_dir()
    path = root / filename
    raw = yaml.safe_load(path.read_text())

    model_block = raw.get("model", {})
    model = ModelSpec(
        vocab_size=int(model_block["vocab_size"]),
        length=int(model_block["length"]),
        hidden_size=int(model_block["hidden_size"]),
        n_heads=int(model_block["n_heads"]),
        n_blocks=int(model_block["n_blocks"]),
        cond_dim=int(model_block["cond_dim"]),
        dropout=float(model_block.get("dropout", 0.1)),
        mlp_ratio=int(model_block.get("mlp_ratio", 4)),
        tie_word_embeddings=bool(model_block.get("tie_word_embeddings", False)),
        scale_by_sigma=bool(model_block.get("scale_by_sigma", True)),
    )
    optim_block = raw.get("optim", {})
    optim = OptimSpec(
        lr=float(optim_block["lr"]),
        beta1=float(optim_block["beta1"]),
        beta2=float(optim_block["beta2"]),
        eps=float(optim_block.get("eps", 1e-8)),
        weight_decay=float(optim_block["weight_decay"]),
    )
    return SudokuConfig(
        name=path.stem,
        path=path,
        sha256=sha256_file(path),
        raw=raw,
        model=model,
        optim=optim,
        lambda_remask=float(_get(raw, "apmdm.lambda_remask", 1.0)),
        lambda_expand=float(_get(raw, "apmdm.lambda_expand", 1.0)),
        lambda_contract=float(_get(raw, "apmdm.lambda_contract", 1.0)),
        remasking_threshold=float(_get(raw, "apmdm.remasking_threshold", 0.5)),
        expansion_threshold=float(_get(raw, "apmdm.expansion_threshold", 0.5)),
        contraction_threshold=float(_get(raw, "apmdm.contraction_threshold", 0.5)),
        global_batch_size=int(_get(raw, "loader.global_batch_size", 256)),
        max_steps=int(_get(raw, "trainer.max_steps", 1_000_000)),
        warmup_steps=int(_get(raw, "lr_scheduler.num_warmup_steps", 250)),
        gradient_clip_val=float(_get(raw, "trainer.gradient_clip_val", 1.0)),
        precision=str(_get(raw, "trainer.precision", "bf16")),
        seed=int(_get(raw, "seed", 42)),
        time_conditioning=bool(_get(raw, "time_conditioning", False)),
        sampling_max_steps=int(_get(raw, "sampling.max_steps", 50)),
        train_ratio=float(_get(raw, "data.train_ratio", 0.99)),
    )


def load_historical_config(directory: Path | None = None) -> SudokuConfig:
    """Load the historical ``train/configs/sudoku.yaml`` (read-only)."""
    return load_sudoku_config(HISTORICAL_CONFIG, directory)


def load_paper_config(directory: Path | None = None) -> SudokuConfig:
    """Load the paper-faithful ``train/configs/sudoku_paper.yaml``."""
    return load_sudoku_config(PAPER_CONFIG, directory)
