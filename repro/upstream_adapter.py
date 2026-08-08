"""Minimal adapter around the released AP-MDM Sudoku implementation.

Training remains in ``train/main.py`` / ``train/diffusion.py``.  This module
does not reimplement the released backbone or losses; it only gives the
official ``DIT`` the one-argument interface used by the audited conditional
Sudoku sampler and records the distinction between the configured and
effective vocabulary sizes.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf

from repro.hashing import sha256_file
from repro.paths import repo_root, train_config_dir

UPSTREAM_COMMIT = "836b79b30286301ee1cca975db7e8468bc2a653e"
UPSTREAM_SOURCE_HASHES = {
    "train/models/dit.py": "e407b4fc2d64cd33e11d84f1410a91cd2b51a7c000028b54d8694ad6dbf92ccb",
    "train/models/__init__.py": "c48cfb3e47b4cdf66e473d15ce9a73833da039b4889c36bcaa4d17ec5acf6c42",
    "train/dataloader.py": "55e9fff0e8bffac949cec9c0749de334fdad0feba7973de6e24f7d24b7df6ebc",
    "train/apmdm_dataloader.py": "6048e77d5f9866123a72ccb3df68a252d633f697207d6f24fdf7b3a2abe3de3c",
    "train/diffusion.py": "4c3132f82d80a009cc4fa3afc383a8fca954155b72b0b81965e7c92fc60a0275",
    "dataset/sudoku/sudoku_generator.py": "454e6dff3348e4f12dbdb6e7ca8fe6d6ab01e5f6340960bf09375d90c11225ed",
}


def _install_train_path() -> Path:
    """Expose the released ``train`` modules without copying their code."""
    path = repo_root() / "train"
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
    return path


def _load_source_module(name: str, path: Path):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _upstream_modules():
    train_path = _install_train_path()
    # Loading ``models.dit`` through the released package also imports the
    # unrelated autoregressive backend, which hard-requires FlashAttention.
    # Load the two exact source files instead; no code is copied or altered.
    models = _load_source_module("_apmdm_upstream_dit", train_path / "models/dit.py")
    tokenizer_module = _load_source_module(
        "_apmdm_upstream_dataloader", train_path / "apmdm_dataloader.py"
    )
    model_path = Path(models.__file__).resolve()
    if train_path not in model_path.parents:
        raise RuntimeError(f"imported a non-upstream models.dit: {model_path}")
    return models, tokenizer_module


@dataclass(frozen=True)
class UpstreamArchitecture:
    config_name: str
    config_path: str
    config_sha256: str
    configured_vocab_size: int
    effective_vocab_size: int
    parameters: int
    n_blocks: int
    n_heads: int
    hidden_size: int
    cond_dim: int
    max_length: int
    attention_backend: str

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class OfficialBackboneAdapter(nn.Module):
    """Zero-time-conditioning wrapper around the released ``DIT`` backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        max_length: int,
        configured_vocab_size: int,
        effective_vocab_size: int,
        config_name: str,
        attention_backend: str,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.max_length = int(max_length)
        self.vocab_size = int(effective_vocab_size)
        self.configured_vocab_size = int(configured_vocab_size)
        self.config_name = config_name
        self.attention_backend = attention_backend

    def forward(self, indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        if indices.ndim != 2:
            raise ValueError("indices must be (batch, sequence)")
        if indices.shape[1] > self.max_length:
            raise ValueError(
                f"sequence length {indices.shape[1]} exceeds {self.max_length}"
            )
        if indices.numel() and (
            int(indices.min()) < 0 or int(indices.max()) >= self.vocab_size
        ):
            raise ValueError(f"token id outside [0, {self.vocab_size})")
        # ``Diffusion._process_sigma`` squeezes the released training signal to
        # one dimension before it reaches DIT.  The adapter calls DIT directly.
        sigma = torch.zeros((indices.shape[0],), device=indices.device)
        return self.backbone(indices, sigma)

    def architecture_signature(self) -> str:
        cfg = self.backbone.config.model
        return (
            f"upstream-apmdm|config={self.config_name}|L={cfg.n_blocks}"
            f"|H={cfg.n_heads}|d={cfg.hidden_size}|cond={cfg.cond_dim}"
            f"|configuredV={self.configured_vocab_size}|effectiveV={self.vocab_size}"
            f"|maxlen={self.max_length}|attention={self.attention_backend}"
        )


def _load_raw_config(config_name: str) -> tuple[Path, dict]:
    name = config_name if config_name.endswith(".yaml") else f"{config_name}.yaml"
    path = train_config_dir() / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, yaml.safe_load(path.read_text())


def build_upstream_adapter(
    config_name: str,
    vocab_cache_path: str | Path,
    *,
    seed: int = 42,
    device: Optional[torch.device | str] = None,
) -> OfficialBackboneAdapter:
    """Instantiate the released backbone with its *actual* tokenizer size."""
    models, tokenizer_module = _upstream_modules()
    path, raw = _load_raw_config(config_name)
    vocab_path = Path(vocab_cache_path).expanduser().resolve()
    if not vocab_path.is_file():
        raise FileNotFoundError(vocab_path)

    tokenizer = tokenizer_module.APMDMTokenizer(vocab_file=str(vocab_path))
    torch.manual_seed(seed)
    backbone = models.DIT(OmegaConf.create(raw), vocab_size=tokenizer.vocab_size)
    attention_backend = "flash_attn" if models.FLASH_ATTN_AVAILABLE else "torch_sdpa"
    adapter = OfficialBackboneAdapter(
        backbone,
        max_length=int(raw["model"]["length"]),
        configured_vocab_size=int(raw["model"]["vocab_size"]),
        effective_vocab_size=int(tokenizer.vocab_size),
        config_name=path.stem,
        attention_backend=attention_backend,
    )
    if device is not None:
        adapter.to(device)
    return adapter


def load_lightning_checkpoint(
    adapter: OfficialBackboneAdapter,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load only ``backbone.*`` tensors from a released Lightning checkpoint."""
    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location=map_location, weights_only=False)
    state = payload.get("state_dict", payload)
    prefix = "backbone."
    backbone_state = {
        key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)
    }
    if not backbone_state:
        raise ValueError(f"checkpoint has no {prefix!r} tensors: {path}")
    adapter.backbone.load_state_dict(backbone_state, strict=True)
    return payload


def architecture_report(vocab_cache_path: str | Path) -> dict:
    """Report both released configurations using the effective vocabulary."""
    models, _ = _upstream_modules()
    rows = {}
    for config_name in ("sudoku", "sudoku_paper"):
        path, raw = _load_raw_config(config_name)
        adapter = build_upstream_adapter(config_name, vocab_cache_path, device="cpu")
        rows[config_name] = UpstreamArchitecture(
            config_name=config_name,
            config_path=str(path),
            config_sha256=sha256_file(path),
            configured_vocab_size=int(raw["model"]["vocab_size"]),
            effective_vocab_size=adapter.vocab_size,
            parameters=sum(parameter.numel() for parameter in adapter.backbone.parameters()),
            n_blocks=int(raw["model"]["n_blocks"]),
            n_heads=int(raw["model"]["n_heads"]),
            hidden_size=int(raw["model"]["hidden_size"]),
            cond_dim=int(raw["model"]["cond_dim"]),
            max_length=int(raw["model"]["length"]),
            attention_backend=adapter.attention_backend,
        ).as_dict()
    return {
        "upstream_repository": "https://github.com/chr26195/AP-MDM",
        "upstream_commit": UPSTREAM_COMMIT,
        "compatibility_changes": [
            "FlashAttention import falls back to torch SDPA; tensors and modules are unchanged",
            "the unused autoregressive backend is not eagerly imported when FlashAttention is absent",
            "dataset factory forwards the configured vocabulary path",
            "streaming chunk filenames include the split to prevent validation overwriting training",
        ],
        "upstream_source_hashes": UPSTREAM_SOURCE_HASHES,
        "arm_source_hashes": {
            "train/models/dit.py": sha256_file(repo_root() / "train/models/dit.py"),
            "train/models/__init__.py": sha256_file(repo_root() / "train/models/__init__.py"),
            "train/dataloader.py": sha256_file(repo_root() / "train/dataloader.py"),
            "train/diffusion.py": sha256_file(repo_root() / "train/diffusion.py"),
            "train/apmdm_dataloader.py": sha256_file(repo_root() / "train/apmdm_dataloader.py"),
            "dataset/sudoku/sudoku_generator.py": sha256_file(
                repo_root() / "dataset/sudoku/sudoku_generator.py"
            ),
        },
        "architectures": rows,
        "paper_reported_parameters": 1_200_000,
        "paper_reported_vocab_size": 31,
        "note": (
            "The released tokenizer adds BOS and PAD to the 32-token generated "
            "vocabulary, so the model actually instantiates with 34 tokens."
        ),
    }


def write_architecture_report(vocab_cache_path: str | Path, output: str | Path) -> dict:
    report = architecture_report(vocab_cache_path)
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report
