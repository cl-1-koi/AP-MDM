"""Compatibility gates for the released AP-MDM implementation."""

from __future__ import annotations

import importlib
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from repro.config import load_historical_config
from repro.model import build_model
from repro.official import sample_generator
from repro.upstream_adapter import (
    _install_train_path,
    architecture_report,
    build_upstream_adapter,
    load_lightning_checkpoint,
)


def _vocab_cache(tmp_path):
    path = tmp_path / "vocab_cache.pkl"
    with path.open("wb") as handle:
        pickle.dump(sample_generator().create_vocabulary(), handle)
    return path


def test_released_architectures_use_effective_34_token_vocabulary(tmp_path):
    report = architecture_report(_vocab_cache(tmp_path))
    historical = report["architectures"]["sudoku"]
    paper = report["architectures"]["sudoku_paper"]

    assert historical["configured_vocab_size"] == 31
    assert historical["effective_vocab_size"] == 34
    assert historical["parameters"] == 1_036_197
    assert paper["configured_vocab_size"] == 31
    assert paper["effective_vocab_size"] == 34
    assert paper["parameters"] == 6_052_133
    assert paper["parameters"] != report["paper_reported_parameters"]


def test_sudoku_dit_package_import_does_not_require_unused_ar_backend():
    _install_train_path()
    models = importlib.import_module("models")
    assert models.dit is not None
    assert models.dit.FLASH_ATTN_AVAILABLE is False
    assert models.autoregressive is None


def test_sdpa_fallback_is_tensor_equivalent_to_audited_backbone(tmp_path):
    vocab_path = _vocab_cache(tmp_path)
    upstream = build_upstream_adapter("sudoku", vocab_path, seed=7).eval()

    historical = load_historical_config()
    portable = build_model(
        replace(historical.model, vocab_size=upstream.vocab_size),
        time_conditioning=False,
        seed=9,
    ).eval()
    missing, unexpected = portable.load_state_dict(
        upstream.backbone.state_dict(), strict=False
    )
    assert missing == []
    assert unexpected == ["rotary_emb.inv_freq"]

    generator = torch.Generator().manual_seed(13)
    inputs = torch.randint(0, upstream.vocab_size, (2, 24), generator=generator)
    with torch.no_grad():
        released_outputs = upstream(inputs)
        portable_outputs = portable(inputs)

    assert released_outputs.keys() == portable_outputs.keys()
    for name in released_outputs:
        torch.testing.assert_close(
            released_outputs[name], portable_outputs[name], atol=2e-5, rtol=2e-5
        )


def test_released_lightning_checkpoint_loader_is_prefix_strict(tmp_path):
    vocab_path = _vocab_cache(tmp_path)
    source = build_upstream_adapter("sudoku", vocab_path, seed=17)
    checkpoint = tmp_path / "released.ckpt"
    torch.save(
        {
            "global_step": 123,
            "state_dict": {
                f"backbone.{name}": tensor.detach().clone()
                for name, tensor in source.backbone.state_dict().items()
            },
        },
        checkpoint,
    )
    restored = build_upstream_adapter("sudoku", vocab_path, seed=19)
    payload = load_lightning_checkpoint(restored, checkpoint)
    assert payload["global_step"] == 123
    for name, expected in source.backbone.state_dict().items():
        torch.testing.assert_close(restored.backbone.state_dict()[name], expected)


def test_released_dataset_factory_honours_configured_vocab_path(tmp_path):
    vocab_path = _vocab_cache(tmp_path)
    sample = {
        "x_k": np.asarray([0, 11, 27, 30], dtype=np.uint8),
        "y_star": np.asarray([1, 11, 27, 30], dtype=np.uint8),
        "r_star": np.zeros(4, dtype=np.uint8),
        "e_star": np.zeros(4, dtype=np.uint8),
        "c_star": np.zeros(4, dtype=np.uint8),
    }
    dataset_path = tmp_path / "tiny.pkl"
    with dataset_path.open("wb") as handle:
        pickle.dump({"samples": [sample, sample]}, handle)

    _install_train_path()
    dataloader = importlib.import_module("dataloader")
    config = OmegaConf.create(
        {
            "data": {
                "dataset_path": str(dataset_path),
                "vocab_cache_path": str(vocab_path),
                "train_ratio": 0.5,
                "streaming": False,
                "cache_dir": str(tmp_path / "cache"),
                "chunk_size": 2,
            },
            "model": {"length": 400},
        }
    )
    dataset = dataloader.get_dataset(
        "apmdm",
        tokenizer=None,
        wrap=False,
        mode="train",
        cache_dir=str(tmp_path / "cache"),
        config=config,
    )
    assert len(dataset) == 1
    assert dataset.tokenizer.mask_token_id == 10
    assert dataset.tokenizer.vocab_size == 34


def test_streaming_train_and_validation_chunks_cannot_overwrite_each_other(tmp_path):
    _install_train_path()
    official_loader = importlib.import_module("apmdm_dataloader")
    vocab_path = _vocab_cache(tmp_path)

    samples = []
    for token in (1, 2, 3, 4):
        samples.append(
            {
                "x_k": np.asarray([token, 11, 27, 30], dtype=np.uint8),
                "y_star": np.asarray([token, 11, 27, 30], dtype=np.uint8),
                "r_star": np.zeros(4, dtype=np.uint8),
                "e_star": np.zeros(4, dtype=np.uint8),
                "c_star": np.zeros(4, dtype=np.uint8),
            }
        )
    dataset_path = tmp_path / "stream.pkl"
    with dataset_path.open("wb") as handle:
        pickle.dump({"samples": samples}, handle)

    kwargs = {
        "data_path": str(dataset_path),
        "streaming": True,
        "max_length": 400,
        "train_ratio": 0.5,
        "vocab_cache_path": str(vocab_path),
        "cache_dir": str(tmp_path / "chunks"),
        "chunk_size": 1,
    }
    train = official_loader.APMDMDataset(mode="train", **kwargs)
    validation = official_loader.APMDMDataset(mode="validation", **kwargs)

    assert all(Path(name).name.startswith("train_") for name in train.data.chunk_files)
    assert all(
        Path(name).name.startswith("validation_")
        for name in validation.data.chunk_files
    )
    assert train[0]["input_ids"][0] in {1, 2}
    assert validation[0]["input_ids"][0] in {3, 4}
