"""Regression test for AP-MDM streaming-loader epoch reconstruction."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

TRAIN_DIR = Path(__file__).resolve().parents[1] / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

import dataloader  # noqa: E402


class TinyTokenizer:
    pad_token_id = 32


class TinyChunkDataset:
    def __init__(self):
        self.rows = [
            {
                "input_ids": [1, 11, 27, 30],
                "target_ids": [2, 11, 27, 30],
                "attention_mask": [1, 1, 1, 1],
                "target_attention_mask": [1, 1, 1, 1],
                "remask_labels": [0, 0, 0, 0],
                "expansion_labels": [0, 0, 0, 0],
                "contraction_labels": [0, 0, 0, 0],
            },
            {
                "input_ids": [3, 11, 27, 30],
                "target_ids": [4, 11, 27, 30],
                "attention_mask": [1, 1, 1, 1],
                "target_attention_mask": [1, 1, 1, 1],
                "remask_labels": [0, 0, 0, 0],
                "expansion_labels": [0, 0, 0, 0],
                "contraction_labels": [0, 0, 0, 0],
            },
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def iter_chunks_sequentially(self, batch_size, tokenizer):
        import apmdm_dataloader

        yield apmdm_dataloader.collate_fn(self.rows, tokenizer)  # pragma: no cover


def test_replacement_loader_preserves_tensor_collation():
    dataset = TinyChunkDataset()
    wrapper = dataloader.SequentialChunkDataLoader(dataset, 2, TinyTokenizer())
    replacement = torch.utils.data.DataLoader(
        wrapper.dataset,
        batch_size=wrapper.batch_size,
        collate_fn=wrapper.collate_fn,
        shuffle=False,
    )
    batch = next(iter(replacement))
    assert isinstance(batch["input_ids"], torch.Tensor)
    assert batch["input_ids"].shape == (2, 4)
    assert batch["target_ids"].shape == (2, 4)
