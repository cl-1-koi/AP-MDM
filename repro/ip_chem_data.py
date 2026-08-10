"""Authenticated GuacaMol/ChEMBL SMILES data for the insertion-process gate.

The official GuacaMol v1 split is distributed by BenevolentAI on Figshare.
This module deliberately keeps chemistry out of the training input pipeline:
SMILES are lexed losslessly and RDKit is used only by evaluation.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterable, Sequence

import torch


OFFICIAL_SPLITS = {
    "train": {
        "filename": "guacamol_v1_train.smiles",
        "count": 1_273_104,
        "md5": "05ad85d871958a05c02ab51a4fde8530",
        "sha256": "3c67ee945f351dbbdc02d9016da22efaffc32a39d882021b6f213d5cd60b6a80",
    },
    "valid": {
        "filename": "guacamol_v1_valid.smiles",
        "count": 79_568,
        "md5": "e53db4bff7dc4784123ae6df72e3b1f0",
        "sha256": "124c4e76062bebf3a9bba9812e76fea958a108f25e114a98ddf49c394c4773bf",
    },
    "test": {
        "filename": "guacamol_v1_test.smiles",
        "count": 238_706,
        "md5": "677b757ccec4809febd83850b43e1616",
        "sha256": "0b7e1e88e7bd07ee7fe5d2ef668e8904c763635c93654af094fa5446ff363015",
    },
}

PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

# Bracket atoms must remain atomic. Outside brackets the SMILES organic subset
# permits only Br/Cl as two-character symbols; e.g. ``Cc`` is C followed by c.
SMILES_TOKEN = re.compile(
    r"(\[[^\]]+\]|%\d{2}|Br|Cl|B|C|N|O|P|S|F|I|b|c|n|o|p|s|"
    r"\d|\(|\)|\.|=|#|-|\+|\\|/|@|:|~|\?|\*|\$)"
)


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticate_split(path: Path, split: str) -> dict[str, object]:
    expected = OFFICIAL_SPLITS[split]
    sha256 = file_digest(path)
    md5 = file_digest(path, "md5")
    count = sum(1 for _ in path.open())
    if sha256 != expected["sha256"] or md5 != expected["md5"]:
        raise ValueError(f"{split} GuacaMol digest mismatch")
    if count != expected["count"]:
        raise ValueError(f"{split} GuacaMol count mismatch: {count}")
    return {"path": str(path.resolve()), "count": count, "sha256": sha256, "md5": md5}


def lex_smiles(smiles: str) -> tuple[str, ...]:
    tokens = tuple(SMILES_TOKEN.findall(smiles))
    if "".join(tokens) != smiles:
        raise ValueError(f"SMILES lexer did not cover input: {smiles!r}")
    return tokens


@dataclass(frozen=True)
class SmilesTokenizer:
    tokens: tuple[str, ...]

    @classmethod
    def from_training_file(cls, path: Path) -> "SmilesTokenizer":
        vocabulary: set[str] = set()
        with path.open() as handle:
            for line in handle:
                vocabulary.update(lex_smiles(line.strip()))
        return cls((PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, *sorted(vocabulary)))

    @cached_property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    def encode(self, smiles: str) -> tuple[int, ...]:
        lookup = self.token_to_id
        return tuple(lookup[token] for token in lex_smiles(smiles))

    def decode(self, token_ids: Iterable[int]) -> str:
        return "".join(self.tokens[int(token)] for token in token_ids)

    def canonical_text(self) -> str:
        return "\n".join(self.tokens) + "\n"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_text().encode()).hexdigest()


def read_smiles(path: Path, limit: int | None = None) -> list[str]:
    rows: list[str] = []
    with path.open() as handle:
        for line in handle:
            rows.append(line.strip())
            if limit is not None and len(rows) >= limit:
                break
    return rows


@dataclass(frozen=True)
class ChemBatch:
    targets: torch.Tensor
    target_mask: torch.Tensor
    lengths: torch.Tensor
    smiles: tuple[str, ...]

    def to(self, device: torch.device | str) -> "ChemBatch":
        return ChemBatch(
            targets=self.targets.to(device),
            target_mask=self.target_mask.to(device),
            lengths=self.lengths.to(device),
            smiles=self.smiles,
        )


def collate_smiles(smiles: Sequence[str], tokenizer: SmilesTokenizer) -> ChemBatch:
    if not smiles:
        raise ValueError("cannot collate an empty SMILES batch")
    encoded = [tokenizer.encode(row) for row in smiles]
    width = max(len(row) for row in encoded)
    targets = torch.full((len(encoded), width), tokenizer.pad_id, dtype=torch.long)
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for index, row in enumerate(encoded):
        value = torch.tensor(row, dtype=torch.long)
        targets[index, : len(row)] = value
        mask[index, : len(row)] = True
    return ChemBatch(
        targets=targets,
        target_mask=mask,
        lengths=mask.sum(dim=-1),
        smiles=tuple(smiles),
    )


def structural_token(token: str) -> bool:
    """Whether a token belongs to the paper's branch/ring/bond scaffold."""
    return bool(
        token in {"(", ")", ".", "=", "#", "-", "+", "\\", "/", "@", ":", "~"}
        or token.isdigit()
        or (token.startswith("%") and token[1:].isdigit())
    )
