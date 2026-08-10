"""Search-free sampling and GuacaMol smoke metrics for SMILES IP models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch

from repro.ip_chem_data import SmilesTokenizer, structural_token
from repro.ip_chem_model import ChemInsertionDecoder


DecodeMode = Literal["fixed", "insertion"]


@dataclass(frozen=True)
class ChemDecodedBatch:
    token_sequences: tuple[tuple[int, ...], ...]
    insertion_traces: tuple[tuple[int, ...], ...]
    terminated: tuple[bool, ...]
    steps: int


def _sample(logits: torch.Tensor, generator: torch.Generator) -> int:
    probability = torch.softmax(logits.float(), dim=-1)
    return int(torch.multinomial(probability, 1, generator=generator).item())


@torch.inference_mode()
def sample_decode(
    decoder: ChemInsertionDecoder,
    batch_size: int,
    *,
    mode: DecodeMode,
    generator: torch.Generator,
    max_tokens: int = 100,
) -> ChemDecodedBatch:
    tokenizer = decoder.tokenizer
    device = next(decoder.parameters()).device
    sequences: list[list[int]] = [[] for _ in range(batch_size)]
    traces: list[list[int]] = [[] for _ in range(batch_size)]
    terminated = [False] * batch_size
    was_training = decoder.training
    decoder.eval()
    steps = 0
    for steps in range(1, max_tokens + 2):
        width = max(1, max((len(row) for row in sequences), default=0))
        partial = torch.full(
            (batch_size, width), tokenizer.pad_id, dtype=torch.long, device=device
        )
        mask = torch.zeros_like(partial, dtype=torch.bool)
        for row, sequence in enumerate(sequences):
            if sequence:
                partial[row, : len(sequence)] = torch.tensor(sequence, device=device)
                mask[row, : len(sequence)] = True
        output = decoder(partial, mask)
        for row, sequence in enumerate(sequences):
            if terminated[row]:
                continue
            if mode == "fixed":
                slot = len(sequence)
            elif mode == "insertion":
                slot = _sample(output.location_logits[row, : len(sequence) + 1], generator)
            else:
                raise ValueError(f"unknown decode mode: {mode}")
            content = output.content_logits[row, slot].clone()
            content[tokenizer.pad_id] = -torch.inf
            content[tokenizer.bos_id] = -torch.inf
            # Classifier termination is EOS content at the final insertion slot.
            if slot != len(sequence):
                content[tokenizer.eos_id] = -torch.inf
            token = _sample(content, generator)
            if token == tokenizer.eos_id and slot == len(sequence):
                terminated[row] = True
            elif len(sequence) < max_tokens:
                sequence.insert(slot, token)
                traces[row].append(token)
        if all(terminated):
            break
    decoder.train(was_training)
    return ChemDecodedBatch(
        token_sequences=tuple(tuple(row) for row in sequences),
        insertion_traces=tuple(tuple(row) for row in traces),
        terminated=tuple(terminated),
        steps=steps,
    )


def generation_metrics(
    decoded: ChemDecodedBatch,
    tokenizer: SmilesTokenizer,
    training_smiles: set[str],
) -> tuple[dict[str, float], list[dict[str, object]]]:
    try:
        from rdkit import Chem, RDLogger
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("RDKit is required only for chemistry evaluation") from exc
    RDLogger.DisableLog("rdApp.*")
    rows: list[dict[str, object]] = []
    canonical_valid: list[str] = []
    pattern_hits = 0
    paired_hits = 0
    for sequence, trace, terminated in zip(
        decoded.token_sequences, decoded.insertion_traces, decoded.terminated
    ):
        smiles = tokenizer.decode(sequence)
        molecule = Chem.MolFromSmiles(smiles)
        canonical = Chem.MolToSmiles(molecule) if molecule is not None else None
        if canonical is not None:
            canonical_valid.append(canonical)
        trace_tokens = [tokenizer.tokens[token] for token in trace]
        structural_steps = [
            index for index, token in enumerate(trace_tokens) if structural_token(token)
        ]
        atom_steps = [
            index for index, token in enumerate(trace_tokens) if not structural_token(token)
        ]
        scaffold_first = not structural_steps or not atom_steps or max(structural_steps) < min(atom_steps)
        pattern_hits += int(scaffold_first)
        paired = smiles.count("(") == smiles.count(")")
        for marker in tuple(str(digit) for digit in range(1, 10)) + ("%10", "%11"):
            paired = paired and smiles.count(marker) % 2 == 0
        paired_hits += int(paired)
        rows.append(
            {
                "smiles": smiles,
                "canonical_smiles": canonical,
                "valid": canonical is not None,
                "terminated": terminated,
                "scaffold_before_atoms": scaffold_first,
                "paired_parentheses_and_rings": paired,
                "insertion_trace": trace_tokens,
            }
        )
    count = len(rows)
    unique = set(canonical_valid)
    novel = {smiles for smiles in unique if smiles not in training_smiles}
    return {
        "examples": float(count),
        "validity": len(canonical_valid) / count,
        "valid_unique": len(unique) / count,
        "valid_unique_novel": len(novel) / count,
        "terminated": sum(decoded.terminated) / count,
        "scaffold_before_atoms": pattern_hits / count,
        "paired_parentheses_and_rings": paired_hits / count,
        "mean_length": sum(len(row) for row in decoded.token_sequences) / count,
        "decode_steps": float(decoded.steps),
    }, rows
