"""Finite-objective and classifier-termination tests for chemistry IP."""

from __future__ import annotations

import torch

from repro.ip_chem_data import SmilesTokenizer, collate_smiles, lex_smiles
from repro.ip_chem_eval import sample_decode
from repro.ip_chem_model import (
    ChemInsertionDecoder,
    ChemInsertionPosterior,
    TransformerSpec,
    _augment_with_eos,
    _sample_orders,
    fixed_order_objective,
    learned_ip_objective,
    random_ip_objective,
)


def _tokenizer() -> SmilesTokenizer:
    vocabulary = sorted({token for row in ("CCO", "c1ccccc1", "CC(=O)N", "ClCBr") for token in lex_smiles(row)})
    return SmilesTokenizer(("<PAD>", "<BOS>", "<EOS>", *vocabulary))


def _spec() -> TransformerSpec:
    return TransformerSpec(width=32, layers=1, heads=4, dropout=0.0)


def _batch(tokenizer: SmilesTokenizer):
    return collate_smiles(("CCO", "c1ccccc1", "CC(=O)N", "ClCBr"), tokenizer)


def _assert_gradients(module: torch.nn.Module) -> None:
    gradients = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
    assert gradients
    assert all(torch.isfinite(row).all() for row in gradients)
    assert sum(float(row.abs().sum()) for row in gradients) > 0


def test_smiles_lexer_is_lossless_for_representative_grammar():
    rows = ("CC(=O)Nc1ccc(Cl)cc1", "C%10CCCCC%10", "C[C@@H](N)C(=O)O", "[NH3+]CC[O-]")
    for row in rows:
        assert "".join(lex_smiles(row)) == row


def test_classifier_eos_is_augmented_last():
    tokenizer = _tokenizer()
    batch = _batch(tokenizer)
    targets, mask, lengths = _augment_with_eos(batch, tokenizer)
    rows = torch.arange(targets.shape[0])
    assert torch.equal(targets[rows, batch.lengths], torch.full_like(batch.lengths, tokenizer.eos_id))
    assert torch.equal(lengths, batch.lengths + 1)
    assert mask[rows, batch.lengths].all()


def test_classifier_posterior_forces_eos_to_final_permutation_position():
    tokenizer = _tokenizer()
    batch = _batch(tokenizer)
    targets, mask, _ = _augment_with_eos(batch, tokenizer)
    logits = torch.zeros_like(targets, dtype=torch.float32).masked_fill(~mask, -torch.inf)
    logits = logits.masked_fill((targets == tokenizer.eos_id) & mask, -1e7)
    orders = _sample_orders(logits, mask, 32, torch.Generator().manual_seed(11))
    rows = torch.arange(targets.shape[0])
    for sample in orders:
        assert torch.equal(sample[rows, batch.lengths], batch.lengths)


def test_all_chemistry_objectives_are_finite_and_differentiable():
    tokenizer = _tokenizer()
    batch = _batch(tokenizer)
    for objective in (fixed_order_objective, random_ip_objective):
        decoder = ChemInsertionDecoder(tokenizer, _spec())
        result = objective(decoder, batch, torch.Generator().manual_seed(7))
        assert torch.isfinite(result.loss)
        result.loss.backward()
        _assert_gradients(decoder)
    decoder = ChemInsertionDecoder(tokenizer, _spec())
    posterior = ChemInsertionPosterior(tokenizer, _spec())
    result = learned_ip_objective(
        decoder, posterior, batch, torch.Generator().manual_seed(7)
    )
    assert torch.isfinite(result.loss)
    result.loss.backward()
    _assert_gradients(decoder)
    _assert_gradients(posterior)


def test_search_free_sampler_never_emits_special_tokens():
    tokenizer = _tokenizer()
    decoder = ChemInsertionDecoder(tokenizer, _spec())
    decoded = sample_decode(
        decoder,
        4,
        mode="insertion",
        generator=torch.Generator().manual_seed(9),
        max_tokens=5,
    )
    for sequence in decoded.token_sequences:
        assert tokenizer.pad_id not in sequence
        assert tokenizer.bos_id not in sequence
        assert tokenizer.eos_id not in sequence
