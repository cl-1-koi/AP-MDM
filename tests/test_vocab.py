"""Vocabulary identity and legal token-role tests."""

from __future__ import annotations

import numpy as np

from repro import vocab


def test_vocabulary_matches_the_official_generator():
    report = vocab.authenticate_vocabulary()
    assert report.generator_declared_size == 32
    assert report.effective_size == 31
    assert report.max_id_reachable_in_data == 30
    assert report.eos_emitted is False


def test_official_vocabulary_ids_are_contiguous_zero_to_eos():
    official = vocab.official_vocabulary()
    assert sorted(official) == list(range(32))
    assert official[vocab.MASK] == "[MASK]"
    assert official[vocab.SEPARATOR] == "[SEP]"
    assert official[vocab.EOS] == "[EOS]"
    for i in range(1, 16):
        assert official[vocab.WHITE + i] == f"[COLOR_{i}]"


def test_token_names_agree_with_the_official_vocabulary():
    official = vocab.official_vocabulary()
    for token_id, name in official.items():
        assert vocab.token_name(token_id) == name


def test_slot_role_partition_is_disjoint_where_it_must_be():
    value_roles = vocab.LEGAL_TOKENS_BY_SLOT[vocab.SLOT_VALUE]
    colour_roles = vocab.LEGAL_TOKENS_BY_SLOT[vocab.SLOT_COLOR]
    separator_roles = vocab.LEGAL_TOKENS_BY_SLOT[vocab.SLOT_SEPARATOR]
    assert separator_roles == frozenset({vocab.SEPARATOR})
    assert vocab.SEPARATOR not in value_roles
    assert vocab.SEPARATOR not in colour_roles
    assert vocab.EOS not in set().union(*vocab.LEGAL_TOKENS_BY_SLOT.values())
    # The marker slot legitimately stores a failed digit during backtracking.
    assert vocab.DIGITS <= vocab.LEGAL_TOKENS_BY_SLOT[vocab.SLOT_MARKER]


def test_generated_transitions_only_use_legal_token_roles(puzzle_transitions):
    for name in ("x_k", "y_star", "x_next"):
        arr = getattr(puzzle_transitions, name)
        for slot, legal in vocab.LEGAL_TOKENS_BY_SLOT.items():
            column = arr[:, slot :: vocab.TOKENS_PER_CELL]
            observed = set(np.unique(column).tolist())
            assert observed <= set(legal), (name, slot, observed - set(legal))


def test_generated_transitions_never_emit_eos_or_exceed_effective_vocab(puzzle_transitions):
    for name in ("x_k", "y_star", "x_next"):
        arr = getattr(puzzle_transitions, name)
        assert int(arr.max()) < vocab.VOCAB_SIZE
        assert not (arr == vocab.EOS).any()


def test_position_helpers():
    assert vocab.slot_of(0) == vocab.SLOT_VALUE
    assert vocab.slot_of(3) == vocab.SLOT_SEPARATOR
    assert vocab.cell_of(0) == (0, 0)
    assert vocab.cell_of(4) == (0, 1)
    assert vocab.cell_of(36) == (1, 0)
    assert vocab.cell_of(vocab.SEQUENCE_LENGTH - 1) == (8, 8)
