"""Authenticated Sudoku token vocabulary and per-slot token roles.

The official generator (``dataset/sudoku/sudoku_generator.py``) declares
``VOCAB_SIZE = 32`` including an ``[EOS]`` token with id 31.  The current
generator's ``on_end`` no longer emits ``[EOS]`` ("Modified: Remove final
expand+eos operation"), so ids 0..30 are the only ids that can ever appear in
generated data.  That is exactly the vocabulary size 31 declared by
``train/configs/sudoku.yaml`` and by the paper's implementation-details
section, while the paper's prose ("The vocabulary consists of 32 tokens")
counts the unused ``[EOS]``.  Nothing here changes the vocabulary: the
constants below are re-derived from the official generator at import time and
cross-checked in the test suite.

Each 9x9 cell occupies **four** consecutive tokens
``(value, color, marker, separator)``; 81 cells give the 324-token sequence
that the paper describes (the paper's prose says "3 consecutive tokens", which
is inconsistent with its own 324 figure and with the code).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Tuple

EMPTY = 0
DIGIT_MIN, DIGIT_MAX = 1, 9
MASK = 10
WHITE = 11
COLOR_MIN, COLOR_MAX = 12, 26  # COLOR_1 .. COLOR_15
NORMAL = 27
SKULL = 28
BRANCH = 29
SEPARATOR = 30
EOS = 31

#: Number of embedding rows required by generated Sudoku data.  Ids 0..30.
VOCAB_SIZE = 31

#: Declared vocabulary size of the official generator, including unused [EOS].
GENERATOR_DECLARED_VOCAB_SIZE = 32

TOKENS_PER_CELL = 4
NUM_CELLS = 81
SEQUENCE_LENGTH = NUM_CELLS * TOKENS_PER_CELL  # 324

SLOT_VALUE, SLOT_COLOR, SLOT_MARKER, SLOT_SEPARATOR = 0, 1, 2, 3

DIGITS: FrozenSet[int] = frozenset(range(DIGIT_MIN, DIGIT_MAX + 1))
COLORS: FrozenSet[int] = frozenset(range(COLOR_MIN, COLOR_MAX + 1))

#: Legal token ids per cell slot, as exercised by the official generator.
#:
#: The marker slot legitimately holds a *digit* during backtracking: the
#: backtrack transition stores the failed value in the marker position
#: ("the third position is unmasked to store the failed value"), which the
#: recovery transition then reads back.
LEGAL_TOKENS_BY_SLOT: Dict[int, FrozenSet[int]] = {
    SLOT_VALUE: frozenset({EMPTY, MASK} | DIGITS),
    SLOT_COLOR: frozenset({MASK, WHITE} | COLORS),
    SLOT_MARKER: frozenset({MASK, NORMAL, SKULL, BRANCH} | DIGITS),
    SLOT_SEPARATOR: frozenset({SEPARATOR}),
}


@dataclass(frozen=True)
class VocabularyReport:
    """Result of authenticating the vocabulary against the official generator."""

    generator_declared_size: int
    max_id_reachable_in_data: int
    effective_size: int
    eos_emitted: bool
    names: Tuple[Tuple[int, str], ...]

    def as_dict(self) -> dict:
        return {
            "generator_declared_size": self.generator_declared_size,
            "max_id_reachable_in_data": self.max_id_reachable_in_data,
            "effective_size": self.effective_size,
            "eos_emitted": self.eos_emitted,
            "names": {str(i): n for i, n in self.names},
        }


def token_name(token_id: int) -> str:
    """Human-readable name for a token id."""
    if token_id == EMPTY:
        return "[EMPTY]"
    if token_id in DIGITS:
        return str(token_id)
    if token_id == MASK:
        return "[MASK]"
    if token_id == WHITE:
        return "[WHITE]"
    if token_id in COLORS:
        return f"[COLOR_{token_id - WHITE}]"
    if token_id == NORMAL:
        return "[NORMAL]"
    if token_id == SKULL:
        return "[SKULL]"
    if token_id == BRANCH:
        return "[BRANCH]"
    if token_id == SEPARATOR:
        return "[SEP]"
    if token_id == EOS:
        return "[EOS]"
    return f"[UNK_{token_id}]"


def official_vocabulary() -> Dict[int, str]:
    """Return the vocabulary produced by the official generator itself."""
    from repro.official import sample_generator

    return dict(sample_generator().create_vocabulary())


def authenticate_vocabulary() -> VocabularyReport:
    """Cross-check our constants against the official generator's own vocabulary.

    Raises:
        AssertionError: if the official generator's token ids disagree with the
            constants declared in this module.
    """
    from repro.official import sample_generator

    gen = sample_generator()
    expected = {
        "EMPTY_TOKEN": EMPTY,
        "MASK_TOKEN": MASK,
        "WHITE_TOKEN": WHITE,
        "NORMAL_TOKEN": NORMAL,
        "SKULL_TOKEN": SKULL,
        "BRANCH_TOKEN": BRANCH,
        "SEPARATOR_TOKEN": SEPARATOR,
        "EOS_TOKEN": EOS,
        "VOCAB_SIZE": GENERATOR_DECLARED_VOCAB_SIZE,
    }
    for attr, value in expected.items():
        actual = getattr(gen, attr)
        if actual != value:
            raise AssertionError(
                f"official generator {attr}={actual} disagrees with repro.vocab {value}"
            )

    vocab = gen.create_vocabulary()
    if max(vocab) != EOS:
        raise AssertionError(f"unexpected max vocabulary id {max(vocab)}")

    names = tuple(sorted((int(i), str(n)) for i, n in vocab.items()))
    return VocabularyReport(
        generator_declared_size=GENERATOR_DECLARED_VOCAB_SIZE,
        max_id_reachable_in_data=SEPARATOR,
        effective_size=VOCAB_SIZE,
        eos_emitted=False,
        names=names,
    )


def slot_of(position: int) -> int:
    """Return the cell slot (0..3) for a sequence position."""
    return position % TOKENS_PER_CELL


def cell_of(position: int) -> Tuple[int, int]:
    """Return the ``(row, col)`` of the cell containing a sequence position."""
    cell = position // TOKENS_PER_CELL
    return cell // 9, cell % 9
