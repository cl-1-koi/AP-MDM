"""Import shim for the official, unmodified Sudoku dataset code.

``dataset/sudoku/*.py`` use bare intra-directory imports (``from sudoku_solver
import SudokuSolver``), so the directory must be on ``sys.path``.  This module
performs that insertion exactly once, in a way that is safe to import from
anywhere and that never mutates the official sources.
"""

from __future__ import annotations

import sys
from types import ModuleType

from repro.paths import dataset_dir

_INSTALLED = False


def _ensure_path() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    path = str(dataset_dir())
    if path not in sys.path:
        sys.path.insert(0, path)
    _INSTALLED = True


def official_generator_module() -> ModuleType:
    """Return the official ``sudoku_generator`` module."""
    _ensure_path()
    import sudoku_generator  # type: ignore[import-not-found]

    return sudoku_generator


def official_solver_module() -> ModuleType:
    """Return the official ``sudoku_solver`` module."""
    _ensure_path()
    import sudoku_solver  # type: ignore[import-not-found]

    return sudoku_solver


def sample_generator():
    """Return a fresh official ``APMDMSampleGenerator`` instance."""
    return official_generator_module().APMDMSampleGenerator()


def data_manager():
    """Return a fresh official ``APMDMDataManager`` instance."""
    return official_generator_module().APMDMDataManager()


def solver():
    """Return a fresh official ``SudokuSolver`` instance."""
    return official_solver_module().SudokuSolver()
