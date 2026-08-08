"""Portable path resolution.

Every large or mutable artefact (generated transitions, checkpoints, caches,
per-puzzle evaluation rows, telemetry) lives under a *data root* that is
outside the Git working tree.  The repository root is discovered relative to
this file, so no absolute ``/workspace/...`` path is ever required.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Environment variable that overrides the default data root.
DATA_ROOT_ENV = "APMDM_REPRO_DATA_ROOT"

#: Default data root used when :data:`DATA_ROOT_ENV` is unset.
DEFAULT_DATA_ROOT = Path.home() / "apmdm-repro-data"


def repo_root() -> Path:
    """Return the repository root (the directory containing ``repro/``)."""
    return Path(__file__).resolve().parent.parent


def dataset_dir() -> Path:
    """Return the checked-in official Sudoku dataset directory."""
    return repo_root() / "dataset" / "sudoku"


def train_config_dir() -> Path:
    """Return the checked-in Hydra config directory."""
    return repo_root() / "train" / "configs"


def data_root(create: bool = False) -> Path:
    """Return the out-of-tree data root.

    Args:
        create: create the directory (and parents) if it does not exist.
    """
    raw = os.environ.get(DATA_ROOT_ENV)
    root = Path(raw).expanduser().resolve() if raw else DEFAULT_DATA_ROOT
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def _sub(name: str, create: bool) -> Path:
    path = data_root(create=create) / name
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def transitions_dir(create: bool = False) -> Path:
    """Directory holding generated solver transitions."""
    return _sub("transitions", create)


def runs_dir(create: bool = False) -> Path:
    """Directory holding training runs (checkpoints, telemetry, manifests)."""
    return _sub("runs", create)


def eval_dir(create: bool = False) -> Path:
    """Directory holding evaluation outputs (per-puzzle rows, aggregates)."""
    return _sub("eval", create)


def assert_outside_repo(path: Path) -> Path:
    """Raise if ``path`` would place bulk artefacts inside the Git work tree."""
    resolved = Path(path).expanduser().resolve()
    root = repo_root()
    if resolved == root or root in resolved.parents:
        raise ValueError(
            f"refusing to write bulk artefacts inside the repository: {resolved}"
        )
    return resolved
