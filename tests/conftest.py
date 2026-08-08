"""Shared pytest fixtures for the AP-MDM Sudoku replication test suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session", autouse=True)
def isolated_data_root(tmp_path_factory):
    """Point the package's data root at a throwaway directory for the session."""
    root = tmp_path_factory.mktemp("apmdm-repro-test-root")
    previous = os.environ.get("APMDM_REPRO_DATA_ROOT")
    os.environ["APMDM_REPRO_DATA_ROOT"] = str(root)
    yield root
    if previous is None:
        os.environ.pop("APMDM_REPRO_DATA_ROOT", None)
    else:
        os.environ["APMDM_REPRO_DATA_ROOT"] = previous


@pytest.fixture(scope="session")
def train_split():
    from repro import data as data_mod

    return data_mod.load_split("train")


@pytest.fixture(scope="session")
def test_split():
    from repro import data as data_mod

    return data_mod.load_split("test")


@pytest.fixture(scope="session")
def paper_config():
    from repro.config import load_paper_config

    return load_paper_config()


@pytest.fixture(scope="session")
def historical_config():
    from repro.config import load_historical_config

    return load_historical_config()


@pytest.fixture(scope="session")
def puzzle_transitions(train_split):
    """Transitions for one real training puzzle (exercises backtracking)."""
    from repro.trajectories import generate_puzzle_transitions

    return generate_puzzle_transitions(np.asarray(train_split.puzzles[0]), 0)


@pytest.fixture(scope="session")
def easy_puzzle(test_split):
    """A held-out puzzle used for cheap generator fixtures."""
    return np.asarray(test_split.puzzles[0])
