"""Paper-faithful reproduction harness for the AP-MDM Sudoku result (arXiv:2510.06190v2).

This package is additive: it does not modify the historical upstream training
stack under ``train/`` or the official dataset code under ``dataset/``.  It
reconstructs, in an auditable and testable form:

* authenticated loading of the official Sudoku arrays,
* deterministic solver-trajectory (state-transition) generation and accounting,
* the paper's supervised AP-MDM objective and encoder architecture,
* the paper's Algorithm 1 iterative generation procedure, prompt-conditioned,
* an exact evaluator with immutable per-puzzle rows,
* a sealed experiment manifest, telemetry, and a fail-closed GPU supervisor.

Nothing in this package writes datasets, checkpoints, caches, raw evaluation
rows or telemetry inside the Git working tree; see :mod:`repro.paths`.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
