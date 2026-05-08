"""Small helpers shared with the ICML'09 MATLAB ports."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def best_i_j(matrix: np.ndarray) -> Tuple[int, int]:
    """Index of the minimum entry in a 2-D array, returned as (row, col).

    Port of best_i_j.m. Tied with MATLAB's column-major argmin so that we
    pick the same (eta, alpha) point even when the MATLAB scan traverses
    columns first.
    """
    flat = np.asarray(matrix).flatten(order="F")
    idx = int(np.argmin(flat))
    n_rows = matrix.shape[0]
    col = idx // n_rows
    row = idx - n_rows * col
    return row, col


__all__ = ["best_i_j"]
