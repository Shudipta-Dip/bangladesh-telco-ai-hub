"""Every model-weight vector on a fixed tenth-step lattice. Each vector is emitted once."""

from __future__ import annotations

from math import comb

import numpy as np


def mix_count(n_models, parts):
    if n_models <= 0:
        raise ValueError("catalog is empty")
    return comb(parts + n_models - 1, parts)


def mix_matrix(n_models, parts):
    """Integer weights, shape (n_mix, n_models), each row sums to `parts`.

    Rows are unique. Order is lexicographic. Divide by `parts` to get shares.
    """
    n_rows = mix_count(n_models, parts)
    out = np.zeros((n_rows, n_models), dtype=np.uint8)
    prefix = np.zeros(n_models, dtype=np.uint8)
    row = 0

    def rec(pos, remain):
        nonlocal row
        if pos == n_models - 1:
            prefix[pos] = remain
            out[row] = prefix
            row += 1
            return
        for value in range(remain + 1):
            prefix[pos] = value
            rec(pos + 1, remain - value)

    rec(0, parts)
    if row != n_rows:
        raise RuntimeError(f"expected {n_rows} mixes, wrote {row}")
    return out
