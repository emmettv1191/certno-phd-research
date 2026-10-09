"""Discrete PDE residuals for candidate surrogate predictions."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def residual_1d(
    A: sparse.spmatrix, f_interior: np.ndarray, u_hat_interior: np.ndarray
) -> np.ndarray:
    """Return ``r = f - A u_hat`` on the interior nodes."""
    f = np.asarray(f_interior, dtype=float).ravel()
    u = np.asarray(u_hat_interior, dtype=float).ravel()
    if u.size != f.size:
        raise ValueError("u_hat and f must have the same interior length")
    return f - A.dot(u)


def residual_2d(
    A: sparse.spmatrix, F_interior: np.ndarray, U_hat_interior: np.ndarray
) -> np.ndarray:
    """Return ``r = F - A U_hat`` flattened in row-major interior order."""
    F = np.asarray(F_interior, dtype=float)
    U = np.asarray(U_hat_interior, dtype=float)
    if U.shape != F.shape:
        raise ValueError("U_hat and F must have the same shape")
    return F.ravel() - A.dot(U.ravel())
