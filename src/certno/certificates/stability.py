"""Stability factors for the a posteriori elliptic error certificate.

The certificate needs the inverse of the smallest eigenvalue of the SPD
operator :math:`A_h`.  We provide:

* ``exact_lambda_min`` -- a sparse eigenvalue computation (one factorisation);
* ``analytic_coercivity_1d/2d`` -- a closed-form lower bound
  :math:`a_{\\min}\\,\\lambda_{\\min}(-\\Delta_h)` that requires no factorisation
  and is therefore usable inside a cheap online certificate.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from ..solvers.elliptic import laplacian_min_eig_1d, laplacian_min_eig_2d


def exact_lambda_min(A: sparse.spmatrix, *, tol: float = 1e-10) -> float:
    """Smallest eigenvalue of an SPD sparse matrix via Lanczos iteration."""
    n = A.shape[0]
    if n <= 2:
        return float(np.linalg.eigvalsh(A.toarray()).min())
    try:
        val = eigsh(A, k=1, which="SA", return_eigenvectors=False, tol=tol)
        return float(val[0])
    except Exception:  # pragma: no cover - dense fallback for tiny/ill-conditioned
        return float(np.linalg.eigvalsh(A.toarray()).min())


def analytic_coercivity_1d(a_min: float, n_interior: int) -> float:
    """Lower bound on ``lambda_min(A_h)`` for the 1D operator."""
    return a_min * laplacian_min_eig_1d(n_interior)


def analytic_coercivity_2d(a_min: float, n_interior: int) -> float:
    """Lower bound on ``lambda_min(A_h)`` for the 2D operator."""
    return a_min * laplacian_min_eig_2d(n_interior)


def energy_norm_error_1d(
    A: sparse.spmatrix, r: np.ndarray
) -> tuple[float, float]:
    """Return ``(||e||_A, ||e||_2)`` exactly for the discrete error ``e = A^{-1} r``.

    Because ``e^T A e = r^T A^{-1} r`` and ``e = A^{-1} r``, this is a
    reference ("oracle") certificate that costs one solve.  It is *exact* for
    the discrete problem, so it is a useful upper benchmark for the cheap
    analytic certificate.
    """
    from scipy.sparse.linalg import spsolve

    r = np.asarray(r, dtype=float).ravel()
    e = spsolve(A, r)
    energy = float(np.sqrt(max(e @ (A.dot(e)), 0.0)))
    l2 = float(np.linalg.norm(e))
    return energy, l2
