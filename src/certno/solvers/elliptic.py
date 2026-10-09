"""High-accuracy finite-difference solvers for variable-coefficient elliptic PDEs.

We solve, in 1D on :math:`\\Omega=(0,1)` and in 2D on :math:`\\Omega=(0,1)^2`,

.. math::

    -\\nabla\\cdot(a(x)\\,\\nabla u) = f, \\qquad u|_{\\partial\\Omega}=0,

with a symmetric positive-definite finite-difference operator :math:`A_h`
assembled from face conductances.  Writing the discrete solution as
:math:`u_h = A_h^{-1} f`, the *exact* discrete error of any candidate
:math:`\\hat u` has a representation that the certificate module exploits:

.. math::

    u_h - \\hat u = A_h^{-1} r, \\qquad r = f - A_h \\hat u .

Because :math:`A_h` is SPD, :math:`\\|A_h^{-1}\\|_2 = 1/\\lambda_{\\min}(A_h)`,
and because the face weights are bounded below by :math:`a_{\\min}`,

.. math::

    \\lambda_{\\min}(A_h) \\;\\ge\\; a_{\\min}\\,\\lambda_{\\min}(-\\Delta_h),

where :math:`-\\Delta_h` is the unit-coefficient Dirichlet Laplacian.  The
latter has a closed form on the tensor grid, giving a fully computable
(storage- and factorization-free) stability factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


@dataclass
class EllipticSolution1D:
    """Container for a 1D elliptic solve."""

    x: np.ndarray  # nodal coordinates, length N+2 (boundaries included)
    u: np.ndarray  # nodal solution, length N+2 (zero at boundaries)
    A: sparse.csr_matrix  # interior operator, shape (N, N)
    f: np.ndarray  # interior right-hand side, length N
    a_nodes: np.ndarray  # nodal coefficient, length N+2
    a_min: float
    a_max: float
    h: float

    @property
    def u_interior(self) -> np.ndarray:
        return self.u[1:-1]


@dataclass
class EllipticSolution2D:
    """Container for a 2D elliptic solve."""

    x: np.ndarray  # 1D coordinate of the (N+2) grid lines
    U: np.ndarray  # (N+2, N+2) nodal solution, zero at boundary
    A: sparse.csr_matrix  # interior operator, shape (N*N, N*N)
    F: np.ndarray  # (N, N) interior right-hand side
    a_nodes: np.ndarray  # (N+2, N+2) nodal coefficient
    a_min: float
    a_max: float
    h: float

    @property
    def U_interior(self) -> np.ndarray:
        return self.U[1:-1, 1:-1]


# --------------------------------------------------------------------------- #
# Stability constants
# --------------------------------------------------------------------------- #
def laplacian_min_eig_1d(n_interior: int) -> float:
    """Smallest eigenvalue of the 1D unit-coefficient Dirichlet Laplacian.

    With ``h = 1/(n_interior + 1)`` the exact value is
    ``4 * sin(pi*h/2)**2 / h**2``.
    """
    if n_interior < 1:
        raise ValueError("n_interior must be >= 1")
    h = 1.0 / (n_interior + 1)
    return 4.0 * np.sin(np.pi * h / 2.0) ** 2 / h**2


def laplacian_min_eig_2d(n_interior: int) -> float:
    """Smallest eigenvalue of the 2D unit-coefficient 5-point Dirichlet Laplacian.

    ``= (8/h**2) * sin(pi*h/2)**2`` with ``h = 1/(n_interior + 1)``.
    """
    if n_interior < 1:
        raise ValueError("n_interior must be >= 1")
    h = 1.0 / (n_interior + 1)
    return 8.0 * np.sin(np.pi * h / 2.0) ** 2 / h**2


def coercivity_bound_1d(a_min: float, n_interior: int) -> float:
    """Lower bound on ``lambda_min(A_h)`` for the 1D variable-coefficient operator."""
    if a_min <= 0:
        raise ValueError("a_min must be positive for coercivity")
    return a_min * laplacian_min_eig_1d(n_interior)


def coercivity_bound_2d(a_min: float, n_interior: int) -> float:
    """Lower bound on ``lambda_min(A_h)`` for the 2D variable-coefficient operator."""
    if a_min <= 0:
        raise ValueError("a_min must be positive for coercivity")
    return a_min * laplacian_min_eig_2d(n_interior)


# --------------------------------------------------------------------------- #
# 1D assembly and solve
# --------------------------------------------------------------------------- #
def _face_conductances(a_nodes: np.ndarray, averaging: str) -> np.ndarray:
    """Compute face conductances from nodal coefficients.

    ``arithmetic`` -> ``(a_i + a_{i+1}) / 2``
    ``harmonic``   -> ``2 a_i a_{i+1} / (a_i + a_{i+1})`` (exact for interfaces)
    """
    a = np.asarray(a_nodes, dtype=float)
    if np.any(a <= 0):
        raise ValueError("coefficient must be strictly positive")
    if averaging == "arithmetic":
        return 0.5 * (a[:-1] + a[1:])
    if averaging == "harmonic":
        return 2.0 * a[:-1] * a[1:] / (a[:-1] + a[1:])
    raise ValueError(f"unknown averaging: {averaging!r}")


def assemble_elliptic_1d(
    a_nodes: np.ndarray,
    n_interior: int | None = None,
    *,
    averaging: str = "arithmetic",
) -> sparse.csr_matrix:
    """Assemble the interior 1D operator ``A_h`` from nodal coefficients.

    Parameters
    ----------
    a_nodes:
        Nodal coefficient values.  If ``n_interior`` is given, its length must
        be ``n_interior + 2`` (boundaries included); otherwise the interior
        count is inferred from ``len(a_nodes) - 2``.
    n_interior:
        Number of interior nodes.
    averaging:
        Face averaging rule, ``"arithmetic"`` or ``"harmonic"``.
    """
    a_nodes = np.asarray(a_nodes, dtype=float).ravel()
    if n_interior is None:
        n_interior = a_nodes.size - 2
    if a_nodes.size != n_interior + 2:
        raise ValueError(
            f"a_nodes length {a_nodes.size} != n_interior + 2 = {n_interior + 2}"
        )
    h = 1.0 / (n_interior + 1)
    w = _face_conductances(a_nodes, averaging)  # length n_interior + 1
    inv_h2 = 1.0 / h**2

    diag = (w[:-1] + w[1:]) * inv_h2
    off = -w[1:-1] * inv_h2  # interior coupling, length n_interior - 1
    A = sparse.diags(
        [off, diag, off],
        offsets=[-1, 0, 1],
        format="csr",
    )
    return A


def solve_elliptic_1d(
    a_nodes: np.ndarray,
    f_interior: np.ndarray,
    n_interior: int | None = None,
    *,
    averaging: str = "arithmetic",
) -> EllipticSolution1D:
    """Solve ``-(a u')' = f`` on (0,1) with homogeneous Dirichlet data."""
    a_nodes = np.asarray(a_nodes, dtype=float).ravel()
    f_interior = np.asarray(f_interior, dtype=float).ravel()
    if n_interior is None:
        n_interior = f_interior.size
    A = assemble_elliptic_1d(a_nodes, n_interior, averaging=averaging)
    u_int = spsolve(A, f_interior)
    x = np.linspace(0.0, 1.0, n_interior + 2)
    u = np.concatenate([[0.0], u_int, [0.0]])
    return EllipticSolution1D(
        x=x,
        u=u,
        A=A,
        f=f_interior,
        a_nodes=a_nodes,
        a_min=float(a_nodes.min()),
        a_max=float(a_nodes.max()),
        h=1.0 / (n_interior + 1),
    )


# --------------------------------------------------------------------------- #
# 2D assembly and solve
# --------------------------------------------------------------------------- #
def _face_average_2d(a: np.ndarray, axis: int, averaging: str) -> np.ndarray:
    if averaging == "arithmetic":
        if axis == 0:
            return 0.5 * (a[:-1, :] + a[1:, :])
        return 0.5 * (a[:, :-1] + a[:, 1:])
    if averaging == "harmonic":
        if axis == 0:
            return 2.0 * a[:-1, :] * a[1:, :] / (a[:-1, :] + a[1:, :])
        return 2.0 * a[:, :-1] * a[:, 1:] / (a[:, :-1] + a[:, 1:])
    raise ValueError(f"unknown averaging: {averaging!r}")


def assemble_elliptic_2d(
    a_nodes: np.ndarray,
    n_interior: int | None = None,
    *,
    averaging: str = "arithmetic",
) -> sparse.csr_matrix:
    """Assemble the interior 2D 5-point operator from a nodal coefficient field.

    ``a_nodes`` is an ``(N+2, N+2)`` array; the returned operator has shape
    ``(N*N, N*N)`` with row-major interior ordering ``(i, j) -> i*N + j``.
    """
    a = np.asarray(a_nodes, dtype=float)
    if a.ndim != 2:
        raise ValueError("a_nodes must be 2D")
    if n_interior is None:
        n_interior = a.shape[0] - 2
    if a.shape != (n_interior + 2, n_interior + 2):
        raise ValueError("a_nodes shape inconsistent with n_interior")
    if np.any(a <= 0):
        raise ValueError("coefficient must be strictly positive")

    n = n_interior
    h = 1.0 / (n + 1)
    inv_h2 = 1.0 / h**2

    w_x = _face_average_2d(a, axis=0, averaging=averaging)  # ((n+1), n+2)
    w_y = _face_average_2d(a, axis=1, averaging=averaging)  # (n+2, (n+1))

    wx_lo = w_x[:-1, 1:-1] * inv_h2  # (n, n) face below interior node
    wx_hi = w_x[1:, 1:-1] * inv_h2  # (n, n) face above
    wy_lo = w_y[1:-1, :-1] * inv_h2  # (n, n) face left
    wy_hi = w_y[1:-1, 1:] * inv_h2  # (n, n) face right

    idx = np.arange(n * n).reshape(n, n)
    diag = (wx_lo + wx_hi + wy_lo + wy_hi).ravel()

    rows = [idx.ravel()]
    cols = [idx.ravel()]
    vals = [diag]

    # x-direction couplings (offset -n and +n)
    m = np.zeros((n, n), dtype=bool)
    m[1:, :] = True
    rows.append(idx[m])
    cols.append((idx - n)[m])
    vals.append(-wx_lo[m])

    m = np.zeros((n, n), dtype=bool)
    m[:-1, :] = True
    rows.append(idx[m])
    cols.append((idx + n)[m])
    vals.append(-wx_hi[m])

    # y-direction couplings (offset -1 and +1)
    m = np.zeros((n, n), dtype=bool)
    m[:, 1:] = True
    rows.append(idx[m])
    cols.append((idx - 1)[m])
    vals.append(-wy_lo[m])

    m = np.zeros((n, n), dtype=bool)
    m[:, :-1] = True
    rows.append(idx[m])
    cols.append((idx + 1)[m])
    vals.append(-wy_hi[m])

    A = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n * n, n * n),
    ).tocsr()
    return A


def solve_elliptic_2d(
    a_nodes: np.ndarray,
    F_interior: np.ndarray,
    n_interior: int | None = None,
    *,
    averaging: str = "arithmetic",
) -> EllipticSolution2D:
    """Solve ``-div(a grad u) = F`` on (0,1)^2 with homogeneous Dirichlet data."""
    a_nodes = np.asarray(a_nodes, dtype=float)
    F = np.asarray(F_interior, dtype=float)
    if n_interior is None:
        n_interior = F.shape[0]
    if F.shape != (n_interior, n_interior):
        raise ValueError("F_interior must be square with side n_interior")
    A = assemble_elliptic_2d(a_nodes, n_interior, averaging=averaging)
    u_int = spsolve(A, F.ravel())
    n = n_interior
    U = np.zeros((n + 2, n + 2))
    U[1:-1, 1:-1] = u_int.reshape(n, n)
    x = np.linspace(0.0, 1.0, n + 2)
    return EllipticSolution2D(
        x=x,
        U=U,
        A=A,
        F=F,
        a_nodes=a_nodes,
        a_min=float(a_nodes.min()),
        a_max=float(a_nodes.max()),
        h=1.0 / (n + 1),
    )
