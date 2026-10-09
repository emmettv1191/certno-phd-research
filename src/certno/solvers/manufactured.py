"""Method-of-manufactured-solutions helpers.

These utilities build right-hand sides (and sources) from a prescribed exact
solution so that solvers can be verified against a known answer.  Derivatives
are approximated with fourth-order compact stencils, which is more accurate
than the second-order solvers under test, so the measured convergence order is
attributable to the solver rather than to the manufactured data.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def d1_4th(y: np.ndarray, h: float) -> np.ndarray:
    """Fourth-order first derivative on a uniform grid (non-periodic)."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 5:
        raise ValueError("need at least 5 points for the 4th-order stencil")
    dy = np.empty_like(y)
    # Interior central stencil.
    dy[2:-2] = (-y[4:] + 8.0 * y[3:-1] - 8.0 * y[1:-3] + y[:-4]) / (12.0 * h)
    # One-sided stencils at the two ends.
    dy[0] = (-25 * y[0] + 48 * y[1] - 36 * y[2] + 16 * y[3] - 3 * y[4]) / (12 * h)
    dy[1] = (-3 * y[0] - 10 * y[1] + 18 * y[2] - 6 * y[3] + y[4]) / (12 * h)
    dy[-2] = (3 * y[-1] + 10 * y[-2] - 18 * y[-3] + 6 * y[-4] - y[-5]) / (12 * h)
    dy[-1] = (25 * y[-1] - 48 * y[-2] + 36 * y[-3] - 16 * y[-4] + 3 * y[-5]) / (12 * h)
    return dy


def d2_4th(y: np.ndarray, h: float) -> np.ndarray:
    """Fourth-order second derivative on a uniform grid (non-periodic)."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 6:
        raise ValueError("need at least 6 points for the 4th-order stencil")
    d2y = np.empty_like(y)
    d2y[2:-2] = (
        -y[4:] + 16.0 * y[3:-1] - 30.0 * y[2:-2] + 16.0 * y[1:-3] - y[:-4]
    ) / (12.0 * h**2)
    d2y[0] = (35 * y[0] - 104 * y[1] + 114 * y[2] - 56 * y[3] + 11 * y[4]) / (12 * h**2)
    d2y[1] = (11 * y[0] - 20 * y[1] + 6 * y[2] + 4 * y[3] - y[4]) / (12 * h**2)
    d2y[-2] = (11 * y[-1] - 20 * y[-2] + 6 * y[-3] + 4 * y[-4] - y[-5]) / (12 * h**2)
    d2y[-1] = (
        35 * y[-1] - 104 * y[-2] + 114 * y[-3] - 56 * y[-4] + 11 * y[-5]
    ) / (12 * h**2)
    return d2y


def manufactured_rhs_1d(
    a_nodes: np.ndarray,
    u_star_nodes: np.ndarray,
    h: float,
) -> np.ndarray:
    """Return interior values of ``f = -(a u*)'`` for ``- (a u')' = f``."""
    a = np.asarray(a_nodes, dtype=float).ravel()
    u = np.asarray(u_star_nodes, dtype=float).ravel()
    given_boundary = np.isclose(u[0], 0.0) and np.isclose(u[-1], 0.0)
    if not given_boundary:
        raise ValueError("manufactured u_star must vanish at both boundaries")
    f_full = -(d1_4th(a, h) * d1_4th(u, h) + a * d2_4th(u, h))
    return f_full[1:-1]


def manufactured_rhs_2d(
    a_nodes: np.ndarray,
    u_star_nodes: np.ndarray,
    h: float,
) -> np.ndarray:
    """Return interior values of ``f = -div(a grad u*)`` on a square grid."""
    a = np.asarray(a_nodes, dtype=float)
    u = np.asarray(u_star_nodes, dtype=float)
    # d/dx and d/dy via axis-wise fourth-order stencils.
    ax = np.empty_like(a)
    ux = np.empty_like(u)
    for j in range(a.shape[1]):
        ax[:, j] = d1_4th(a[:, j], h)
        ux[:, j] = d1_4th(u[:, j], h)
    ay = np.empty_like(a)
    uy = np.empty_like(u)
    for i in range(a.shape[0]):
        ay[i, :] = d1_4th(a[i, :], h)
        uy[i, :] = d1_4th(u[i, :], h)
    uxx = np.empty_like(u)
    uyy = np.empty_like(u)
    for j in range(u.shape[1]):
        uxx[:, j] = d2_4th(u[:, j], h)
    for i in range(u.shape[0]):
        uyy[i, :] = d2_4th(u[i, :], h)
    f = -(ax * ux + a * uxx + ay * uy + a * uyy)
    return f[1:-1, 1:-1]


def elliptic_manufactured_1d(
    a_fn: Callable[[np.ndarray], np.ndarray],
    u_fn: Callable[[np.ndarray], np.ndarray],
    n_interior: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(a_nodes, f_interior, u_exact_nodes)`` for a 1D manufactured test."""
    h = 1.0 / (n_interior + 1)
    x = np.linspace(0.0, 1.0, n_interior + 2)
    a_nodes = a_fn(x)
    u_exact = u_fn(x)
    f = manufactured_rhs_1d(a_nodes, u_exact, h)
    return a_nodes, f, u_exact
