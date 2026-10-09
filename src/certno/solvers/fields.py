"""Random-field generators for PDE coefficients (train / shift distributions).

Coefficients are sampled from a Karhunen-Loeve expansion of a Matern-type
covariance built from the sine basis on :math:`[0,1]` (1D) or
:math:`[0,1]^2` (2D, separable tensor product).  Because the basis vanishes on
the boundary, the resulting fields are smooth and inexpensive to generate
without a dense covariance Cholesky factorisation -- important on a laptop.

The mapping from a standardised field :math:`g` to a positive coefficient is a
logistic map,

.. math::

    a(x) = a_{\\min} + (a_{\\max}-a_{\\min})\\,\\sigma(g(x)),

so that :math:`a_{\\min} \\le a \\le a_{\\max}` holds by construction.  This
keeps the coercivity constant entering the certificates well defined and lets
us control contrast precisely through the ratio :math:`a_{\\max}/a_{\\min}`.
"""

from __future__ import annotations

import numpy as np


def _kl_spectrum(n_modes: int, length_scale: float, smoothness: float, amplitude: float):
    k = np.arange(1, n_modes + 1, dtype=float)
    lam = amplitude**2 * (1.0 + (k * np.pi * length_scale) ** 2) ** (-smoothness)
    return np.sqrt(lam)


def _logistic(g: np.ndarray, a_min: float, a_max: float) -> np.ndarray:
    s = 1.0 / (1.0 + np.exp(-g))
    return a_min + (a_max - a_min) * s


def sample_grf_1d(
    n_nodes: int,
    *,
    length_scale: float = 0.2,
    smoothness: float = 1.5,
    amplitude: float = 1.0,
    n_modes: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a zero-mean standardised random field on ``n_nodes`` nodes of [0,1]."""
    rng = rng or np.random.default_rng()
    if n_modes is None:
        n_modes = min(64, n_nodes)
    x = np.linspace(0.0, 1.0, n_nodes)
    k = np.arange(1, n_modes + 1, dtype=float)
    basis = np.sqrt(2.0) * np.sin(np.outer(x, k) * np.pi)  # (n_nodes, n_modes)
    coef = _kl_spectrum(n_modes, length_scale, smoothness, amplitude)
    xi = rng.standard_normal(n_modes)
    return basis @ (coef * xi)


def coefficient_from_field_1d(
    field: np.ndarray, *, a_min: float = 0.1, a_max: float = 1.0
) -> np.ndarray:
    return _logistic(np.asarray(field, dtype=float), a_min, a_max)


def sample_grf_2d(
    n_nodes: int,
    *,
    length_scale: float = 0.2,
    smoothness: float = 1.5,
    amplitude: float = 1.0,
    n_modes: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a zero-mean standardised random field on an ``(n,n)`` grid of [0,1]^2."""
    rng = rng or np.random.default_rng()
    if n_modes is None:
        n_modes = min(24, n_nodes)
    x = np.linspace(0.0, 1.0, n_nodes)
    k = np.arange(1, n_modes + 1, dtype=float)
    basis = np.sqrt(2.0) * np.sin(np.outer(x, k) * np.pi)  # (n, K)
    coef = _kl_spectrum(n_modes, length_scale, smoothness, amplitude)
    A = basis * coef[None, :]  # sqrt(lambda_k) phi_k(x_i)
    Z = rng.standard_normal((n_modes, n_modes))
    return A @ Z @ A.T


def sample_grf_2d_flat(
    n_nodes: int,
    *,
    length_scale: float = 0.2,
    smoothness: float = 1.5,
    amplitude: float = 1.0,
    a_min: float = 0.1,
    a_max: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Convenience wrapper returning a positive coefficient field on a 2D grid."""
    field = sample_grf_2d(
        n_nodes,
        length_scale=length_scale,
        smoothness=smoothness,
        amplitude=amplitude,
        rng=rng,
    )
    return _logistic(field, a_min, a_max)


def sample_coefficient_1d(
    n_nodes: int,
    *,
    length_scale: float = 0.2,
    a_min: float = 0.1,
    a_max: float = 1.0,
    smoothness: float = 1.5,
    amplitude: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a positive coefficient field on ``n_nodes`` nodes of [0,1]."""
    field = sample_grf_1d(
        n_nodes,
        length_scale=length_scale,
        smoothness=smoothness,
        amplitude=amplitude,
        rng=rng,
    )
    return coefficient_from_field_1d(field, a_min=a_min, a_max=a_max)
