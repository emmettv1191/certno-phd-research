"""Pseudo-spectral solver for 1D periodic advection-diffusion-reaction equations.

Solves

.. math::

    \\partial_t u = \\nu\\, u_{xx} - c\\, u_x + R(u) + s(x,t),
    \\qquad x \\in [0, 2\\pi),\\ u \\text{ periodic},

with an integrating-factor / IMEX scheme: the stiff linear part
:math:`L = \\nu\\,\\partial_{xx} - c\\,\\partial_x` is diagonal in Fourier space
and is integrated with Crank-Nicolson, while the (possibly nonlinear) term
:math:`N(u,t) = -u\\,u_x + R(u) + s` is treated explicitly with a midpoint
predictor.  The scheme is second order in time and spectral in space.

Setting ``c=0`` and a Fisher-type :math:`R` yields reaction-diffusion; setting
:math:`R=0` and supplying the self-advection term (``nonlinear_advection=True``)
yields viscous Burgers' equation.  Method-of-manufactured-solutions forcing is
supplied through ``source``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

ArrayLike = np.ndarray


@dataclass
class SpectralSolution1D:
    x: np.ndarray  # spatial grid, shape (N,)
    t: np.ndarray  # output times, shape (nt,)
    u: np.ndarray  # solution, shape (nt, N)
    n: int
    nu: float
    c: float


def _wavenumbers(n: int) -> np.ndarray:
    return np.fft.fftfreq(n, d=1.0 / n)  # integer wavenumbers


def _dealias(u_hat: np.ndarray, n: int, frac: float = 2.0 / 3.0) -> np.ndarray:
    k = np.abs(_wavenumbers(n))
    kmax = np.max(k)
    mask = k > frac * kmax
    u_hat = u_hat.copy()
    u_hat[mask] = 0.0
    return u_hat


def solve_advection_diffusion_reaction_1d(
    u0: np.ndarray,
    *,
    nu: float,
    c: float = 0.0,
    reaction: Callable[[np.ndarray], np.ndarray] | None = None,
    source: Callable[[np.ndarray, float], np.ndarray] | None = None,
    nonlinear_advection: bool = False,
    T: float = 1.0,
    n_steps: int = 2000,
    n_out: int = 11,
    dealias: bool = True,
) -> SpectralSolution1D:
    """Time-integrate the periodic 1D ADR system.

    Parameters
    ----------
    u0:
        Initial condition sampled on the uniform grid ``x_j = 2*pi*j/N``.
    nu, c:
        Diffusivity and constant advection speed.
    reaction:
        Pointwise reaction term ``R(u)`` (vectorised).
    source:
        Forcing ``s(x, t)`` (vectorised in ``x``).
    nonlinear_advection:
        If True, add the Burgers nonlinearity ``-u u_x``.
    T, n_steps, n_out:
        Final time, number of time steps, number of saved snapshots.
    dealias:
        Apply the 2/3 rule to the nonlinear product.
    """
    u0 = np.asarray(u0, dtype=float)
    n = u0.size
    x = 2.0 * np.pi * np.arange(n) / n
    k = _wavenumbers(n)

    # Linear symbol for L = nu d2/dx2 - c d/dx.
    L = -nu * k**2 - 1j * c * k
    dt = T / n_steps

    def N(u_hat: np.ndarray, t: float) -> np.ndarray:
        u = np.fft.ifft(u_hat).real
        acc = np.zeros_like(u)
        if nonlinear_advection:
            ux_hat = 1j * k * u_hat
            if dealias:
                u_hat_d = _dealias(u_hat, n)
                ux_hat = _dealias(1j * k * u_hat_d, n)
            ux = np.fft.ifft(ux_hat).real
            acc -= u * ux
        if reaction is not None:
            acc += reaction(u)
        if source is not None:
            acc += source(x, t)
        return np.fft.fft(acc)

    # IMEX Crank-Nicolson for the diagonal linear part:
    #   (1 - dt L/2) u^{n+1} = (1 + dt L/2) u^n + dt N(u^{n+1/2}).
    half = 0.5 * dt * L
    CN = (1.0 + half) / (1.0 - half)
    inv_lin = 1.0 / (1.0 - half)

    u_hat = np.fft.fft(u0)
    out_times = np.linspace(0.0, T, n_out)
    out_u = np.zeros((n_out, n))
    out_u[0] = u0
    out_idx = 1
    next_out = out_times[1] if n_out > 1 else T

    for step in range(1, n_steps + 1):
        t_n = (step - 1) * dt
        N_n = N(u_hat, t_n)
        u_star = u_hat + dt * N_n  # explicit midpoint predictor
        N_mid = N(u_star, t_n + 0.5 * dt)
        u_hat = CN * u_hat + dt * inv_lin * N_mid

        if n_out > 1 and step * dt >= next_out - 1e-12:
            while out_idx < n_out and step * dt >= out_times[out_idx] - 1e-12:
                out_u[out_idx] = np.fft.ifft(u_hat).real
                out_idx += 1
            if out_idx < n_out:
                next_out = out_times[out_idx]

    return SpectralSolution1D(
        x=x, t=out_times, u=out_u, n=n, nu=nu, c=c
    )
