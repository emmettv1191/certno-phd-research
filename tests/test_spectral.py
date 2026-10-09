"""Spectral solver tests."""

from __future__ import annotations

import numpy as np

from certno.solvers.spectral1d import solve_advection_diffusion_reaction_1d


def test_spectral_smooth_advection_diffusion():
    n = 64
    x = 2 * np.pi * np.arange(n) / n
    u0 = np.sin(x)
    sol = solve_advection_diffusion_reaction_1d(
        u0, nu=0.1, c=0.5, T=0.1, n_steps=500, n_out=2, dealias=True
    )
    # analytic: u(x,t)=exp(-nu t) sin(x - c t)
    t = sol.t[-1]
    u_exact = np.exp(-0.1 * t) * np.sin(x - 0.5 * t)
    err = np.linalg.norm(sol.u[-1] - u_exact) / np.linalg.norm(u_exact)
    assert err < 1e-3
