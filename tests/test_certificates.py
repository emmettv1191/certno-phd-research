"""Tests for a posteriori certificates."""

from __future__ import annotations

import numpy as np

from certno.solvers.elliptic import solve_elliptic_1d
from certno.solvers.fields import sample_coefficient_1d
from certno.certificates import certify_elliptic_1d, certify_elliptic_2d
from certno.solvers.elliptic import solve_elliptic_2d


def test_residual_zero_gives_zero_bound():
    rng = np.random.default_rng(0)
    n = 15
    a = sample_coefficient_1d(n + 2, a_min=0.1, a_max=1.0, rng=rng)
    x = np.linspace(0, 1, n + 2)
    f = np.sin(np.pi * x)
    sol = solve_elliptic_1d(a, f[1:-1], n)
    cert = certify_elliptic_1d(a, f[1:-1], sol.u_interior, method="analytic")
    assert cert.bound < 1e-10
    assert cert.residual_norm < 1e-10


def test_certificate_bounds_actual_error():
    rng = np.random.default_rng(5)
    n = 31
    a = sample_coefficient_1d(n + 2, a_min=0.15, a_max=0.9, length_scale=0.25, rng=rng)
    x = np.linspace(0, 1, n + 2)
    f = np.sin(2 * np.pi * x)
    sol = solve_elliptic_1d(a, f[1:-1], n)
    u_hat = sol.u_interior + 0.005 * rng.standard_normal(size=n)
    cert = certify_elliptic_1d(a, f[1:-1], u_hat, method="analytic", a_min_assumed=0.1)
    err = np.linalg.norm(sol.u_interior - u_hat)
    assert cert.bound >= err - 1e-10


def test_analytic_vs_discrete_exact_ordered():
    rng = np.random.default_rng(6)
    n = 23
    a = sample_coefficient_1d(n + 2, a_min=0.2, a_max=0.8, rng=rng)
    x = np.linspace(0, 1, n + 2)
    f = np.cos(np.pi * x)
    sol = solve_elliptic_1d(a, f[1:-1], n)
    u_hat = sol.u_interior + 0.01 * rng.standard_normal(size=n)
    cert_a = certify_elliptic_1d(a, f[1:-1], u_hat, method="analytic")
    cert_d = certify_elliptic_1d(a, f[1:-1], u_hat, method="discrete-exact")
    # analytic bound uses lower bound on lambda_min -> factor >= exact
    assert cert_a.stability_factor >= cert_d.stability_factor - 1e-12
    assert cert_a.bound >= cert_d.bound - 1e-12


def test_hypothesis_violation_flagged():
    # Construct a case where a_min_actual < a_min_assumed
    n = 19
    x = np.linspace(0, 1, n + 2)
    a = 0.05 * np.ones_like(x)
    a[5:15] = 0.2  # but overall min is 0.05
    f = np.sin(np.pi * x)
    sol = solve_elliptic_1d(a, f[1:-1], n)
    u_hat = sol.u_interior
    cert = certify_elliptic_1d(a, f[1:-1], u_hat, method="analytic", a_min_assumed=0.1)
    assert not cert.assumptions_ok


def test_2d_certificate_basic():
    rng = np.random.default_rng(8)
    n = 11
    from certno.solvers.fields import sample_grf_2d

    field = sample_grf_2d(n + 2, length_scale=0.3, rng=rng)
    a = 0.25 + 0.6 / (1.0 + np.exp(-field))  # a_min ~ 0.25
    x = np.linspace(0, 1, n + 2)
    F = np.sin(np.pi * x[:, None]) * np.sin(np.pi * x[None, :])
    F = F[1:-1, 1:-1]
    sol = solve_elliptic_2d(a, F, n)
    cert = certify_elliptic_2d(a, F, sol.U_interior, method="analytic", a_min_assumed=0.2)
    assert cert.bound >= 0.0
    assert cert.residual_norm < 1e-7
    assert cert.bound < 1e-6
