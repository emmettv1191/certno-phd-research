"""Tests for elliptic PDE solvers (MMS, SPD properties, coercivity)."""

from __future__ import annotations

import numpy as np

from certno.solvers.elliptic import (
    assemble_elliptic_1d,
    assemble_elliptic_2d,
    solve_elliptic_1d,
    solve_elliptic_2d,
    laplacian_min_eig_1d,
    laplacian_min_eig_2d,
    coercivity_bound_1d,
)
from certno.solvers.manufactured import elliptic_manufactured_1d
from certno.solvers.fields import sample_coefficient_1d


def test_elliptic_mms_converges_like_second_order():
    # Check relative L2 error decreases as h^2 for smooth solution.
    errs = []
    for n in [15, 31, 63]:
        a_fn = lambda x: 1.0 + 0.2 * np.sin(2 * np.pi * x)
        u_fn = lambda x: x * (1 - x) * np.exp(x)
        a_nodes, f_int, u_star = elliptic_manufactured_1d(a_fn, u_fn, n)
        sol = solve_elliptic_1d(a_nodes, f_int, n)
        err = np.linalg.norm(sol.u - u_star) / np.linalg.norm(u_star)
        errs.append(err)
    # ratio should be roughly ( (63+1)/(31+1) )^2 etc
    assert errs[0] / errs[1] > 3.0
    assert errs[1] / errs[2] > 3.0


def test_elliptic_1d_spd():
    rng = np.random.default_rng(0)
    n = 31
    a = sample_coefficient_1d(n + 2, a_min=0.1, a_max=1.0, length_scale=0.3, rng=rng)
    A = assemble_elliptic_1d(a, n)
    eigs = np.linalg.eigvalsh(A.toarray())
    assert np.all(eigs > 0)


def test_elliptic_2d_spd():
    rng = np.random.default_rng(1)
    n = 15
    from certno.solvers.fields import sample_grf_2d

    field = sample_grf_2d(n + 2, length_scale=0.25, rng=rng)
    a = 0.1 + (1.0 - 0.1) / (1.0 + np.exp(-field))
    A = assemble_elliptic_2d(a, n)
    eigs = np.linalg.eigvalsh(A.toarray())
    assert np.all(eigs > 0)


def test_coercivity_bound_is_lower_bound_on_lambdamin():
    rng = np.random.default_rng(2)
    n = 31
    a = sample_coefficient_1d(n + 2, a_min=0.2, a_max=0.8, length_scale=0.2, rng=rng)
    A = assemble_elliptic_1d(a, n)
    lam_min = float(np.linalg.eigvalsh(A.toarray()).min())
    bound = coercivity_bound_1d(float(a.min()), n)
    assert bound <= lam_min + 1e-12


def test_laplacian_eigenvalues_close_to_exact():
    n = 31
    h = 1.0 / (n + 1)
    lam = laplacian_min_eig_1d(n)
    assert abs(lam - 4.0 * np.sin(np.pi * h / 2.0) ** 2 / h**2) < 1e-12
    lam2 = laplacian_min_eig_2d(n)
    assert abs(lam2 - 8.0 * np.sin(np.pi * h / 2.0) ** 2 / h**2) < 1e-12
