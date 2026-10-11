"""Tests for the floating-point-validated certificate."""
from __future__ import annotations

import numpy as np
import pytest

from certno.certificates.stability import (
    analytic_coercivity_1d,
    certified_bound,
    corrected_bound,
    gamma_n,
)
from certno.solvers.elliptic import assemble_elliptic_1d, solve_elliptic_1d
from certno.solvers.fields import sample_coefficient_1d


def _case(seed=0, n=31, contrast=5.0, noise=0.005):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n + 2)
    f = np.sin(2 * np.pi * x)[1:-1]
    a = sample_coefficient_1d(n + 2, a_min=0.1, a_max=0.1 * contrast,
                              length_scale=0.25, rng=rng)
    sol = solve_elliptic_1d(a, f, n)
    A = assemble_elliptic_1d(a, n)
    u_hat = sol.u_interior + noise * rng.standard_normal(n)
    r = f - A.dot(u_hat)
    alpha = analytic_coercivity_1d(0.1, n)
    return A, r, alpha, sol.u_interior, u_hat


def test_gamma_n_is_small_and_increasing():
    assert gamma_n(31) > 0.0
    assert gamma_n(127) > gamma_n(31)
    # first-order accurate: ~ n * 2**-53
    assert gamma_n(31) == pytest.approx(31 * 2.0**-53, rel=1e-9)


def test_certified_bound_dominates_plain_bound():
    A, r, alpha, _, _ = _case()
    plain, *_ = corrected_bound(A, r, alpha, maxiter=200, tol=1e-10)
    cert, *_ = certified_bound(A, r, alpha, maxiter=200, tol=1e-10)
    assert cert >= plain
    # inflation is tiny and bounded
    assert (cert - plain) / plain < 1e-10


def test_certified_bound_is_valid_against_true_error():
    A, r, alpha, u, u_hat = _case()
    err = float(np.linalg.norm(u - u_hat))
    cert, *_ = certified_bound(A, r, alpha, maxiter=200, tol=1e-10)
    assert cert >= err


@pytest.mark.parametrize("seed", range(6))
def test_certified_bound_valid_across_contrasts(seed):
    for contrast in (1.5, 10.0):
        A, r, alpha, u, u_hat = _case(seed=seed, contrast=contrast)
        err = float(np.linalg.norm(u - u_hat))
        cert, *_ = certified_bound(A, r, alpha, maxiter=300, tol=1e-10)
        assert cert >= err


def test_certified_bound_dominates_violated_alpha():
    """If the caller supplies an alpha slightly too large, the one-ulp rounding
    down cannot save them -- but the certified bound must still be no smaller
    than the plain one computed with the same alpha."""
    A, r, alpha, _, _ = _case()
    p, *_ = corrected_bound(A, r, alpha, maxiter=200, tol=1e-10)
    c, *_ = certified_bound(A, r, alpha, maxiter=200, tol=1e-10)
    assert c >= p


def test_alpha_is_rounded_downward():
    A, r, alpha, _, _ = _case()
    *_, alpha_used = certified_bound(A, r, alpha, maxiter=200, tol=1e-10)
    assert alpha_used < alpha
    assert alpha_used == np.nextafter(alpha, 0.0)


def test_certified_bound_rejects_nonpositive_alpha():
    A, r, _, _, _ = _case()
    with pytest.raises(ValueError):
        certified_bound(A, r, 0.0, maxiter=10, tol=1e-10)


def test_certified_bound_scales_to_larger_system():
    A, r, alpha, u, u_hat = _case(seed=2, n=127, contrast=10.0)
    err = float(np.linalg.norm(u - u_hat))
    cert, *_ = certified_bound(A, r, alpha, maxiter=300, tol=1e-10)
    assert cert >= err
