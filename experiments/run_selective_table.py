#!/usr/bin/env python
"""Selective verification: acceptance rate / FNR / fallback savings vs tau."""
from __future__ import annotations
import json
import numpy as np
from certno.solvers.elliptic import solve_elliptic_1d, assemble_elliptic_1d
from certno.solvers.fields import sample_coefficient_1d
from certno.certificates.stability import corrected_bound, analytic_coercivity_1d
from certno.metrics import coverage, false_negative_rate


def main():
    n, n_samp, seed = 31, 200, 4242
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n + 2)
    f = np.sin(2 * np.pi * x)[1:-1]
    bnds_g, bnds_c, errs = [], [], []
    for _ in range(n_samp):
        a = sample_coefficient_1d(n + 2, a_min=0.1, a_max=1.0,
                                  length_scale=0.25, rng=rng)
        sol = solve_elliptic_1d(a, f, n)
        A = assemble_elliptic_1d(a, n)
        u_hat = sol.u_interior + 0.005 * rng.standard_normal(n)
        r = f - A.dot(u_hat)
        alpha = analytic_coercivity_1d(0.1, n)
        B, _, _, _ = corrected_bound(A, r, alpha, maxiter=200, tol=1e-10)
        bnds_g.append(np.linalg.norm(r) / alpha)
        bnds_c.append(B)
        errs.append(float(np.linalg.norm(sol.u_interior - u_hat)))
    bnds_g = np.array(bnds_g); bnds_c = np.array(bnds_c); errs = np.array(errs)

    rows = []
    for tau in [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]:
        ag, ac = bnds_g <= tau, bnds_c <= tau
        rows.append({
            "tau": tau,
            "accept_rate_global": float(ag.mean()),
            "accept_rate_cg": float(ac.mean()),
            "fnr_global": float(false_negative_rate(bnds_g, errs, tau)),
            "fnr_cg": float(false_negative_rate(bnds_c, errs, tau)),
            "coverage_global": float(coverage(bnds_g, errs)),
            "coverage_cg": float(coverage(bnds_c, errs)),
        })
    print("JSON_START")
    print(json.dumps({"n_samples": n_samp, "n": n, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
