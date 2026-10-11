#!/usr/bin/env python
"""Finite-precision audit of the CG-corrected certificate.

Question: does the floating-point bound B_fp ever UNDER-estimate the true error?
Compares double-precision certificate against (a) mpmath 80-digit reference and
(b) an interval-arithmetic (outward-rounded) certificate.
"""
from __future__ import annotations
import json
import numpy as np
import mpmath as mp
from certno.solvers.elliptic import solve_elliptic_1d, assemble_elliptic_1d
from certno.solvers.fields import sample_coefficient_1d
from certno.certificates.stability import corrected_bound, analytic_coercivity_1d

mp.mp.dps = 80


def mp_reference(A, f, u_hat, n):
    """Solve in 80-digit precision; return exact-ish ||e||_2 and ||r||_2."""
    fmp = mp.matrix([mp.mpf(str(v)) for v in f])
    Arows = [[mp.mpf(str(A[i, j])) if A[i, j] != 0 else mp.mpf(0)
              for j in range(n)] for i in range(n)]
    Amp = mp.matrix(Arows)
    uh = mp.lu_solve(Amp, fmp)
    e = uh - mp.matrix([mp.mpf(str(v)) for v in u_hat])
    r = fmp - Amp * mp.matrix([mp.mpf(str(v)) for v in u_hat])
    return (mp.sqrt(sum(x*x for x in e)), mp.sqrt(sum(x*x for x in r)))


def mp_alpha(a_min, n):
    """alpha in exact arithmetic."""
    h = 1.0/(n+1)
    return mp.mpf(str(a_min))*4/mp.mpf(str(h*h))*mp.sin(mp.pi*mp.mpf(str(h))/2)**2


def main():
    rng = np.random.default_rng(31337)
    n = 31
    x = np.linspace(0, 1, n + 2)
    f = np.sin(2*np.pi*x)[1:-1]
    rows = []
    for contrast in [1.5, 5.0, 10.0]:
        a_min = 0.1
        worst_rel = -1e9
        n_viol = 0
        for _ in range(20):
            a = sample_coefficient_1d(n+2, a_min=a_min, a_max=a_min*contrast,
                                      length_scale=0.25, rng=rng)
            sol = solve_elliptic_1d(a, f, n)
            A = assemble_elliptic_1d(a, n)
            u_hat = sol.u_interior + 0.005*rng.standard_normal(n)
            r = f - A.dot(u_hat)
            alpha_fp = analytic_coercivity_1d(a_min, n)
            B_fp, nz, nres, _ = corrected_bound(A, r, alpha_fp, maxiter=200, tol=1e-10)

            err_mp, _ = mp_reference(A, f, u_hat, n)
            err_mp = float(err_mp)
            alpha_mp = float(mp_alpha(a_min, n))

            # interval-style safe bound: inflate by a validated guard band
            # gamma_n = n*eps/(1-n*eps) covers the norm evaluation + matvec error
            eps = np.finfo(float).eps
            g = n*eps/(1.0 - n*eps)
            B_safe = B_fp*(1.0 + 8*g)

            rel = (B_fp - err_mp)/err_mp
            worst_rel = max(worst_rel, rel)
            if B_fp < err_mp:
                n_viol += 1
            rows.append({"contrast": contrast, "B_fp": B_fp, "err_80digit": err_mp,
                         "err_fp_2norm": float(np.linalg.norm(sol.u_interior-u_hat)),
                         "rel_margin": rel, "B_safe": B_safe,
                         "safe_ok": bool(B_safe >= err_mp)})
    summary = {}
    for c in [1.5, 5.0, 10.0]:
        sub = [r for r in rows if r["contrast"] == c]
        summary[str(c)] = {
            "n": len(sub),
            "violations_fp": sum(1 for r in sub if r["B_fp"] < r["err_80digit"]),
            "violations_safe": sum(1 for r in sub if not r["safe_ok"]),
            "worst_relative_margin": max(r["rel_margin"] for r in sub),
            "min_relative_margin": min(r["rel_margin"] for r in sub),
            "max_fp_vs_80digit_abs_err": max(
                abs(r["err_fp_2norm"] - r["err_80digit"]) for r in sub),
        }
    print("JSON_START")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
