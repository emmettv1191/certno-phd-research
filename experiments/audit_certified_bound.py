#!/usr/bin/env python
"""Audit: certified_bound vs 80-digit reference. Confirms the guard band
converts the certificate from 'empirically safe' to 'validated'."""
from __future__ import annotations
import json
import numpy as np
import mpmath as mp
from certno.solvers.elliptic import solve_elliptic_1d, assemble_elliptic_1d
from certno.solvers.fields import sample_coefficient_1d
from certno.certificates.stability import (
    corrected_bound, certified_bound, analytic_coercivity_1d, gamma_n)

mp.mp.dps = 80


def err_80(A, f, u_hat, n):
    Amp = mp.matrix([[mp.mpf(float(A[i, j])) if A[i, j] != 0 else mp.mpf(0)
                      for j in range(n)] for i in range(n)])
    fmp = mp.matrix([mp.mpf(float(v)) for v in f])
    uh = mp.lu_solve(Amp, fmp)
    e = uh - mp.matrix([mp.mpf(float(v)) for v in u_hat])
    return float(mp.sqrt(sum(x*x for x in e)))


def main():
    rng = np.random.default_rng(31337)
    out = {}
    for n in [31, 127]:
        x = np.linspace(0, 1, n + 2)
        f = np.sin(2*np.pi*x)[1:-1]
        for contrast in [1.5, 10.0]:
            a_min = 0.1
            v_plain = v_cert = 0
            infl = []
            margins_plain = []
            for _ in range(15):
                a = sample_coefficient_1d(n+2, a_min=a_min, a_max=a_min*contrast,
                                          length_scale=0.25, rng=rng)
                sol = solve_elliptic_1d(a, f, n)
                A = assemble_elliptic_1d(a, n)
                uh = sol.u_interior + 0.005*rng.standard_normal(n)
                r = f - A.dot(uh)
                alpha = analytic_coercivity_1d(a_min, n)
                Bp, *_ = corrected_bound(A, r, alpha, maxiter=300, tol=1e-10)
                Bc, *_ = certified_bound(A, r, alpha, maxiter=300, tol=1e-10)
                e = err_80(A, f, uh, n)
                if Bp < e:
                    v_plain += 1
                if Bc < e:
                    v_cert += 1
                infl.append((Bc - Bp)/Bp)
                margins_plain.append((Bp - e)/e)
            key = f"n{n}_c{contrast}"
            out[key] = {
                "n_trials": 15,
                "violations_plain": v_plain,
                "violations_certified": v_cert,
                "gamma_n": gamma_n(n),
                "max_relative_inflation": max(infl),
                "min_relative_margin_plain": min(margins_plain),
            }
            print(key, out[key], flush=True)
    print("JSON_START")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
