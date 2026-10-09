#!/usr/bin/env python
"""Benchmark 1: Elliptic diffusion. Solver verification + certificate coverage."""
from __future__ import annotations

import json

import numpy as np

from certno.solvers.elliptic import solve_elliptic_1d
from certno.solvers.fields import sample_coefficient_1d
from certno.solvers.manufactured import elliptic_manufactured_1d
from certno.certificates import certify_elliptic_1d
from certno.metrics import summarize, relative_l2


def manufactured_test(n=63) -> float:
    # smooth manufactured solution
    a_fn = lambda x: 1.0 + 0.3 * np.sin(2 * np.pi * x)
    u_fn = lambda x: x * (1 - x) * np.exp(x)
    a_nodes, f_int, u_star = elliptic_manufactured_1d(a_fn, u_fn, n)
    sol = solve_elliptic_1d(a_nodes, f_int, n)
    return relative_l2(sol.u, u_star)


def main() -> None:
    cfg = json.load(open("configs/elliptic.json"))
    n = int(cfg["n_interior"])
    tol = float(cfg["tolerance"])
    # convergence/manufactured sanity
    errs = []
    for N in [15, 31, 63]:
        errs.append(manufactured_test(N))
    # certificate coverage
    bounds = []
    errors = []
    rng = np.random.default_rng(42)
    for _ in range(256):
        a = sample_coefficient_1d(n + 2, a_min=cfg["a_min"], a_max=cfg["a_max"], length_scale=cfg["length_scale"], rng=rng)
        x = np.linspace(0, 1, n + 2)
        f = np.sin(2 * np.pi * x)
        sol = solve_elliptic_1d(a, f[1:-1], n)
        u_hat = sol.u_interior + rng.normal(0, 0.01, size=sol.u_interior.size)
        cert = certify_elliptic_1d(a, f[1:-1], u_hat, method=cfg["certificate_method"], a_min_assumed=cfg["a_min_assumed"])
        bounds.append(cert.bound)
        errors.append(relative_l2(u_hat, sol.u_interior))
    stats = summarize(np.array(errors), np.array(bounds), tolerance=tol)
    print(json.dumps({"mms_rel_l2": errs, "cert_stats": stats}, indent=2))


if __name__ == "__main__":
    main()
