#!/usr/bin/env python
"""Certificate coverage and sharpness study (Problem A)."""
from __future__ import annotations

import json

import numpy as np

from certno.solvers.elliptic import solve_elliptic_1d
from certno.solvers.fields import sample_coefficient_1d
from certno.certificates import certify_elliptic_1d
from certno.metrics import summarize, relative_l2


def main() -> None:
    cfg = json.load(open("configs/elliptic.json"))
    n = cfg["n_interior"]
    tol = cfg["tolerance"]
    bounds = []
    errors = []
    rng = np.random.default_rng(7)
    for _ in range(512):
        a = sample_coefficient_1d(n + 2, a_min=cfg["a_min"], a_max=cfg["a_max"], length_scale=cfg["length_scale"], rng=rng)
        x = np.linspace(0, 1, n + 2)
        f = np.sin(3 * np.pi * x)
        sol = solve_elliptic_1d(a, f[1:-1], n)
        # model error
        u_hat = sol.u_interior + rng.normal(0, 0.02, size=sol.u_interior.size) * (1 + np.abs(sol.u_interior))
        cert = certify_elliptic_1d(a, f[1:-1], u_hat, method=cfg["certificate_method"], a_min_assumed=cfg["a_min_assumed"])
        bounds.append(cert.bound)
        errors.append(relative_l2(u_hat, sol.u_interior))
    stats = summarize(np.array(errors), np.array(bounds), tolerance=tol)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
