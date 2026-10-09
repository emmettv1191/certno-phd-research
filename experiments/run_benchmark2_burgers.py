#!/usr/bin/env python
"""Benchmark 2: Burgers equation (viscous). Long-horizon stability."""
from __future__ import annotations

import json

import numpy as np

from certno.solvers.spectral1d import solve_advection_diffusion_reaction_1d


def main() -> None:
    cfg = json.load(open("configs/burgers.json"))
    n = int(cfg["n"])
    nu = float(cfg["nu"])
    T = float(cfg["T"])
    n_steps = int(cfg["n_steps"])
    n_out = int(cfg["n_out"])
    x = 2 * np.pi * np.arange(n) / n
    u0 = -np.sin(x)
    sol = solve_advection_diffusion_reaction_1d(
        u0,
        nu=nu,
        nonlinear_advection=True,
        T=T,
        n_steps=n_steps,
        n_out=n_out,
        dealias=True,
    )
    # measure pointwise variation (proxy for shock steepening)
    var = float(np.var(sol.u[-1]))
    print(json.dumps({"final_var": var, "n_out": n_out, "T": T}, indent=2))


if __name__ == "__main__":
    main()
