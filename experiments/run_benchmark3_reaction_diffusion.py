#!/usr/bin/env python
"""Benchmark 3: Reaction-diffusion transfer."""
from __future__ import annotations

import json

import numpy as np

from certno.solvers.spectral1d import solve_advection_diffusion_reaction_1d


def main() -> None:
    cfg = json.load(open("configs/reaction_diffusion.json"))
    n = int(cfg["n"])
    nu = float(cfg["nu"])
    T = float(cfg["T"])
    n_steps = int(cfg["n_steps"])
    n_out = int(cfg["n_out"])
    x = 2 * np.pi * np.arange(n) / n
    u0 = 0.5 + 0.1 * np.sin(x)
    sol = solve_advection_diffusion_reaction_1d(
        u0,
        nu=nu,
        reaction=lambda u: u * (1 - u),
        T=T,
        n_steps=n_steps,
        n_out=n_out,
        dealias=True,
    )
    final_mean = float(sol.u[-1].mean())
    print(json.dumps({"final_mean": final_mean}, indent=2))


if __name__ == "__main__":
    main()
