#!/usr/bin/env python
"""Tiny end-to-end smoke test.

Runs: 1D elliptic solve -> certificate -> operator training (very small).
"""
from __future__ import annotations

import json
import os

import numpy as np

from certno.solvers.elliptic import solve_elliptic_1d
from certno.solvers.fields import sample_coefficient_1d
from certno.certificates import certify_elliptic_1d
from certno.operators.datasets import generate_elliptic_1d_dataset, EllipticDataset1D

try:
    from certno.operators import FNO1d, train_operator, predict_operator
except Exception as e:
    FNO1d = None


def main() -> None:
    cfg = json.load(open("configs/elliptic.json"))
    # 1D certificate check
    n = int(cfg["n_interior"])
    x = np.linspace(0, 1, n + 2)
    a = sample_coefficient_1d(x.size, a_min=cfg["a_min"], a_max=cfg["a_max"], length_scale=cfg["length_scale"], rng=np.random.default_rng(123))
    f = np.sin(np.pi * x)
    sol = solve_elliptic_1d(a, f[1:-1], n)
    # perturb u_hat slightly
    u_hat = sol.u_interior.copy()
    u_hat = u_hat + 0.01 * np.sin(2 * np.pi * np.linspace(0, 1, u_hat.size))
    cert = certify_elliptic_1d(a, f[1:-1], u_hat, method=cfg["certificate_method"], a_min_assumed=cfg["a_min_assumed"])
    print("CERT:", cert.as_dict())

    # tiny operator train
    ds = generate_elliptic_1d_dataset(n_samples=16, n_interior=n, length_scale=cfg["length_scale"], a_min=cfg["a_min"], a_max=cfg["a_max"], forcing=cfg["forcing"], seed=0)
    if FNO1d is not None:
        model = FNO1d(in_channels=2, out_channels=1, width=16, modes=8, depth=2)
        hist = train_operator(model, ds, kind="fno", epochs=5, batch_size=8, seed=0, verbose=False)
        pred = predict_operator(model, ds, kind="fno")
        err = np.mean((pred - ds.u) ** 2) ** 0.5
        print("OP_TRAIN:", {"final_loss": hist["final_loss"], "rmse": float(err)})

    os.makedirs("results/runs", exist_ok=True)
    np.savez("results/runs/demo.npz", a=a, u=sol.u, cert=cert.as_dict())


if __name__ == "__main__":
    main()
