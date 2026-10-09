#!/usr/bin/env python
"""Goal-oriented active learning study (Problem B)."""
from __future__ import annotations

import json

import numpy as np

from certno.solvers.elliptic import solve_elliptic_1d
from certno.solvers.fields import sample_coefficient_1d
from certno.active import RFFFeatures, BayesianLinearSurrogate, select_batch


def main() -> None:
    cfg = json.load(open("configs/active_learning.json"))
    n_nodes = 65  # interior+2
    n_pool = cfg["n_pool"]
    n_test = cfg["n_test"]
    n_init = cfg["n_init"]
    n_acq = cfg["n_acquire"]
    rng = np.random.default_rng(123)
    # generate random coeff inputs and QoI (max displacement proxy)
    X_pool = []
    Y_pool = []
    for _ in range(n_pool + n_test):
        a = sample_coefficient_1d(n_nodes, a_min=cfg["a_min"], a_max=cfg["a_max"], length_scale=cfg["length_scale"], rng=rng)
        x = np.linspace(0, 1, n_nodes)
        f = np.sin(np.pi * x)
        sol = solve_elliptic_1d(a, f[1:-1], n_nodes - 2)
        qoi = float(np.max(np.abs(sol.u_interior)))
        X_pool.append(a[1:-1])  # interior features
        Y_pool.append(qoi)
    X_pool = np.array(X_pool)
    Y_pool = np.array(Y_pool)
    X_test, Y_test = X_pool[:n_test], Y_pool[:n_test]
    X_train_pool, Y_train_pool = X_pool[n_test:], Y_pool[n_test:]

    # train initial surrogate
    feats = RFFFeatures(d_in=X_train_pool.shape[1], n_features=256, length_scale=0.5, rng=np.random.default_rng(1))
    sur = BayesianLinearSurrogate(feats, noise=1e-3, prior=1.0)
    init = rng.choice(X_train_pool.shape[0], n_init, replace=False)
    sur.fit(X_train_pool[init], Y_train_pool[init])
    res = {"policy": "goal_oriented", "rmse_init": float(np.sqrt(np.mean((sur.predict(X_test)[0] - Y_test) ** 2)))}
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
