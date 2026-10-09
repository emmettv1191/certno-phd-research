#!/usr/bin/env python
"""Failure-mode diagnostics study (Problem D)."""
from __future__ import annotations

import json

import numpy as np

from certno.solvers.elliptic import solve_elliptic_1d
from certno.solvers.fields import sample_coefficient_1d
from certno.diagnostics import residual_features, LogisticClassifier, identifiability_note


def main() -> None:
    cfg = json.load(open("configs/diagnostics.json"))
    n = cfg["n_interior"]
    n_nodes = n + 2
    rng = np.random.default_rng(42)
    X_list = []
    y_list = []
    # class 0: normal
    for _ in range(80):
        a = sample_coefficient_1d(n_nodes, a_min=cfg["a_min"], a_max=cfg["a_max"], length_scale=cfg["length_scale"], rng=rng)
        x = np.linspace(0, 1, n_nodes)
        f = np.sin(np.pi * x)
        sol = solve_elliptic_1d(a, f[1:-1], n)
        u_hat = sol.u_interior + 0.01 * rng.standard_normal(size=n)
        r = f[1:-1] - sol.A.dot(u_hat)
        X_list.append(residual_features(r, 1.0 / (n + 1)))
        y_list.append(0)
    # class 1: forcing mismatch
    for _ in range(80):
        a = sample_coefficient_1d(n_nodes, a_min=cfg["a_min"], a_max=cfg["a_max"], length_scale=cfg["length_scale"], rng=rng)
        x = np.linspace(0, 1, n_nodes)
        f = np.sin(2 * np.pi * x)
        sol = solve_elliptic_1d(a, f[1:-1], n)
        u_hat = sol.u_interior + 0.001 * rng.standard_normal(size=n)
        f_wrong = np.cos(np.pi * x)
        r = f_wrong[1:-1] - sol.A.dot(u_hat)
        X_list.append(residual_features(r, 1.0 / (n + 1)))
        y_list.append(1)
    X = np.array(X_list)
    y = np.array(y_list)
    clf = LogisticClassifier(n_features=X.shape[1], n_classes=2)
    clf.fit(X, y, epochs=cfg["epochs"], lr=0.5)
    acc = clf.accuracy(X, y)
    print(json.dumps({"train_accuracy": acc, "note": identifiability_note()}, indent=2))


if __name__ == "__main__":
    main()
