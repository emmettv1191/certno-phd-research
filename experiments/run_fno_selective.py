#!/usr/bin/env python
"""Selective verification applied to REAL FNO predictions.

Fixes the ambiguity flagged in review: the acceptance table in the manuscript
was computed on SYNTHETIC perturbations, which must not be read as applying to
FNO predictions. This script produces the FNO-specific acceptance table.
"""
from __future__ import annotations
import json
import numpy as np
from certno.operators import FNO1d, train_operator, predict_operator
from certno.operators.datasets import generate_elliptic_1d_dataset
from certno.solvers.elliptic import solve_elliptic_1d, assemble_elliptic_1d
from certno.certificates.stability import certified_bound, analytic_coercivity_1d
from certno.metrics import coverage, false_negative_rate


def main():
    ds_train = generate_elliptic_1d_dataset(
        n_samples=256, n_interior=63, length_scale=0.2,
        a_min=0.1, a_max=0.8, forcing="sin", seed=0)
    model = FNO1d(in_channels=2, out_channels=1, width=32, modes=16, depth=4)
    train_operator(model, ds_train, kind="fno", epochs=150, batch_size=32, seed=0)

    splits = {
        "ID": dict(n_samples=80, n_interior=63, length_scale=0.2,
                   a_min=0.1, a_max=0.8, seed=123),
        "OOD contrast": dict(n_samples=80, n_interior=63, length_scale=0.2,
                             a_min=0.1, a_max=3.0, seed=321),
        "OOD scale": dict(n_samples=80, n_interior=63, length_scale=0.6,
                          a_min=0.1, a_max=0.8, seed=777),
    }
    out = {}
    for name, kw in splits.items():
        kw = dict(kw); seed = kw.pop("seed")
        ds = generate_elliptic_1d_dataset(seed=seed, **kw)
        pred = predict_operator(model, ds, kind="fno")
        ni = ds.n_interior
        bg, bc, errs = [], [], []
        for i in range(ds.n_samples):
            a, f, v = ds.a[i], ds.f[i], pred[i][1:-1]
            sol = solve_elliptic_1d(a, f[1:-1], ni)
            A = assemble_elliptic_1d(a, ni)
            r = f[1:-1] - A.dot(np.asarray(v, float))
            alpha = analytic_coercivity_1d(min(0.05, kw["a_min"]), ni)
            B, *_ = certified_bound(A, r, alpha, maxiter=400, tol=1e-11)
            bg.append(np.linalg.norm(r)/alpha); bc.append(B)
            errs.append(float(np.linalg.norm(sol.u_interior - np.asarray(v, float))))
        rows = []
        for tau in [1e-2, 5e-2, 1e-1, 5e-1, 1.0, 5.0]:
            rows.append({
                "tau": tau,
                "accept_global": float(np.mean(np.array(bg) <= tau)),
                "accept_certified": float(np.mean(np.array(bc) <= tau)),
                "fnr_global": float(false_negative_rate(np.array(bg), np.array(errs), tau)),
                "fnr_certified": float(false_negative_rate(np.array(bc), np.array(errs), tau)),
            })
        out[name] = {
            "n": len(errs), "mean_err": float(np.mean(errs)),
            "median_err": float(np.median(errs)),
            "max_err": float(np.max(errs)),
            "coverage_certified": float(coverage(np.array(bc), np.array(errs))),
            "rows": rows,
        }
        print(name, "mean_err", out[name]["mean_err"], flush=True)
    print("JSON_START")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
