#!/usr/bin/env python
"""Certify REAL FNO predictions with BOTH the global analytic bound and the
CG-corrected bound, on ID and OOD test sets. Emits real JSON numbers."""
from __future__ import annotations
import json
import numpy as np
from certno.operators import FNO1d, train_operator, predict_operator
from certno.operators.datasets import generate_elliptic_1d_dataset
from certno.solvers.elliptic import solve_elliptic_1d, assemble_elliptic_1d
from certno.certificates.stability import corrected_bound, analytic_coercivity_1d
from certno.metrics import coverage, false_negative_rate


def stats(bnds, errs, tol):
    b = np.asarray(bnds, dtype=float)
    e = np.asarray(errs, dtype=float)
    mask = e > 1e-14
    eff = b[mask] / e[mask]
    return {
        "n": int(len(b)),
        "coverage": float(coverage(b, e)),
        "fnr": float(false_negative_rate(b, e, tol)),
        "mean_err": float(np.mean(e)),
        "mean_bound": float(np.mean(b)),
        "median_effectivity": float(np.median(eff)) if eff.size else None,
        "mean_effectivity": float(np.mean(eff)) if eff.size else None,
        "min_effectivity": float(np.min(eff)) if eff.size else None,
    }


def certify_split(ds, pred, a_min_assumed, maxiter=400, tol=1e-11):
    bnds_a, bnds_c, errs = [], [], []
    for i in range(ds.n_samples):
        a, f, uhat = ds.a[i], ds.f[i], pred[i]
        ni = ds.n_interior
        sol = solve_elliptic_1d(a, f[1:-1], ni)
        A = assemble_elliptic_1d(a, ni)
        v = np.asarray(uhat[1:-1], dtype=float)
        r = f[1:-1] - A.dot(v)
        alpha = analytic_coercivity_1d(a_min_assumed, ni)
        bc, _, _, _ = corrected_bound(A, r, alpha, maxiter=maxiter, tol=tol)
        bnds_a.append(np.linalg.norm(r) / alpha)
        bnds_c.append(bc)
        errs.append(float(np.linalg.norm(sol.u_interior - v)))
    return bnds_a, bnds_c, errs


def main():
    tol = 5e-3
    out = {}
    ds_train = generate_elliptic_1d_dataset(
        n_samples=256, n_interior=63, length_scale=0.2,
        a_min=0.1, a_max=0.8, forcing="sin", seed=0)
    model = FNO1d(in_channels=2, out_channels=1, width=32, modes=16, depth=4)
    train_operator(model, ds_train, kind="fno", epochs=150, batch_size=32, seed=0)

    splits = {
        "ID": dict(n_samples=80, n_interior=63, length_scale=0.2,
                   a_min=0.1, a_max=0.8, seed=123),
        "OOD_contrast": dict(n_samples=80, n_interior=63, length_scale=0.2,
                             a_min=0.1, a_max=3.0, seed=321),
        "OOD_scale": dict(n_samples=80, n_interior=63, length_scale=0.6,
                          a_min=0.1, a_max=0.8, seed=777),
    }
    for name, kw in splits.items():
        seed = kw.pop("seed")
        ds = generate_elliptic_1d_dataset(seed=seed, **kw)
        pred = predict_operator(model, ds, kind="fno")
        a_assumed = min(0.05, kw["a_min"])
        ba, bc, errs = certify_split(ds, pred, a_assumed)
        out[name] = {
            "analytic": stats(ba, errs, tol),
            "cg_corrected": stats(bc, errs, tol),
            "a_min_assumed": a_assumed,
            "true_a_min": kw["a_min"],
        }
        print(name, json.dumps(out[name], indent=2))
    print("RESULTS_JSON_START")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
