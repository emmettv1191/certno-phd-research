#!/usr/bin/env python
"""Produce all real numbers for the manuscript tables/figures. JSON out."""
from __future__ import annotations
import json
import time
import numpy as np
from certno.solvers.elliptic import solve_elliptic_1d, assemble_elliptic_1d
from certno.solvers.fields import sample_coefficient_1d
from certno.certificates.stability import corrected_bound, analytic_coercivity_1d
from certno.metrics import coverage, false_negative_rate


def eff_stats(b, e):
    b = np.asarray(b, float); e = np.asarray(e, float)
    m = e > 1e-14
    r = b[m] / e[m]
    if r.size == 0:
        return {}
    return {
        "median": float(np.median(r)),
        "mean": float(np.mean(r)),
        "std": float(np.std(r, ddof=1)) if r.size > 1 else 0.0,
        "q25": float(np.percentile(r, 25)),
        "q75": float(np.percentile(r, 75)),
        "min": float(np.min(r)),
        "max": float(np.max(r)),
    }


def contrast_table(n=31, n_samp=200, seed=20240, tol=5e-3):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n + 2)
    f = np.sin(2 * np.pi * x)[1:-1]
    rows = []
    for c in [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        a_min = 0.1
        ba, bc, errs, iters, times = [], [], [], [], []
        for _ in range(n_samp):
            a = sample_coefficient_1d(n + 2, a_min=a_min, a_max=a_min * c,
                                      length_scale=0.25, rng=rng)
            sol = solve_elliptic_1d(a, f, n)
            A = assemble_elliptic_1d(a, n)
            u_hat = sol.u_interior + 0.005 * rng.standard_normal(n)
            r = f - A.dot(u_hat)
            alpha = analytic_coercivity_1d(a_min, n)
            t0 = time.perf_counter()
            B, nz, nres, info = corrected_bound(A, r, alpha, maxiter=200, tol=1e-10)
            times.append(time.perf_counter() - t0)
            iters.append(info if isinstance(info, int) else 0)
            ba.append(np.linalg.norm(r) / alpha)
            bc.append(B)
            errs.append(float(np.linalg.norm(sol.u_interior - u_hat)))
        rows.append({
            "contrast": c, "n_samples": n_samp,
            "coverage_global": float(coverage(np.array(ba), np.array(errs))),
            "coverage_cg": float(coverage(np.array(bc), np.array(errs))),
            "fnr_global": float(false_negative_rate(np.array(ba), np.array(errs), tol)),
            "fnr_cg": float(false_negative_rate(np.array(bc), np.array(errs), tol)),
            "eff_global": eff_stats(ba, errs),
            "eff_cg": eff_stats(bc, errs),
            "mean_err": float(np.mean(errs)),
            "mean_time_ms": float(np.mean(times) * 1e3),
            "median_time_ms": float(np.median(times) * 1e3),
        })
        print("contrast", c, "done", flush=True)
    return rows


def size_table(seeds=3, n_samp=30, seed=99, tol=5e-3):
    rng = np.random.default_rng(seed)
    rows = []
    for n in [15, 31, 63, 127]:
        x = np.linspace(0, 1, n + 2)
        f = np.sin(2 * np.pi * x)[1:-1]
        ts, bs, es, cs = [], [], [], []
        for _ in range(n_samp):
            a = sample_coefficient_1d(n + 2, a_min=0.1, a_max=1.0,
                                      length_scale=0.25, rng=rng)
            sol = solve_elliptic_1d(a, f, n)
            A = assemble_elliptic_1d(a, n)
            u_hat = sol.u_interior + 0.005 * rng.standard_normal(n)
            r = f - A.dot(u_hat)
            alpha = analytic_coercivity_1d(0.1, n)
            t0 = time.perf_counter()
            B, _, _, _ = corrected_bound(A, r, alpha, maxiter=200, tol=1e-10)
            ts.append((time.perf_counter() - t0) * 1e3)
            bs.append(B); es.append(float(np.linalg.norm(sol.u_interior - u_hat)))
            cs.append(float(coverage(np.array([B]), np.array([np.linalg.norm(sol.u_interior - u_hat)]))))
        rows.append({"n": n, "N": n, "mean_time_ms": float(np.mean(ts)),
                     "median_time_ms": float(np.median(ts)),
                     "coverage": float(np.mean(cs)),
                     "eff_cg": eff_stats(bs, es)})
        print("size", n, "done", flush=True)
    return rows


def main():
    out = {"contrast": contrast_table(), "size": size_table()}
    print("JSON_START")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
