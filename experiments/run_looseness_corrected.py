#!/usr/bin/env python
from __future__ import annotations
import json
import numpy as np
from certno.solvers.elliptic import solve_elliptic_1d
from certno.solvers.fields import sample_coefficient_1d
from certno.certificates import certify_elliptic_1d
from certno.metrics import coverage, false_negative_rate, sharpness

def main():
    rng = np.random.default_rng(123)
    n = 31
    tol = 5e-3
    out = []
    for contrast in [1.5,2.0,3.0,4.0,5.0,6.0,8.0,10.0]:
        a_min,a_max = 0.1, 0.1*contrast
        ratios, bnds, errs = [], [], []
        for _ in range(60):
            a = sample_coefficient_1d(n+2, a_min=a_min, a_max=a_max, length_scale=0.25, rng=rng)
            x = np.linspace(0,1,n+2)
            f = np.sin(2*np.pi*x)
            sol = solve_elliptic_1d(a,f[1:-1],n)
            u_hat = sol.u_interior + 0.005*rng.standard_normal(size=n)
            ca = certify_elliptic_1d(a,f[1:-1],u_hat,method='analytic',a_min_assumed=a_min)
            cd = certify_elliptic_1d(a,f[1:-1],u_hat,method='discrete-exact',a_min_assumed=a_min)
            err = float(np.linalg.norm(sol.u_interior-u_hat))
            if cd.bound > 1e-18:
                ratios.append(ca.bound/cd.bound)
            bnds.append(ca.bound); errs.append(err)
        out.append({
            'contrast': contrast,
            'median_ratio_analytic_over_exact': float(np.median(ratios)),
            'mean_ratio': float(np.mean(ratios)),
            'coverage_abs': coverage(np.array(bnds), np.array(errs)),
            'fnr_abs': false_negative_rate(np.array(bnds), np.array(errs), tolerance=tol),
            'sharpness_abs_med': sharpness(np.array(bnds), np.array(errs)),
            'mean_bound': float(np.mean(bnds)),
            'mean_err_abs': float(np.mean(errs)),
        })
    print(json.dumps(out, indent=2))

if __name__=='__main__':
    main()
