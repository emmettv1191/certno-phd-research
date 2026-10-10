#!/usr/bin/env python
from __future__ import annotations
import json
import numpy as np
from certno.solvers.elliptic import solve_elliptic_1d, assemble_elliptic_1d
from certno.solvers.fields import sample_coefficient_1d
from certno.certificates import certify_elliptic_1d
from certno.certificates.stability import corrected_bound, analytic_coercivity_1d
from certno.metrics import coverage, false_negative_rate

def main():
    rng = np.random.default_rng(111)
    n = 31
    tol = 5e-3
    out = []
    for contrast in [1.5,3.0,5.0,10.0]:
        a_min,a_max = 0.1,0.1*contrast
        b_a,b_c,errs = [],[],[]
        acc_c = 0
        for _ in range(40):
            a = sample_coefficient_1d(n+2,a_min=a_min,a_max=a_max,length_scale=0.25,rng=rng)
            x = np.linspace(0,1,n+2); f = np.sin(2*np.pi*x)
            sol = solve_elliptic_1d(a,f[1:-1],n)
            A = assemble_elliptic_1d(a,n)
            u_hat = sol.u_interior + 0.005*rng.standard_normal(n)
            ca = certify_elliptic_1d(a,f[1:-1],u_hat,method='analytic',a_min_assumed=a_min)
            alpha = analytic_coercivity_1d(a_min,n)
            r = f[1:-1] - A.dot(u_hat)
            bc,z,res,info = corrected_bound(A,r,alpha,maxiter=120,tol=1e-10)
            err = float(np.linalg.norm(sol.u_interior-u_hat))
            b_a.append(ca.bound); b_c.append(bc); errs.append(err)
            if bc <= tol: acc_c +=1
        out.append({
            'contrast':contrast,
            'cov_a':coverage(np.array(b_a),np.array(errs)),
            'cov_c':coverage(np.array(b_c),np.array(errs)),
            'fnr_a':false_negative_rate(np.array(b_a),np.array(errs),tol),
            'fnr_c':false_negative_rate(np.array(b_c),np.array(errs),tol),
            'med_a_over_e':float(np.median([b/e if e>1e-12 else np.inf for b,e in zip(b_a,errs)])),
            'med_c_over_e':float(np.median([b/e if e>1e-12 else np.inf for b,e in zip(b_c,errs)])),
            'frac_acc_c':acc_c/40
        })
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
