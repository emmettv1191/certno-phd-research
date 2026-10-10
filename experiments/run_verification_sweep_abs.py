#!/usr/bin/env python
from __future__ import annotations
import json
import numpy as np
from certno.solvers.elliptic import solve_elliptic_1d
from certno.solvers.fields import sample_coefficient_1d
from certno.certificates import certify_elliptic_1d, adaptive_verify

def main():
    rng = np.random.default_rng(7)
    n=31
    a = sample_coefficient_1d(n+2,a_min=0.1,a_max=0.8,length_scale=0.25,rng=rng)
    x=np.linspace(0,1,n+2); f=np.sin(2*np.pi*x)
    sol=solve_elliptic_1d(a,f[1:-1],n)
    res=[]
    for tol in [1e-3,2e-3,5e-3,1e-2,2e-2]:
        calls=0; accepted=0; fn=0
        for _ in range(120):
            u_hat = sol.u_interior + rng.normal(0,0.015,size=n)*(1+np.abs(sol.u_interior))
            c = certify_elliptic_1d(a,f[1:-1],u_hat,method='analytic',a_min_assumed=0.05)
            err = float(np.linalg.norm(sol.u_interior-u_hat))
            def ref(): return sol.u_interior
            v = adaptive_verify(c,tolerance=tol,solve_reference=ref,u_hat=u_hat)
            calls += v.reference_calls
            if v.accepted: accepted+=1
            if v.accepted and err > tol: fn+=1
        res.append({'tol':tol,'ref_per_120':calls,'accepted':accepted,'fn_if_accepted':fn,'frac_accepted':accepted/120})
    print(json.dumps(res,indent=2))

if __name__=='__main__':
    main()
