#!/usr/bin/env python
from __future__ import annotations
import json
import numpy as np
from certno.solvers.elliptic import solve_elliptic_1d
from certno.solvers.fields import sample_coefficient_1d
from certno.certificates import certify_elliptic_1d, adaptive_verify

def main():
    rng = np.random.default_rng(8)
    n=31
    res=[]
    for tol in [0.05,0.1,0.5,1.0]:
        calls=0; accepted=0
        for _ in range(80):
            a = sample_coefficient_1d(n+2,a_min=0.1,a_max=0.6,length_scale=0.3,rng=rng)
            x=np.linspace(0,1,n+2); f=np.sin(np.pi*x)
            sol=solve_elliptic_1d(a,f[1:-1],n)
            u_hat = sol.u_interior + 0.01*rng.normal(size=n)
            c = certify_elliptic_1d(a,f[1:-1],u_hat,method='analytic',a_min_assumed=0.05)
            def ref(): return sol.u_interior
            v = adaptive_verify(c,tolerance=tol,solve_reference=ref,u_hat=u_hat)
            calls+=v.reference_calls; 
            if v.accepted: accepted+=1
        res.append({'tol':tol,'ref_per_80':calls,'accepted':accepted,'frac':accepted/80})
    print(json.dumps(res,indent=2))

if __name__=='__main__': main()
