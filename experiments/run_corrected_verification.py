#!/usr/bin/env python
from __future__ import annotations
import json
import numpy as np
from scipy.sparse.linalg import cg

from certno.solvers.elliptic import solve_elliptic_1d, assemble_elliptic_1d
from certno.solvers.fields import sample_coefficient_1d
from certno.certificates import certify_elliptic_1d, adaptive_verify
from certno.certificates.stability import analytic_coercivity_1d, corrected_certificate

def main():
    rng = np.random.default_rng(42)
    n=31
    # ID-like case
    a = sample_coefficient_1d(n+2,a_min=0.1,a_max=0.8,length_scale=0.25,rng=rng)
    x=np.linspace(0,1,n+2); f=np.sin(2*np.pi*x)
    sol = solve_elliptic_1d(a,f[1:-1],n)
    A = assemble_elliptic_1d(a,n)
    alpha = analytic_coercivity_1d(0.05,n)
    res=[]
    for tol in [1e-3,5e-3,1e-2]:
        calls=0; acc=0; fn=0
        for _ in range(40):
            uhat = sol.u_interior + 0.01*rng.normal(size=n)
            r = f[1:-1]-A.dot(uhat)
            bc,zn,resn,info = corrected_certificate(A,r,alpha,maxiter=120,tol=1e-10)
            err = float(np.linalg.norm(sol.u_interior-uhat))
            # build cert-like obj
            class C: pass
            c=C(); c.valid=True; c.bound=bc; c.assumptions_ok=True
            v = adaptive_verify(c,tolerance=tol,solve_reference=lambda: sol.u_interior,u_hat=uhat)
            calls += v.reference_calls
            if v.accepted: acc+=1
            if v.accepted and err>tol: fn+=1
        res.append({'tol':tol,'ref':calls,'acc':acc,'fn':fn})
    print(json.dumps(res,indent=2))
if __name__=='__main__': main()
