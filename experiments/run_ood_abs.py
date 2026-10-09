#!/usr/bin/env python
from __future__ import annotations
import json
import numpy as np
from certno.solvers.elliptic import solve_elliptic_1d
from certno.solvers.fields import sample_coefficient_1d
from certno.certificates import certify_elliptic_1d
from certno.metrics import coverage, false_negative_rate, sharpness

def main():
    rng = np.random.default_rng(10)
    n=31; tol=5e-3
    # ID
    b,e=[],[]
    for _ in range(150):
        a = sample_coefficient_1d(n+2,a_min=0.12,a_max=0.6,length_scale=0.25,rng=rng)
        x=np.linspace(0,1,n+2); f=np.sin(np.pi*x)
        sol=solve_elliptic_1d(a,f[1:-1],n)
        u_hat = sol.u_interior + 0.01*rng.normal(size=n)
        c = certify_elliptic_1d(a,f[1:-1],u_hat,method='analytic',a_min_assumed=0.08)
        b.append(c.bound); e.append(float(np.linalg.norm(sol.u_interior-u_hat)))
    idd={'cov':coverage(np.array(b),np.array(e)),'fnr':false_negative_rate(np.array(b),np.array(e),tol)}
    # OOD
    b,e=[],[]
    for _ in range(150):
        a = sample_coefficient_1d(n+2,a_min=0.05,a_max=1.8,length_scale=0.6,rng=rng)
        x=np.linspace(0,1,n+2); f=np.sin(2.5*np.pi*x)
        sol=solve_elliptic_1d(a,f[1:-1],n)
        u_hat = sol.u_interior + 0.015*rng.normal(size=n)
        c = certify_elliptic_1d(a,f[1:-1],u_hat,method='analytic',a_min_assumed=0.05)
        b.append(c.bound); e.append(float(np.linalg.norm(sol.u_interior-u_hat)))
    ood={'cov':coverage(np.array(b),np.array(e)),'fnr':false_negative_rate(np.array(b),np.array(e),tol)}
    print(json.dumps({'id':idd,'ood':ood},indent=2))

if __name__=='__main__': main()
