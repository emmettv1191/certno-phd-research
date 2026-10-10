#!/usr/bin/env python
from __future__ import annotations
import json
import time
import numpy as np
from scipy.sparse.linalg import cg
from certno.solvers.elliptic import assemble_elliptic_1d, solve_elliptic_1d
from certno.solvers.fields import sample_coefficient_1d
from certno.certificates.stability import analytic_coercivity_1d

def main():
    rng = np.random.RandomState(5)
    n=63
    times=[]; its=[]
    for _ in range(30):
        a = sample_coefficient_1d(n+2,a_min=0.1,a_max=1.0,length_scale=0.25,rng=rng)
        A = assemble_elliptic_1d(a,n)
        x=np.linspace(0,1,n+2); f=np.sin(2*np.pi*x)
        sol=solve_elliptic_1d(a,f[1:-1],n)
        u = sol.u_interior+0.01*rng.randn(n)
        r = f[1:-1]-A.dot(u)
        alpha = analytic_coercivity_1d(0.05,n)
        t0=time.perf_counter()
        z,info = cg(A,r,maxiter=120,rtol=1e-10)
        res = r-A.dot(z)
        bc = float(np.linalg.norm(z)+np.linalg.norm(res)/max(alpha,1e-18))
        times.append(time.perf_counter()-t0); its.append(info)
    print(json.dumps({'mean_t':float(np.mean(times)),'mean_it':float(np.mean(its)),'n':n}))
if __name__=='__main__': main()
