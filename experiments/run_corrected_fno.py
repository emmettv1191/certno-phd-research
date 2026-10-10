#!/usr/bin/env python
from __future__ import annotations
import json
import numpy as np
from scipy.sparse.linalg import cg

from certno.solvers.elliptic import solve_elliptic_1d, assemble_elliptic_1d
from certno.certificates import certify_elliptic_1d
from certno.certificates.stability import corrected_bound, analytic_coercivity_1d
from certno.metrics import coverage, false_negative_rate

try:
    import torch
    from certno.operators import FNO1d, train_operator, predict_operator
    from certno.operators.datasets import generate_elliptic_1d_dataset
    torch_ok = True
except Exception:
    torch_ok = False

def main():
    if not torch_ok:
        print(json.dumps({'ok':False}))
        return
    ds = generate_elliptic_1d_dataset(n_samples=80,n_interior=31,length_scale=0.2,a_min=0.1,a_max=0.8,forcing='sin',seed=0)
    m = FNO1d(in_channels=2,out_channels=1,width=20,modes=10,depth=3)
    train_operator(m,ds,epochs=40,batch_size=32,seed=0)
    dst = generate_elliptic_1d_dataset(n_samples=25,n_interior=31,length_scale=0.2,a_min=0.1,a_max=0.8,forcing='sin',seed=5)
    pred = predict_operator(m,dst)
    tol=5e-3
    b_a,b_c,errs=[],[],[]
    for i in range(dst.n_samples):
        a=dst.a[i]; f=dst.f[i]; uh=pred[i]
        A = assemble_elliptic_1d(a,dst.n_interior)
        sol=solve_elliptic_1d(a,f[1:-1],dst.n_interior)
        ca = certify_elliptic_1d(a,f[1:-1],uh[1:-1],method='analytic',a_min_assumed=0.05)
        r = f[1:-1]-A.dot(uh[1:-1])
        alpha = analytic_coercivity_1d(0.05,dst.n_interior)
        z,info = cg(A,r,maxiter=100,rtol=1e-10)
        res = r-A.dot(z)
        bc = float(np.linalg.norm(z)+np.linalg.norm(res)/max(alpha,1e-18))
        err = float(np.linalg.norm(uh-sol.u))
        b_a.append(ca.bound); b_c.append(bc); errs.append(err)
    print(json.dumps({
        'cov_a':coverage(np.array(b_a),np.array(errs)),
        'cov_c':coverage(np.array(b_c),np.array(errs)),
        'fnr_a':false_negative_rate(np.array(b_a),np.array(errs),tol),
        'fnr_c':false_negative_rate(np.array(b_c),np.array(errs),tol),
        'med_c_over_e':float(np.median([b/e if e>1e-12 else np.inf for b,e in zip(b_c,errs)])),
    }, indent=2))
if __name__=='__main__': main()
