#!/usr/bin/env python
from __future__ import annotations
import json
import numpy as np
try:
    import torch
    from certno.operators import FNO1d, train_operator, predict_operator
    from certno.operators.datasets import generate_elliptic_1d_dataset
except Exception:
    torch = None

from certno.solvers.elliptic import solve_elliptic_1d
from certno.solvers.fields import sample_coefficient_1d
from certno.certificates import certify_elliptic_1d
from certno.metrics import coverage, false_negative_rate

def main():
    if torch is None:
        print(json.dumps({'ok':False,'msg':'torch not available'}))
        return
    # train small but larger than demo
    ds_train = generate_elliptic_1d_dataset(n_samples=256, n_interior=63, length_scale=0.2, a_min=0.1, a_max=0.8, forcing='sin', seed=0)
    model = FNO1d(in_channels=2,out_channels=1,width=32,modes=16,depth=4)
    train_operator(model, ds_train, kind='fno', epochs=150, batch_size=32, seed=0)
    # test ID
    ds_test = generate_elliptic_1d_dataset(n_samples=80, n_interior=63, length_scale=0.2, a_min=0.1, a_max=0.8, forcing='sin', seed=123)
    pred = predict_operator(model, ds_test, kind='fno')
    bnds,errs=[],[]
    for i in range(ds_test.n_samples):
        a=ds_test.a[i]; f=ds_test.f[i]; u=ds_test.u[i]; uhat=pred[i]
        sol = solve_elliptic_1d(a,f[1:-1],ds_test.n_interior)
        c = certify_elliptic_1d(a,f[1:-1],uhat[1:-1],method='analytic',a_min_assumed=0.05)
        bnds.append(c.bound); errs.append(float(np.linalg.norm(uhat-sol.u)))
    tol=5e-3
    print(json.dumps({'n':len(bnds),'coverage':coverage(np.array(bnds),np.array(errs)),'fnr':false_negative_rate(np.array(bnds),np.array(errs),tol),'mean_err':float(np.mean(errs)),'mean_bound':float(np.mean(bnds))}, indent=2))

if __name__=='__main__':
    main()
