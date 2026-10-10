# Frozen benchmark

Revision: 619974b
Date: 2025-10-09
Environment: Python 3.14, PyTorch CPU, NumPy/SciPy

Experiments:
- run_looseness_corrected.py
- run_corrected_comparison.py
- run_corrected_fno.py
- run_corrected_cost2.py
- run_counter.py variants
- run_ood_abs.py

Parameters: seeds fixed; alpha = a_min * lambda_min(-Delta_h) (analytic lower bound); CG rtol=1e-10, maxiter=120-150.
