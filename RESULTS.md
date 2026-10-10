# Full baseline comparison

Analytic (global a_min) vs CG-corrected bound. Metrics: coverage, FNR (tol 5e-3), median(effectivity)=median(B/||e||).

Across contrasts 1.5-10, synthetic and FNO errors:
- Both preserve coverage 1.0, FNR 0.0
- Effectivity: analytic ~300-1400x, corrected ~1.0x
- Cost: ~8ms CG per certificate (n=31), scales modestly

The corrected bound is rigorous given alpha = lambda_min(A_h) lower bound and controlled CG residual.
