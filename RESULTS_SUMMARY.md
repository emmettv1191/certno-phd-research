# Results summary (corrected with absolute error)

## Looseness study (analytic vs discrete-exact), absolute L2
With error measured as $||u_h - \hat{u}||_2$ and certificate as absolute bound:
- Contrast 1.5: median ratio ~2.1x, coverage 1.0, FNR 0.0 (tol 5e-3)
- Contrast 3: ratio ~3.0x, coverage 1.0, FNR 0.0
- Contrast 5: ratio ~4.0x, coverage 1.0, FNR 0.0
- Contrast 10: ratio ~5.6x, coverage 1.0, FNR 0.0

Conclusion: bound is valid (never violated) but increasingly loose with contrast.

## Counterexamples
Tested high-contrast + high-frequency perturbations: no case found with $||r||$ small and $||e||$ large; $bound >= ||e||$ holds for all cases. This matches $e=A_h^{-1}r$.

## OOD (absolute error)
ID and OOD both show coverage 1.0, FNR 0.0 at tol 5e-3.

## Selective verification (adaptive)
For small surrogate errors, certificate accepts only at very loose tolerances; at practical tolerances we fall back to reference. The policy is safe (no false negatives observed).

Key point: the earlier "sharpness/coverage" numbers using relative error were misleading. Using absolute error, the certificate is conservative but correct.
