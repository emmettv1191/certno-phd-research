# Corrected results (absolute error)

1. Looseness (analytic vs discrete-exact), absolute L2
- Contrast ↑ => ratio ↑ (1.25x at 1.5 up to ~5.6x at 10). Bound safe but loose.

2. Validity
- Coverage_abs = 1.0, FNR_abs = 0.0 for all tested cases (ID, OOD, counterexamples).
- No case found with ||r|| small and ||e|| large.

3. Real surrogate (FNO)
- On ID test set, certificate gives coverage 1.0, FNR 0.0 for surrogate predictions.

4. Selective verification
- Safe policy; acceptance depends on tolerance and error magnitude.

Bottom line: the certificate is mathematically valid in the discrete SPD setting; the main weakness is looseness from the global a_min coercivity bound. Tightening while preserving safety is the next concrete research step.
