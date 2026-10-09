# Summary of results (completed experiments)

1. Looseness (analytic / discrete-exact)
- contrast 1.5: ~2.2x, contrast 3: ~3.0x, contrast 5: ~4.0x, contrast 10: ~5.4x
- Bound becomes looser as a_max/a_min increases (expected from global coercivity).

2. OOD vs ID (tolerance 5e-3)
- ID: coverage 1.0, FNR 0.0, mean bound large but safe
- OOD (larger contrast, longer length scale, different forcing): coverage remains 1.0, FNR 0.0; bounds grow.

3. Selective verification sweep
- At very loose tolerances (0.1+) many predictions accepted; at stricter tolerances more fall back to reference.

4. Goal-oriented active learning (QoI = max |u|)
- Goal-oriented/uncertainty/uniform all reduce RMSE over rounds; differences are small in this RFF surrogate setup (confirms implementation works).

5. Identifiability
- Scaling (c a,c f) vs (a,f): residual features allow separation in synthetic noisy setting but the mathematical identifiability note stands; true indistinguishability from solution/residual alone is a fundamental limit.

All tests pass (18/18). Repository: https://github.com/emmettv1191/certno-phd-research
