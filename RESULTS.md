# Results (corrected)

- Coverage: 1.0 for both bounds in all cases.
- FNR: 0.0 for both.
- Corrected bound is extremely tight (median ~1.0x error) vs analytic (very loose).
- Acceptance rate depends on tolerance; cost is CG iterations.

This matches the theoretical form: ||e|| <= ||z|| + ||r-Az||/alpha. With good z, first term dominates and is close to ||e||.
