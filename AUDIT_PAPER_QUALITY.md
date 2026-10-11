# Publication-Quality Audit — `certno-phd-research`

**Auditor:** automated manuscript audit
**Date:** 2026-10-10
**Revision audited:** `15b16d5` (pre-fix), `HEAD` (post-fix)
**Verdict:** previous drafts were **NOT submittable**. Root causes and remediations below.

---

## Executive summary

Three defects were severe enough to invalidate the manuscript's central claims. All
have been remediated by re-running experiments and rewriting the affected sections.
Two further issues are **conceptual and cannot be fixed by more experiments**; they
are now disclosed explicitly in the paper rather than hidden.

| Severity | Finding | Status |
|---|---|---|
| **Critical** | All table/figure numbers were hand-written, never computed | **Fixed** — tables regenerated from real runs |
| **Critical** | The "surrogate coverage" experiment used the *global* bound, not the CG bound; the near-sharp surrogate claim was never actually measured | **Fixed** — new `run_corrected_fno_cert.py` |
| **Critical** | Effectivity ≈ 1.0 is near-tautological given the certified target is the algebraic error | **Disclosed** — now stated as the central limitation |
| **Major** | Iteration counting was broken (`mean_iter = 0.0`) | **Fixed** |
| **Major** | Figures generated from hard-coded arrays, not from data | **Fixed** — regenerated from JSON |
| **Major** | Bibliography missing all 2024–2026 work on neural-operator verification | **Fixed** — 6 current references added |
| **Moderate** | 2D experiments claimed in text but absent from results | **Disclosed** — moved to future work |
| **Moderate** | No adversarial/counterexample experiment actually present | **Disclosed** — future work |
| **Minor** | No conservative rounding of floating-point certificate | **Disclosed** — future work |

---

## 1. Critical: fabricated quantitative results

### Finding
The previous manuscript reported, e.g., "median effectivity 312.3 / 499.5 / 767.6 /
1361.1" for contrasts 1.5/3/5/10. These numbers were typed by hand into the `.tex`
source. The `results/` tree contained no artifacts — only `demo.npz`. No script had
been executed to produce them. `results/data/` was empty.

### Remediation
Created `experiments/run_paper_tables.py` (200 samples/contrast, seed 20240) and
`experiments/run_selective_table.py` (200 samples, seed 4242). Real measured values:

| contrast | cov. global | cov. CG | FNR CG | eff. global (median) | eff. CG (median) |
|---|---|---|---|---|---|
| 1.5 | 1.00 | 1.00 | 0.00 | 316.1 | 1.000000 |
| 2.0 | 1.00 | 1.00 | 0.00 | 375.9 | 1.000000 |
| 3.0 | 1.00 | 1.00 | 0.00 | 508.3 | 1.000000 |
| 4.0 | 1.00 | 1.00 | 0.00 | 648.6 | 1.000000 |
| 5.0 | 1.00 | 1.00 | 0.00 | 744.8 | 1.000000 |
| 6.0 | 1.00 | 1.00 | 0.00 | 910.4 | 1.000000 |
| 8.0 | 1.00 | 1.00 | 0.00 | 1126.8 | 1.000000 |
| 10.0 | 1.00 | 1.00 | 0.00 | 1429.5 | 1.000000 |

Real FNO certification (80 samples/split, `run_corrected_fno_cert.py`):

| split | mean err | global eff. (median) | CG eff. (median) | cov. | FNR |
|---|---|---|---|---|---|
| ID | 0.0283 | 993.7 | 1.0000 | 1.00 | 0.00 |
| OOD contrast | 0.5457 | 1469.6 | 1.0000 | 1.00 | 0.00 |
| OOD scale | 0.0118 | 666.7 | 1.0000 | 1.00 | 0.00 |

Selective verification (τ sweep, 200 samples):

| τ | accept (global) | accept (CG) | FNR (CG) |
|---|---|---|---|
| 1e-3 | 0.000 | 0.000 | 0.00 |
| 5e-3 | 0.000 | 0.000 | 0.00 |
| 1e-2 | 0.000 | 0.000 | 0.00 |
| 5e-2 | 0.000 | **1.000** | 0.00 |
| 1e-1 | 0.000 | **1.000** | 0.00 |
| 5e-1 | 0.000 | **1.000** | 0.00 |

Cost scaling (median, n = interior nodes): 3.73 ms (n=15), 7.45 ms (n=31),
16.00 ms (n=63), 33.11 ms (n=127).

### Verdict
Headline numbers were ~within 5% of the fabricated values by coincidence, but they
were not evidence. The manuscript is now backed by `results/data/*.json`, committed.

---

## 2. Critical: the surrogate experiment certified the wrong bound

### Finding
`experiments/run_surrogate_coverage.py` calls
`certify_elliptic_1d(..., method='analytic', ...)`. That is the **global** bound.
Its output was `mean_bound = 17.86`, `mean_err = 0.0216` — an effectivity of
**825.7**, i.e. *as loose as the baseline*. The previous manuscript nevertheless
claimed "effectivity ≈ 1.0" for FNO predictions. The claim was unsupported.

### Remediation
New script `experiments/run_corrected_fno_cert.py` certifies real FNO predictions with
**both** bounds on ID and two OOD splits. The claim survives: CG effectivity is
1.0000 with coverage 1.00 and FNR 0.00 on all splits, while the global bound is
667–1470× loose.

---

## 3. Critical (conceptual): near-sharpness is close to a tautology

This cannot be fixed by more experiments and must be stated plainly.

The certificate targets the **algebraic** error $e = u_h - \hat u = A_h^{-1}r$.
Because CG solves $A_h z = r$ to high accuracy, $\|z\|_2 \to \|A_h^{-1}r\|_2 = \|e\|_2$
and $\|r - A_h z\|_2 \to 0$. Hence $B_{\mathrm{CG}} \to \|e\|_2$.

**Consequence:** the observed effectivity $\approx 1.0$ says that the certificate is
*optimal for the quantity it bounds*. It does **not** say that the certificate
detects total error. A surrogate with large discretization error, wrong coefficients,
or model bias can have $\hat u$ far from $u$ while its residual stays small whenever
the underlying operator is mis-specified — and that failure mode is exactly what
$u_h$-relative certification cannot see.

This is not a defect of the bound; it is the correct behaviour of an algebraic
residual estimator, and it is precisely the same statement classical algebraic
a posteriori estimators make. But presenting "effectivity ≈ 1.0" as a headline
*without* this caveat invites a reviewer rejection.

**Action taken.** The manuscript now states this in the abstract, in a dedicated
"Scope of the guarantee" subsection, and in Limitations. The contribution is
reframed from "we achieved near-sharpness" to "we obtain a *computable* certificate
that is optimal on its declared target, at O(n) cost, with a measured fallback
policy."

---

## 4. Major: broken iteration accounting

`results/data/run_corrected_cost.json` reported `mean_iter = 0.0`. Cause:
`scipy.sparse.linalg.cg` returns `info` (a status flag), not an iteration count;
the script stored `info` as the iteration count. Any cost/efficiency claim built on
that number is void.

**Remediation.** Iteration counting now uses a `callback` (see
`corrected_bound` call sites). Cost table in the manuscript reports wall-clock time
only, with no fabricated iteration counts.

---

## 5. Major: figures not derived from data

`experiments/gen_figures.py` contained literal arrays
(`med_eff_a = [312.3, 420.0, ...]`) written by hand. Figures are now generated by
reading `results/data/*.json` at runtime (`figures/fig1…fig5`). Regenerating data and
re-running the figure script is the only way to update a figure.

---

## 6. Major: bibliography was pre-2021 only

The prior reference list contained nothing from 2024–2026, despite the topic
(verification of neural PDE surrogates) having substantial recent literature.
Reviewers in this area will know this immediately.

**Remediation — added current references:**

- Ernst, Rekatsinas & Urban (2025/2026), *A posteriori certification of PDE
  approximations with particular application to neural networks*, arXiv:2502.20336.
  Closest existing work to ours; we now cite it explicitly and differentiate.
- Mukherjee, Fitzsimmons, Del Rey Fernández & Liu (2026), *Rigorous Error
  Certification for Neural PDE Solvers: From Empirical Residuals to Solution
  Guarantees*, arXiv:2603.19165. Argues residual control alone is insufficient
  without structural regularity — directly supports our §"Scope" caveat.
- Baheri & Laguna Peralta (2026), *Can We Formally Verify Neural PDE Surrogates? SMT
  Compilation of Small Fourier Neural Operators*, arXiv:2605.08938. Formal-verification
  route; contrasted with our a posteriori route.
- Zou et al. (2025), *Uncertainty quantification and stability of neural operators*,
  J. Comput. Phys. Empirical (statistical UQ, not certified).
- Song et al. (2026), *Structure-Aware Epistemic Uncertainty Quantification for
  Neural Operator PDE Surrogates*, arXiv:2603.11052 (statistical UQ).
- Residual-based error corrector operators for neural operators (2023), CMAME —
  a posteriori correction rather than certification; we now contrast correction
  vs. certification.

---

## 7. Moderate: claims without evidence

| Claim in old draft | Reality | Action |
|---|---|---|
| "2D elliptic experiments" | only 1D implemented in results | 2D moved to future work |
| "adversarial perturbations" | no such experiment in repo | removed from results, listed as future work |
| "counterexamples" | no such experiment in repo | removed from results, listed as future work |
| "Burgers / reaction–diffusion" | implemented, but no certificate | explicitly excluded from the theorem's scope |
| "coverage 1.0 / FNR 0.0" | true, but *guaranteed by the theorem*, not an empirical finding | reframed: empirics confirm implementation, not the proof |

The last row is the most important framing correction. Coverage = 1.0 is a theorem,
not a discovery. Reporting it as a result invites "this is trivially true." The
empirically meaningful quantity is **effectivity**, and secondarily **acceptance rate**.

---

## 8. Rigorousness gaps that remain open

These are genuine and are stated in the paper:

1. **No verified rounding.** $B_{\mathrm{CG}}$ is computed in IEEE double precision
   without outward rounding. A truly rigorous certificate would add a validated
   guard band, e.g. $B_{\mathrm{safe}} = B_{\mathrm{CG}}(1 + \varepsilon_{\mathrm{mach}} \cdot C)$
   with $C$ from standard dot-product error bounds.
2. **No 2D.** $\lambda_{\min}(-\Delta_h)$ closed forms and the assembly are 1D only.
3. **No nonlinear extension.** The theorem covers linear SPD only; Burgers and
   reaction–diffusion in the repo have no certificate.
4. **No discretization error.** Total error $\|u - \hat u\|$ is unbounded; only
   $\|u_h - \hat u\|$ is certified.
5. **No comparison to stronger algebraic baselines.** Chebyshev bounds, Lanczos-based
   lower bounds, and preconditioned-certificate variants are not benchmarked. A
   reviewer will ask whether $\|z\|+\|d\|/\alpha$ is the best computable form.
6. **Single reference implementation.** `u_h` from the same tridiagonal solver used
   to build $A_h$; a bug shared by both would not be detected. Coverage of 1.0 does
   not rule this out.

---

## 9. What the paper can honestly claim

Supported by data in this repository:

- A **computable** a posteriori certificate for the algebraic error of a learned
  elliptic surrogate, at O(n) cost per instance (3.7–33 ms for n = 15…127).
- The certificate is **optimal on its declared target**: measured median effectivity
  1.000000 across eight contrasts and three FNO test splits, versus 316–1470× for
  the standard global bound.
- A **measured fallback policy**: at τ = 5·10⁻² the certificate accepts 100% of
  instances while the global bound accepts 0%, with zero false negatives.
- **Implementation agreement**: no bound violation observed in any of ~2000
  certified instances.

Not supported (must not be claimed):

- Anything about total error $\|u - \hat u\|$.
- Certification for nonlinear PDEs.
- Verified (machine-checkable) arithmetic guarantees.
- Any result for 2D geometries, which were never run.
---

## Addendum — external review response (round 2)

An external review of `paper_journal.pdf` raised five concerns. Each is addressed
below with the evidence produced in response.

### R1. "Rigorous" needed stricter delimitation — ADDRESSED

The review was correct that a double-precision evaluation without outward rounding is
not a machine-checkable certificate. Rather than only softening the language, we
implemented a validated variant.

`src/certno/certificates/stability.py::certified_bound` applies three outward
roundings: `alpha` via `nextafter(alpha, 0)`, the residual norm inflated by
`1 + 2*gamma_n`, and the total by `1 + 4*gamma_n`, where
`gamma_n = n*eps/(1-n*eps)` is the standard Higham dot-product factor.

Measured inflation: `1.4e-14` at `n=31`, `5.6e-14` at `n=127` — four orders of
magnitude below the CG tolerance `1e-10`, so effectivity is unaffected.

Validated against an **80-digit reference** (`mpmath`, `mp.dps=80`) over 60 instances
(`experiments/audit_certified_bound.py`, `experiments/audit_numerics.py`):

| condition | trials | plain violations | validated violations | min rel. margin (plain) |
|---|---|---|---|---|
| n=31, c=1.5 | 15 | 0 | 0 | 2.6e-12 |
| n=31, c=10 | 15 | 0 | 0 | 7.3e-11 |
| n=127, c=1.5 | 15 | 0 | 0 | 9.6e-10 |
| n=127, c=10 | 15 | 0 | 0 | 1.0e-07 |

The smallest observed margin (`2.6e-12`) is ~100× the rounding scale, consistent with
the analysis rather than luck. The paper now distinguishes explicitly between the
*bound* (rigorous in exact arithmetic, Thm. 1) and the *validated* bound (rigorous for
the computed iterate under IEEE-754), and states that neither extends to end-to-end
interval verification. Limitation 4 was rewritten accordingly rather than deleted.

13 new tests in `tests/test_certified_bound.py`; suite is now **31 passing**.

### R2. Effectivity 1.0 is not the novel result — ADDRESSED (already, now reinforced)

The paper already carried this argument; the reviewer's endorsement prompted moving it
into the abstract, where a reader now encounters it before the numbers. The
contribution is framed as computability plus the measured acceptance policy, not as a
sharpness discovery.

### R3. Synthetic vs FNO acceptance conflation — ADDRESSED

Correct catch: the 100%-acceptance table was computed on synthetic perturbations and a
reader could conflate it with the FNO results. Two fixes:

1. The synthetic sweep now states explicitly, in the text, that its acceptance rates
   apply to the synthetic set and not to surrogates.
2. New experiment `experiments/run_fno_selective.py` measures acceptance on **real FNO
   predictions** per split:

| split | mean err | acc@τ=0.05 | acc@τ=0.5 | FNR |
|---|---|---|---|---|
| ID | 0.0210 | 0.950 | 1.000 | 0.00 |
| OOD contrast | 0.2954 | **0.000** | 1.000 | 0.00 |
| OOD scale | 0.0092 | 1.000 | 1.000 | 0.00 |

This produced a better result than the synthetic table: the OOD-contrast split is
**correctly refused** at tight tolerance, because its errors are genuinely large. The
certificate distinguishes surrogate quality per instance, which a split-level mean
cannot. This is now Table V and Sec. `fnosel`.

### R4. Baselines too narrow — PARTIALLY ADDRESSED

Still open. The paper compares only against the global residual bound. Chebyshev-type
iterative bounds, rigorous Lanczos lower bounds for `lambda_min`, and preconditioned
variants remain unbenchmarked. This is stated in Limitation 5 and is the single
largest gap for a strong journal.

### R5. Five pages insufficient — PARTIALLY ADDRESSED

Now 6 pages with the validated-certificate section and the FNO acceptance study.
Still short of a Q1 submission: no 2D results, no deeper related work, no confidence
intervals on medians (now flagged in Limitation 8 rather than glossed over).

### Reproducibility verification (reviewer's recommended next step)

Re-ran `run_paper_tables.py` from the committed state and compared against the
published JSON. **All scientific quantities reproduce bit-identically** (effectivity
medians, coverage, FNR, error means) across all 8 contrasts and 4 problem sizes. Only
wall-clock timings differ, which is expected and is now excluded from the
reproducibility claim. Determinism rests on fixed seeds: `20240` (contrast study),
`4242` (tolerance sweep), `31337` (numerics audit).

### Author

Changed to **Emmett Vitruvious** as requested.
