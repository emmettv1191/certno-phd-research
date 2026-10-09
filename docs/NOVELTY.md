# certno — literature and novelty matrix

This document distills the closest existing work for Problem A (a posteriori error certification of learned PDE solvers / neural operators) and clarifies the novelty of the present implementation.

## Closest work

| Work | Year | Key idea | Relation to this project | Gap addressed here |
|---|---|---|---|---|
| De Ryck & Mishra. "Numerical analysis of physics-informed neural networks..." (arXiv:2402.10926) | 2024 | Separates approximation, training, generalization, stability errors for PINNs/operator learning | Provides the conceptual decomposition (certificates target the residual/stability part) | Focuses on *computable, per-sample* error bounds for elliptic operators with explicit validity assumptions and selective verification (not asymptotic/error estimates of training). |
| Out-of-distribution risk bounds for neural operators (JCP 2024) | 2024 | Provides theoretical OOD risk bounds for NOs in some settings | Establishes that distribution shift is tractable under assumptions | Moves from population risk to a per-instance a posteriori bound using the PDE residual and a computable stability constant, plus an actionable fallback policy. |
| Serrano et al. "Test-time Generalization for Physics through Neural Operator Splitting" (ICML 2026) | 2026 | Compositional operator splitting improves test-time generalization | Addresses test-time generalization of the model itself | Orthogonal: our focus is *trustworthiness/certification* (error bound + when to abstain) rather than architecture generalization. |
| Ainsworth & Oden (a posteriori FE error estimation) | Classical | Residual-based and goal-oriented a posteriori estimates for FEM | The mathematical foundation is identical (Céa-like / stability + residual) | Transferred to learned surrogates: the certificate is applied to any $\hat{u}$ (possibly not from FEM) with a cheap, closed-form stability factor and an adaptive verify-or-abstain policy measured by coverage/FNR. |
| Bayesian NO / uncertainty quantification (e.g. variational/ensemble) | Recent | Provide predictive uncertainty (epistemic/aleatoric) | Often correlated with error but lacks *formal* error bounds under stated PDE assumptions | This is a *deterministic a posteriori error bound* for the discrete SPD problem (when hypotheses hold), not a probabilistic confidence interval. |

## What is already well-known

- For linear, well-posed (coercive) problems, $\|u-\hat{u}\| \le \|A^{-1}\| \|r\|$ is a standard residual inequality.
- Residual indicators are used extensively in adaptive FEM and in verification.
- Neural operators can be fast surrogates; reliability/generalization under shift is a frontier.

## Novelty (this implementation)

1. **Explicit validity domain**: `assumptions_ok` tracks whether the assumed $a_{\min}$ is satisfied by the input coefficient; the bound is only claimed valid in that case. This makes "when the certificate fails" a first-class object.
2. **Cheap, computable stability factor**: analytic lower bound on $\lambda_{\min}(A_h)$ via the unit-Laplacian eigenvalue (no per-sample eigendecomposition needed). `discrete-exact` is also provided for benchmarking.
3. **Selective verification policy**: `adaptive_verify` abstains and calls a reference solver only when the certificate is invalid or the bound exceeds tolerance. The cost is measured directly (`reference_calls`) and traded against coverage.
4. **Reliability metrics aligned with engineering decisions**: `coverage(bounds, errors)`, `sharpness`, and `false_negative_rate` (bound <= τ but error > τ) — the latter is the critical safety metric.
5. **Reproducible computational study design**: MMS verification, controlled distribution shifts, multiple seeds, held-out ranges, and documented failure cases. The experiment runners output JSON for post-hoc analysis.
6. **Identifiability analysis (Problem D)**: makes the scaling symmetry $ (a,f)\mapsto(ca,cf)$ explicit and shows it is observationally unresolvable from solution/residual fields alone — a necessary caveat for any residual-based diagnostic.
7. **Minimal, testable scaffold**: the core certificate is ~200 lines of logic with unit tests covering MMS, SPD properties, residual-zero, bound monotonicity, and hypothesis violations.

## Distinction from uncertainty quantification

Ensemble variance or Bayesian posterior variance may correlate with error but does not provide a guaranteed upper bound under PDE stability assumptions. The present bound is *guaranteed for the discrete SPD system* whenever `assumptions_ok` is true. UQ and this certificate are complementary: UQ can guide data acquisition (as in Problem B), while the certificate can gate deployment (accept/reject/verify).

## Potential PhD-level contribution

The original contribution is not re-deriving the residual inequality, but developing a **practical, per-instance reliability framework for learned surrogates** that (a) states its domain of validity explicitly, (b) quantifies looseness via analytic vs discrete-exact factors, (c) implements a cost-aware abstention policy, and (d) is validated against controlled OOD regimes with safety-relevant metrics. Extending to tighter stability constants for heterogeneous media, to energy/H^1 norms, and proving coverage under structured shift would strengthen the theoretical novelty.
