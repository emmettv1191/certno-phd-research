# certno — A PhD-level research program on error-certified neural operators

## Problem statement (central question)

This repository implements Problem A from the research program: **a posteriori reliability certificates for learned PDE solvers**, with selective verification (adaptive fallback to a reference solver) and supporting investigations for goal-oriented active learning (Problem B) and failure-mode diagnostics (Problem D).

The central hypothesis is:
> Can we produce a computable diagnostic `C_θ(μ)` such that
> `||G_μ - \hat{G}_θ||_2 <= C_θ(μ)`
> under explicit assumptions, with useful empirical coverage outside the training distribution and affordable selective verification?

For the discrete elliptic problem `-∇·(a∇u)=f`, the analytic certificate is
`||u_h - \hat{u}||_2 <= ||r||_2 / (a_min * λ_min(-Δ_h))` with `r = f - A_h \hat{u}`.
The code uses a closed-form `λ_min(-Δ_h)` and face-averaged, SPD 5-point (1D/2D) operators; the certificate is valid when `a_min_actual >= a_min_assumed`.

## Core modules

- `certno.solvers` — Verified reference solvers: 1D/2D variable-coefficient elliptic (SPD), 1D periodic spectral ADR/Burgers, manufactured solutions (MMS), KL random fields for coefficients.
- `certno.certificates` — Residuals, stability (analytic + exact), `EllipticCertificate`, `certify_*`, and `adaptive_verify` (fallback only when bound above tolerance or hypotheses violated).
- `certno.operators` — FNO1d and DeepONet1d, small dataset generators, and training helpers (torch-based, lazy imports).
- `certno.active` — Bayesian linear surrogate on RFF, `goal_oriented` acquisition (expected variance reduction over a QoI test set).
- `certno.diagnostics` — Interpretable residual features + logistic classifier, plus identifiability note (the scaling `c a, c f` is observationally equivalent).
- `certno.metrics` — Coverage, sharpness, false-negative rate, confidence intervals.

## Getting started

### Environment (already done here)
Python 3.14 venv with numpy, scipy, torch (CPU), matplotlib, pytest. Also included: `src/certno`.

### Quick check
```bash
cd "/home/okta-cory/Documentos/Default Project"
source .venv/bin/activate  # if needed
python -m pytest -q
python experiments/run_demo.py
python experiments/run_benchmark1_elliptic.py
python experiments/run_benchmark2_burgers.py
python experiments/run_benchmark3_reaction_diffusion.py
python experiments/run_certificate_study.py
python experiments/run_active_learning.py
python experiments/run_failure_diagnostics.py
```

Or use Make:
```bash
make test
make demo
make bench1 bench2 bench3 cert al diag
```

## Tests
- `tests/test_elliptic.py` — MMS convergence, positive-definiteness, coercivity bounds
- `tests/test_certificates.py` — residual-zero case, bound sanity, analytic vs discrete-exact relation, hypothesis violation
- `tests/test_spectral.py` — spectral solver smooth initial condition
- `tests/test_metrics.py` — coverage, FNR, sharpness, relative errors
- `tests/test_diagnostics.py` — residual features shape, classifier accuracy sanity
- `tests/test_active.py` — RFF/Bayesian surrogate shapes and acquisition policies

## Research outputs (what this enables)

### Problem A (primary)
- Closed-form, computable stability factor for 1D/2D elliptic; certificates can be evaluated on every candidate without full eigensolves if using the analytic bound.
- Selective verification: call the expensive solver only when `bound > tolerance` or assumptions fail. This is measured via `reference_calls`.
- Metrics: `coverage`, `sharpness`, `false_negative_rate` (key failure mode: accepting prediction with error > tolerance).

### Problem B
- Goal-oriented acquisition reduces uncertainty in a chosen QoI rather than global field error; includes cost-aware variant and `uncertainty`/`uniform` baselines.

### Problem D
- Residual-field signature separates certain failure classes (e.g. forcing mismatch) in a synthetic setting.
- Explicit identifiability: scaling symmetry means coefficient/forcing scale cannot be distinguished from solution/residual alone.

## Known limitations (honest)
- Certificate is for the *discrete* operator `A_h` (not the continuous PDE in all regimes). Assumptions: `a > 0`, coercivity holds, homogeneous Dirichlet.
- Analytic stability factor is a (safe, often loose) lower bound on `λ_min(A_h)`; "discrete-exact" uses a Lanczos eigensolve (one solve-like cost).
- Long-horizon Burgers and 2D cases are prototyped; the current FNO/DeepONet are small 1D models for demonstration (not state-of-the-art scale).
- Failure diagnostics: synthetic labels; true identifiability failures are only partially explored.
- Adaptive verification uses the actual discrete reference solution to compute `verified_error` in this implementation; that is useful for reporting, not free.

## Reproducibility
- All random generation uses `np.random.default_rng(seed)`.
- Configs in `configs/`.
- Experiment runners in `experiments/` with JSON output.
- Tests cover the mathematical core (MMS, SPD structure, residual properties).
- Theory in `docs/THEORY.md`.

## Next steps (if extending to a thesis)
1. Derive tighter stability factors for specific coefficient classes; quantify looseness.
2. Extend certificate to energy/H^1 norm and to other boundary conditions.
3. Build FNO/DeepONet trained on larger distributions, then report certificate coverage under controlled OOD splits.
4. Implement the goal-oriented policy on the learned operator (e.g. select training samples to minimize QoI error) and compare to baselines.
5. Design genuinely indistinguishable diagnostic cases and measure when separation is impossible.

## Citation / usage
This is a computational research scaffold. If you use it, refer to the central question above. The code is MIT-licensed.
