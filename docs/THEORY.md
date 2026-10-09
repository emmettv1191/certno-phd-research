# Theory: a posteriori error certificates for elliptic PDEs

## Setup

We consider the discrete symmetric positive definite (SPD) system
$$
A_h u_h = f, \qquad A_h \succ 0,
$$
arising from a standard finite-difference discretisation of
$$
- \nabla\cdot(a(x)\nabla u) = f, \quad x \in \Omega=(0,1)^d, \quad u|_{\partial\Omega}=0,
$$
with $a(x) \ge a_{\min} > 0$.

For any candidate $\hat{u} \in \mathbb{R}^N$, define the discrete residual
$$
r = f - A_h \hat{u}.
$$
The true discrete error is $e = u_h - \hat{u} = A_h^{-1} r$.

## L2 error bound (residual + coercivity)

Because $A_h$ is SPD,
$$
\|e\|_2 = \|A_h^{-1} r\|_2 \le \|A_h^{-1}\|_2 \|r\|_2 = \frac{\|r\|_2}{\lambda_{\min}(A_h)},
$$
where $\lambda_{\min}(A_h) > 0$ is the smallest eigenvalue.

We have the coercivity bound
$$
\lambda_{\min}(A_h) \ge a_{\min}\, \lambda_{\min}(-\Delta_h).
$$
For a uniform grid with $h = 1/(n+1)$ and Dirichlet boundaries:

- 1D: $\lambda_{\min}(-\Delta_h) = \frac{4}{h^2}\sin^2\!\left(\frac{\pi h}{2}\right)$.
- 2D: $\lambda_{\min}(-\Delta_h) = \frac{8}{h^2}\sin^2\!\left(\frac{\pi h}{2}\right)$.

Hence the computable (analytic) certificate is
$$
\mathcal{C}(\hat{u}) = \frac{\|f - A_h \hat{u}\|_2}{a_{\min}\, \lambda_{\min}(-\Delta_h)} \ge \|u_h - \hat{u}\|_2.
$$

## Validity

The bound is valid whenever:
1. $A_h$ is the SPD operator corresponding to $a(x) \ge a_{\min}$;
2. the assumed $a_{\min}^{(\text{ass})} \le \min_x a(x)$ (implemented as $\min_i a_i \ge a_{\min}^{(\text{ass})}$ within tolerance);
3. homogeneous Dirichlet boundary conditions hold.

If (2) is violated, the certificate's hypotheses are not satisfied and we do not claim validity. The code sets `assumptions_ok = false` in that case and triggers selective verification.

## Energy norm (optional reference)

The discrete energy (A-norm) of the error satisfies
$$
\|e\|_{A_h}^2 = e^T A_h e = r^T A_h^{-1} r \implies \|e\|_{A_h} = \sqrt{r^T A_h^{-1} r}.
$$
Computing it requires solving $A_h z = r$. In this codebase, `energy_norm_error_1d` does exactly that (an "oracle" certificate) to benchmark the cheap L2 bound.

## Selective (adaptive) verification policy

Given tolerance $\tau$:
- If `assumptions_ok` and $\mathcal{C}(\hat{u}) \le \tau$: accept $\hat{u}$ (no reference solve). `reference_calls = 0`.
- Otherwise: run the high-accuracy reference solver to obtain $u_h$ (cost 1 solve). Compute the actual error and set `verified_error = \|u_h - \hat{u}\|_2$. `reference_calls = 1`.

The experiment reports coverage, sharpness, and false-negative rate (cases where actual error $> \tau$ but the bound accepted the prediction).

## Limitations

- Bound depends on the global coercivity constant; it is safe but can be loose for highly heterogeneous $a$ with large contrast.
- The discrete-exact method (`method="discrete-exact"`) uses $\lambda_{\min}(A_h)$ directly (more accurate factor, but requires a sparse eigensolve).
- Results are for the discrete system; to relate to the continuous solution one must also account for discretisation error of the reference solver.
- For nonlinear problems (e.g. Burgers) the linear SPD certificate does not apply directly; those modules are separate exploratory studies.

## Identifiability (Problem D)

For the linear elliptic problem, $(a,f) \mapsto u = A(a)^{-1} f$. The scaling
$$
(a,f) \equiv (c a, c f),\; c>0,
$$
produces the same solution $u$. Consequently, solution-based or residual-based diagnostics cannot distinguish "coefficient upscaled by $c$" from "forcing upscaled by $c$" without extra priors or measurements. This is an inherent identifiability limit.
