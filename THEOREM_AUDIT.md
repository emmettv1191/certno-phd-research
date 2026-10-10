# Theorem audit: CG-corrected a posteriori bound

We consider SPD $A_h \succ 0$, $A_h u_h = f$, residual $r = f - A_h \hat{u}$, error $e = u_h - \hat{u} = A_h^{-1}r$.

Let $z$ be any vector (CG approximation) and define $d = r - A_h z = A_h (e-z)$ so $e = z + A_h^{-1}d$.
Then
$$
\|e\|_2 \le \|z\|_2 + \|A_h^{-1}d\|_2 \le \|z\|_2 + \frac{\|d\|_2}{\lambda_{\min}(A_h)}.
$$
If $\alpha > 0$ is a rigorous lower bound on $\lambda_{\min}(A_h)$ (here $\alpha = a_{\min}\lambda_{\min}(-\Delta_h)$ under stated assumptions), then
$$
B_{CG} = \|z\|_2 + \frac{\|r - A_h z\|_2}{\alpha} \ge \|u_h - \hat{u}\|_2.
$$

Assumptions:
- Discrete SPD operator from coercive elliptic problem.
- $\alpha \le \lambda_{\min}(A_h)$ is valid (coefficient bounds).
- Norms computed in floating point; we use tight stopping (rtol) so algebraic residual is controlled.
- Finite precision: CG is run in standard double precision; the bound remains rigorous in exact arithmetic given $\alpha$.
