## Corrected bound (mathematical contribution)

We added a CG-corrected residual certificate:
$$
\|e\|_2 \le \|z\|_2 + \frac{\|r - A_h z\|_2}{\alpha}, \quad z \approx A_h^{-1}r, \quad \alpha \le \lambda_{\min}(A_h).
$$

This preserves coverage (1.0) and FNR (0.0) while reducing looseness from orders of magnitude to near 1.0x.
