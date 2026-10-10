# Contribution

The main contribution is a CG-corrected a posteriori certificate for the discrete SPD elliptic system. 
Given $A_h u_h=f$, $r=f-A_h\hat{u}$, $z\approx A_h^{-1}r$, $\alpha\le\lambda_{\min}(A_h)$:
$$
\|u_h-\hat{u}\|_2 \le \|z\|_2 + \frac{\|r-A_h z\|_2}{\alpha}.
$$

This preserves rigorousness under stated assumptions and reduces effectivity from $O(10^2-10^3)$ to $\approx 1.0\times$ while maintaining coverage 1.0/FNR 0.0 across contrasts and real FNO errors.
