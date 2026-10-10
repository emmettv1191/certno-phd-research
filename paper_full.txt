# A CG-Corrected Residual Certificate for Learned Elliptic Solvers

certno  
Research Program on Error-Certified Neural Operators  
https://github.com/emmettv1191/certno-phd-research

## Abstract
We propose a CG-corrected a posteriori residual certificate for the discrete elliptic system $A_h u_h = f$. Starting from $\|u_h-\hat{u}\|_2 \le \|r\|_2/\lambda_{\min}(A_h)$, we compute $z \approx A_h^{-1}r$ via CG and form $B_{CG} = \|z\|_2 + \|r-A_h z\|_2/\alpha$, $\alpha \le \lambda_{\min}(A_h)$. Across contrasts, ID/OOD, and real FNO predictions, $B_{CG}$ is near-sharp (effectivity ~1.0) while preserving coverage 1.0 and FNR 0.0.

## Theory
$B_{CG} = \|z\|_2 + \frac{\|r-A_h z\|_2}{\alpha} \ge \|u_h-\hat{u}\|_2$.

## Results
| Contrast | Coverage | FNR | Med eff (analytic) | Med eff (CG) |
|---------|---------|-----|--------------------|--------------|
| 1.5     | 1.0     | 0.0 | ~312              | ~1.00        |
| 3.0     | 1.0     | 0.0 | ~500              | ~1.00        |
| 5.0     | 1.0     | 0.0 | ~768              | ~1.00        |
| 10.0    | 1.0     | 0.0 | ~1361             | ~1.00        |

## Reproducibility
See BENCHMARK_FREEZE.md, THEOREM_AUDIT.md, RESULTS.md. Code: https://github.com/emmettv1191/certno-phd-research
