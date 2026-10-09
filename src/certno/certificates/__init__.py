"""A posteriori reliability certificates for learned elliptic PDE surrogates.

The central object is the *computable* bound

.. math::

    \\|u_h - \\hat u\\|_2 \\;\\le\\; \\frac{\\|r\\|_2}{\\lambda_{\\min}(A_h)},
    \\qquad r = f - A_h \\hat u,

valid for any candidate :math:`\\hat u` because :math:`A_h` is symmetric
positive definite.  Two ways of supplying the stability factor
:math:`1/\\lambda_{\\min}` are supported: an exact discrete value from a sparse
eigendecomposition, and a storage-free analytic lower bound derived from
coercivity.  See ``docs/THEORY.md`` for statements and proofs.
"""

from .residual import residual_1d, residual_2d
from .stability import (
    analytic_coercivity_1d,
    analytic_coercivity_2d,
    exact_lambda_min,
    energy_norm_error_1d,
)
from .certificate import (
    EllipticCertificate,
    certify_elliptic_1d,
    certify_elliptic_2d,
    certify_with_verification,
    adaptive_verify,
)

__all__ = [
    "residual_1d",
    "residual_2d",
    "analytic_coercivity_1d",
    "analytic_coercivity_2d",
    "exact_lambda_min",
    "energy_norm_error_1d",
    "EllipticCertificate",
    "certify_elliptic_1d",
    "certify_elliptic_2d",
    "certify_with_verification",
    "adaptive_verify",
]
