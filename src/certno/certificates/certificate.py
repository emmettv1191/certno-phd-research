"""The a posteriori elliptic certificate and adaptive verification policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from ..solvers.elliptic import (
    assemble_elliptic_1d,
    assemble_elliptic_2d,
    solve_elliptic_1d,
    solve_elliptic_2d,
)
from .residual import residual_1d, residual_2d
from .stability import (
    analytic_coercivity_1d,
    analytic_coercivity_2d,
    exact_lambda_min,
)


@dataclass
class EllipticCertificate:
    """A computable reliability certificate for one surrogate prediction.

    Attributes
    ----------
    bound:
        Upper bound on the discrete interior error ``||u_h - u_hat||_2``.
    residual_norm:
        ``||f - A_h u_hat||_2``.
    stability_factor:
        ``1 / coercivity`` used to convert residual to error.
    coercivity:
        Lower bound on ``lambda_min(A_h)`` (analytic) or its exact value.
    method:
        ``"analytic"`` or ``"discrete-exact"``.
    assumptions_ok:
        False when the test coefficient violates the assumed lower bound, i.e.
        the certificate's hypotheses are not met and the bound is not valid.
    a_min_assumed, a_min_actual:
        Assumed and realized coercivity floors.
    verified:
        Whether an expensive reference solve was executed.
    verified_error:
        Actual error after verification, when available.
    """

    bound: float
    residual_norm: float
    stability_factor: float
    coercivity: float
    method: str
    assumptions_ok: bool
    a_min_assumed: float
    a_min_actual: float
    verified: bool = False
    verified_error: float | None = None
    notes: str = ""

    @property
    def valid(self) -> bool:
        return bool(self.assumptions_ok and np.isfinite(self.bound))

    def as_dict(self) -> dict:
        return {
            "bound": self.bound,
            "residual_norm": self.residual_norm,
            "stability_factor": self.stability_factor,
            "coercivity": self.coercivity,
            "method": self.method,
            "assumptions_ok": self.assumptions_ok,
            "a_min_assumed": self.a_min_assumed,
            "a_min_actual": self.a_min_actual,
            "verified": self.verified,
            "verified_error": self.verified_error,
            "notes": self.notes,
        }


def _stability(
    method: str,
    *,
    a_min_used: float,
    n_interior: int,
    A: sparse.spmatrix,
    dim: int,
) -> tuple[float, float]:
    if method == "analytic":
        coercivity = (
            analytic_coercivity_1d(a_min_used, n_interior)
            if dim == 1
            else analytic_coercivity_2d(a_min_used, n_interior)
        )
    elif method == "discrete-exact":
        coercivity = exact_lambda_min(A)
    else:
        raise ValueError(f"unknown certificate method: {method!r}")
    return coercivity, 1.0 / coercivity


def certify_elliptic_1d(
    a_nodes: np.ndarray,
    f_interior: np.ndarray,
    u_hat_interior: np.ndarray,
    *,
    method: str = "analytic",
    a_min_assumed: float | None = None,
    n_interior: int | None = None,
    averaging: str = "arithmetic",
) -> EllipticCertificate:
    """Certify a candidate interior solution of the 1D elliptic problem."""
    a_nodes = np.asarray(a_nodes, dtype=float).ravel()
    if n_interior is None:
        n_interior = np.asarray(f_interior).size
    A = assemble_elliptic_1d(a_nodes, n_interior, averaging=averaging)
    r = residual_1d(A, f_interior, u_hat_interior)
    rn = float(np.linalg.norm(r))
    a_min_actual = float(a_nodes.min())
    a_min_used = a_min_actual if a_min_assumed is None else float(a_min_assumed)
    coercivity, factor = _stability(
        method, a_min_used=a_min_used, n_interior=n_interior, A=A, dim=1
    )
    assumptions_ok = a_min_assumed is None or a_min_actual >= a_min_assumed - 1e-12
    return EllipticCertificate(
        bound=factor * rn,
        residual_norm=rn,
        stability_factor=factor,
        coercivity=coercivity,
        method=method,
        assumptions_ok=assumptions_ok,
        a_min_assumed=a_min_used,
        a_min_actual=a_min_actual,
        notes="" if assumptions_ok else "coefficient violates assumed coercivity floor",
    )


def certify_elliptic_2d(
    a_nodes: np.ndarray,
    F_interior: np.ndarray,
    U_hat_interior: np.ndarray,
    *,
    method: str = "analytic",
    a_min_assumed: float | None = None,
    n_interior: int | None = None,
    averaging: str = "arithmetic",
) -> EllipticCertificate:
    """Certify a candidate interior solution of the 2D elliptic problem."""
    a_nodes = np.asarray(a_nodes, dtype=float)
    if n_interior is None:
        n_interior = np.asarray(F_interior).shape[0]
    A = assemble_elliptic_2d(a_nodes, n_interior, averaging=averaging)
    r = residual_2d(A, F_interior, U_hat_interior)
    rn = float(np.linalg.norm(r))
    a_min_actual = float(a_nodes.min())
    a_min_used = a_min_actual if a_min_assumed is None else float(a_min_assumed)
    coercivity, factor = _stability(
        method, a_min_used=a_min_used, n_interior=n_interior, A=A, dim=2
    )
    assumptions_ok = a_min_assumed is None or a_min_actual >= a_min_assumed - 1e-12
    return EllipticCertificate(
        bound=factor * rn,
        residual_norm=rn,
        stability_factor=factor,
        coercivity=coercivity,
        method=method,
        assumptions_ok=assumptions_ok,
        a_min_assumed=a_min_used,
        a_min_actual=a_min_actual,
        notes="" if assumptions_ok else "coefficient violates assumed coercivity floor",
    )


@dataclass
class VerificationResult:
    certificate: EllipticCertificate
    accepted: bool
    used_reference_solver: bool
    reference_calls: int
    notes: str = ""


def adaptive_verify(
    certificate: EllipticCertificate,
    *,
    tolerance: float,
    solve_reference,
    u_hat: np.ndarray,
) -> VerificationResult:
    """Accept the surrogate if the certificate is valid and below ``tolerance``.

    Otherwise fall back to the reference solver (``solve_reference`` returns the
    exact discrete solution as a numpy array).  This is the selective-verification
    policy evaluated in ``experiments/run_certificate_study.py``.
    """
    if certificate.valid and certificate.bound <= tolerance:
        return VerificationResult(
            certificate=certificate,
            accepted=True,
            used_reference_solver=False,
            reference_calls=0,
            notes="accepted by certificate",
        )

    exact = np.asarray(solve_reference(), dtype=float).ravel()
    u_hat = np.asarray(u_hat, dtype=float).ravel()
    certificate.verified = True
    certificate.verified_error = float(np.linalg.norm(exact - u_hat))
    reason = "hypotheses violated" if not certificate.assumptions_ok else "bound above tolerance"
    return VerificationResult(
        certificate=certificate,
        accepted=False,
        used_reference_solver=True,
        reference_calls=1,
        notes=f"reference solve ({reason})",
    )


def certify_with_verification(
    *,
    problem: str,
    a_nodes: np.ndarray,
    rhs: np.ndarray,
    u_hat: np.ndarray,
    tolerance: float,
    method: str = "analytic",
    a_min_assumed: float | None = None,
    n_interior: int | None = None,
    averaging: str = "arithmetic",
) -> tuple[EllipticCertificate, VerificationResult]:
    """Convenience wrapper: build a certificate then apply adaptive verification.

    ``problem`` is ``"elliptic-1d"`` or ``"elliptic-2d"``.
    """
    if problem == "elliptic-1d":
        cert = certify_elliptic_1d(
            a_nodes,
            rhs,
            u_hat,
            method=method,
            a_min_assumed=a_min_assumed,
            n_interior=n_interior,
            averaging=averaging,
        )

        def solve_reference():
            return solve_elliptic_1d(
                a_nodes, rhs, n_interior, averaging=averaging
            ).u_interior

    elif problem == "elliptic-2d":
        cert = certify_elliptic_2d(
            a_nodes,
            rhs,
            u_hat,
            method=method,
            a_min_assumed=a_min_assumed,
            n_interior=n_interior,
            averaging=averaging,
        )

        def solve_reference():
            return solve_elliptic_2d(
                a_nodes, rhs, n_interior, averaging=averaging
            ).U_interior

    else:
        raise ValueError(f"unknown problem: {problem!r}")

    verified = adaptive_verify(
        cert, tolerance=tolerance, solve_reference=solve_reference, u_hat=u_hat
    )
    return cert, verified
