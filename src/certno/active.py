"""Goal-oriented active learning for surrogate models (Problem B).

Rather than sampling where the *field* prediction is uncertain, we sample where
the uncertainty in an *engineering quantity of interest* (QoI) is largest.
This module implements a self-contained Bayesian linear surrogate on random
Fourier features so that predictive variances -- and hence expected variance
reductions -- are available in closed form without gradient-based training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RFFFeatures:
    """Random Fourier features for a stationary kernel."""

    d_in: int
    n_features: int = 256
    length_scale: float = 1.0
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        rng = self.rng or np.random.default_rng(0)
        self.W = rng.standard_normal((self.d_in, self.n_features)) / self.length_scale
        self.b = rng.uniform(0.0, 2.0 * np.pi, size=self.n_features)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.sqrt(2.0 / self.n_features) * np.cos(X @ self.W + self.b)


class BayesianLinearSurrogate:
    """Bayesian linear regression on a feature map, with predictive variance."""

    def __init__(self, features: RFFFeatures, noise: float = 1e-3, prior: float = 1.0):
        self.features = features
        self.noise = noise
        self.prior = prior
        self.precision = np.eye(features.n_features) / prior
        self.theta = np.zeros(features.n_features)
        self._chol = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianLinearSurrogate":
        Z = self.features(X)
        y = np.asarray(y, dtype=float).ravel()
        A = Z.T @ Z / self.noise + self.precision
        b = Z.T @ y / self.noise
        self._chol = np.linalg.cholesky(A)
        self.theta = np.linalg.solve(A, b)
        self._A = A
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return posterior predictive mean and standard deviation."""
        Z = self.features(X)
        mean = Z @ self.theta
        A_inv_Zt = np.linalg.solve(self._A, Z.T)
        var = np.einsum("ij,ji->i", Z, A_inv_Zt) + self.noise
        return mean, np.sqrt(np.clip(var, 0.0, None))


def _expected_variance_reduction(
    surrogate: BayesianLinearSurrogate,
    X_test: np.ndarray,
    X_candidate: np.ndarray,
) -> np.ndarray:
    """Sum over test points of the posterior-variance drop from observing a candidate.

    For candidate :math:`x`, the drop in posterior variance at a test point
    :math:`x_*` is ``cov(x_*, x)^2 / (var(x) + noise)``; summing over the test
    points makes the acquisition *goal-oriented* -- it values information where
    decisions are made.
    """
    Zt = surrogate.features(X_test)
    Zc = surrogate.features(X_candidate)
    A_inv = np.linalg.inv(surrogate._A)
    # posterior covariance between test and candidate: Zt A^-1 Zc^T
    cross = Zt @ A_inv @ Zc.T  # (n_test, n_cand)
    Zc_Ainv_Zc = np.einsum("ij,jk,ik->i", Zc, A_inv, Zc)  # (n_cand,)
    denom = Zc_Ainv_Zc + surrogate.noise
    reduction = np.einsum("ij,ij->j", cross, cross) / denom
    return reduction


def select_batch(
    surrogate: BayesianLinearSurrogate,
    X_pool: np.ndarray,
    *,
    n_select: int,
    policy: str,
    X_test: np.ndarray | None = None,
    costs: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return indices of the selected pool points under the given policy.

    Policies
    --------
    ``uniform``       : uniformly at random from the pool.
    ``uncertainty``   : largest predictive standard deviation.
    ``goal_oriented`` : largest expected reduction of QoI variance on ``X_test``.
    ``cost_aware``    : goal-oriented value divided by simulation cost.
    """
    rng = rng or np.random.default_rng(0)
    n_pool = X_pool.shape[0]
    if n_select >= n_pool:
        return np.arange(n_pool)
    if policy == "uniform":
        return rng.choice(n_pool, size=n_select, replace=False)
    if policy == "uncertainty":
        _, std = surrogate.predict(X_pool)
        return np.argsort(std)[::-1][:n_select]
    if policy in ("goal_oriented", "cost_aware"):
        if X_test is None:
            raise ValueError("goal-oriented policy requires X_test")
        value = _expected_variance_reduction(surrogate, X_test, X_pool)
        if policy == "cost_aware":
            if costs is None:
                raise ValueError("cost_aware policy requires costs")
            value = value / np.asarray(costs, dtype=float)
        return np.argsort(value)[::-1][:n_select]
    raise ValueError(f"unknown policy: {policy!r}")
