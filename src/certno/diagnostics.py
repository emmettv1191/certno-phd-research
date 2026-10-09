"""Failure-mode diagnostics for learned PDE models (Problem D).

When a surrogate fails on new data, the *cause* matters: an unseen input regime
can be fixed by more training data, whereas an incorrect governing equation
requires changing the model.  Different mechanisms, however, can produce very
similar residual fields.  This module provides:

* ``residual_features`` -- a small, physically interpretable signature of a
  residual field (magnitude, spectral content, correlation length, boundary
  weight);
* a from-scratch multinomial logistic-regression classifier;
* ``identifiability_note`` and a constructive non-identifiability example.

The associated experiment (``experiments/run_failure_diagnostics.py``) compares
residual-norm-only, spectral-only, and combined diagnostics, and reports where
discrimination is fundamentally impossible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def residual_features(r: np.ndarray, h: float) -> np.ndarray:
    """Return an interpretable feature vector for a 1D residual field.

    Features: relative L2 magnitude, spectral centroid, high-frequency energy
    fraction, decay length of the autocorrelation, and the fraction of residual
    energy located within one grid spacing of either boundary.
    """
    r = np.asarray(r, dtype=float).ravel()
    n = r.size
    norm = float(np.linalg.norm(r) + 1e-15)
    level = float(np.linalg.norm(r) / np.sqrt(n))

    r_hat = np.fft.rfft(r)
    power = np.abs(r_hat) ** 2
    freqs = np.fft.rfftfreq(n, d=h)
    total = float(power.sum() + 1e-15)
    centroid = float((freqs * power).sum() / total)
    half = freqs.size // 2
    hf_frac = float(power[half:].sum() / total)

    # Autocorrelation decay length (integrated autocorrelation time).
    rc = r - r.mean()
    ac = np.correlate(rc, rc, mode="full")[n - 1 :]
    ac = ac / (ac[0] + 1e-15)
    below = np.where(ac < 1.0 / np.e)[0]
    decay = float(below[0] * h) if below.size else float(n * h)

    edge = max(1, int(round(1.0 / max(h, 1e-15))) if h > 0 else 1)
    edge = min(edge, n)
    edge_energy = float((r[:edge] ** 2).sum() + (r[-edge:] ** 2).sum())
    edge_frac = float(edge_energy / (norm**2))

    return np.array([level, centroid, hf_frac, decay, edge_frac, norm])


@dataclass
class LogisticClassifier:
    """Multinomial logistic regression trained by gradient descent."""

    n_features: int
    n_classes: int
    l2: float = 1e-3

    def __post_init__(self) -> None:
        self.W = np.zeros((self.n_features, self.n_classes))
        self.b = np.zeros(self.n_classes)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def fit(
        self, X: np.ndarray, y: np.ndarray, *, epochs: int = 800, lr: float = 0.5
    ) -> "LogisticClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        n = X.shape[0]
        Y = np.eye(self.n_classes)[y]
        for _ in range(epochs):
            P = self._softmax(X @ self.W + self.b)
            G = (P - Y) / n
            grad_W = X.T @ G + self.l2 * self.W
            grad_b = G.sum(axis=0)
            self.W -= lr * grad_W
            self.b -= lr * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self._softmax(X @ self.W + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))


def identifiability_note() -> str:
    """Return a statement of the constructive non-identifiability result."""
    return (
        "For the linear elliptic problem A(a) u = f, the pair (a, f) and the "
        "pair (c*a, c*f) have the same solution u for any c > 0. Hence a "
        "hypothesis 'coefficient scaled by c' and a hypothesis 'forcing scaled "
        "by c' are observationally equivalent from the solution or its residual: "
        "no classifier can separate them without additional measurements or a "
        "prior that breaks the scaling symmetry. This is a genuine identifiability "
        "limit, not a small-sample artifact."
    )


def indistinguishable_pair(
    a_nodes: np.ndarray, f_interior: np.ndarray, c: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct the observationally equivalent pair ``(a, f)`` vs ``(c a, c f)``."""
    a = np.asarray(a_nodes, dtype=float)
    f = np.asarray(f_interior, dtype=float)
    return a, f, c * a, c * f
