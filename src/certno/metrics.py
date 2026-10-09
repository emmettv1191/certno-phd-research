"""Error, coverage, sharpness, and cost metrics for the certificate study."""

from __future__ import annotations

import numpy as np


def relative_l2(pred: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> float:
    pred = np.asarray(pred, dtype=float).ravel()
    true = np.asarray(true, dtype=float).ravel()
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + eps))


def max_norm_error(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float).ravel()
    true = np.asarray(true, dtype=float).ravel()
    return float(np.max(np.abs(pred - true)))


def relative_h1(
    pred: np.ndarray,
    true: np.ndarray,
    h: float,
    eps: float = 1e-12,
) -> float:
    """Discrete relative H1 seminorm error on a uniform grid."""
    pred = np.asarray(pred, dtype=float).ravel()
    true = np.asarray(true, dtype=float).ravel()
    dp = np.diff(pred) / h
    dt = np.diff(true) / h
    return float(np.linalg.norm(dp - dt) / (np.linalg.norm(dt) + eps))


def coverage(bounds: np.ndarray, errors: np.ndarray) -> float:
    """Fraction of predictions for which ``bound >= error`` (empirical coverage)."""
    bounds = np.asarray(bounds, dtype=float).ravel()
    errors = np.asarray(errors, dtype=float).ravel()
    if bounds.size == 0:
        return float("nan")
    return float(np.mean(bounds >= errors - 1e-12))


def sharpness(bounds: np.ndarray, errors: np.ndarray, eps: float = 1e-12) -> float:
    """Median ratio ``bound / max(error, eps)``; lower is sharper."""
    bounds = np.asarray(bounds, dtype=float).ravel()
    errors = np.asarray(errors, dtype=float).ravel()
    ratio = bounds / np.maximum(errors, eps)
    return float(np.median(ratio))


def false_negative_rate(
    bounds: np.ndarray, errors: np.ndarray, tolerance: float, eps: float = 1e-12
) -> float:
    """Fraction where the true error exceeds ``tolerance`` but the bound accepts.

    A false negative corresponds to an accepted prediction whose actual error is
    above the engineering tolerance -- the failure mode the certificate is
    supposed to prevent.
    """
    bounds = np.asarray(bounds, dtype=float).ravel()
    errors = np.asarray(errors, dtype=float).ravel()
    bad = errors > tolerance
    if not np.any(bad):
        return 0.0
    accepted_despite_bad = bad & (bounds <= tolerance + eps)
    return float(np.mean(accepted_despite_bad[bad]))


def confidence_interval(
    values: np.ndarray, *, level: float = 0.95, n_boot: int = 2000, rng=None
) -> tuple[float, float]:
    """Nonparametric bootstrap CI for the mean of ``values``."""
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = rng or np.random.default_rng(0)
    means = np.empty(n_boot)
    n = values.size
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = values[idx].mean()
    lo = float(np.quantile(means, (1 - level) / 2))
    hi = float(np.quantile(means, 1 - (1 - level) / 2))
    return lo, hi


def summarize(
    errors: np.ndarray,
    bounds: np.ndarray,
    *,
    tolerance: float,
) -> dict:
    """Bundle the headline reliability statistics for one evaluation set."""
    return {
        "n": int(np.asarray(errors).size),
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "max_error": float(np.max(errors)),
        "coverage": coverage(bounds, errors),
        "sharpness": sharpness(bounds, errors),
        "false_negative_rate": false_negative_rate(bounds, errors, tolerance),
        "mean_bound": float(np.mean(bounds)),
        "tolerance": tolerance,
    }
