"""Metrics tests."""

from __future__ import annotations

import numpy as np

from certno import metrics


def test_coverage_and_fnr():
    bounds = np.array([0.01, 0.001, 0.1])
    errors = np.array([0.005, 0.002, 0.05])
    assert abs(metrics.coverage(bounds, errors) - 2.0 / 3.0) < 1e-12
    fnr = metrics.false_negative_rate(bounds, errors, tolerance=0.02)
    # only third: error 0.05>0.02, bound 0.1>0.02 so accepted? bound>tol -> not accepted; no false negative. second: err=0.002<0.02; first bound>err but check: bounds for bad cases (error>tol)? third error>tol but bound 0.1>tol means certificate would reject -> no fn. First: err 0.005<0.02. So FNR should be 0.
    assert fnr == 0.0
    # create false negative: error>tol, bound<=tol
    bounds2 = np.array([0.01])
    errors2 = np.array([0.05])
    assert metrics.false_negative_rate(bounds2, errors2, tolerance=0.02) == 1.0


def test_sharpness_and_rel_errors():
    pred = np.array([1.0, 2.0, 3.0])
    true = np.array([1.01, 1.99, 3.0])
    assert metrics.relative_l2(pred, true) < 0.01
    h = 0.1
    assert metrics.relative_h1(pred, true, h) >= 0.0
    assert metrics.max_norm_error(pred, true) < 0.02
