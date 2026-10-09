"""Active learning tests (Problem B)."""

from __future__ import annotations

import numpy as np

from certno.active import RFFFeatures, BayesianLinearSurrogate, select_batch


def test_rff_and_bayesian_surrogate():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 3))
    y = X[:, 0] ** 2 + 0.1 * rng.standard_normal(40)
    feats = RFFFeatures(d_in=3, n_features=64, length_scale=1.0, rng=rng)
    sur = BayesianLinearSurrogate(feats, noise=1e-2)
    sur.fit(X[:30], y[:30])
    mean, std = sur.predict(X[30:])
    assert mean.shape == std.shape == (10,)
    assert np.all(std > 0)


def test_acquisition_policies():
    rng = np.random.default_rng(1)
    X_pool = rng.standard_normal((20, 2))
    X_test = rng.standard_normal((5, 2))
    y = X_pool.sum(axis=1) + 0.01 * rng.standard_normal(20)
    feats = RFFFeatures(d_in=2, n_features=32, rng=rng)
    sur = BayesianLinearSurrogate(feats).fit(X_pool, y)
    idx_u = select_batch(sur, X_pool, n_select=3, policy="uncertainty")
    idx_g = select_batch(sur, X_pool, n_select=3, policy="goal_oriented", X_test=X_test)
    idx_r = select_batch(sur, X_pool, n_select=3, policy="uniform", rng=rng)
    assert len(set(idx_u)) == 3 and len(set(idx_g)) == 3 and len(set(idx_r)) == 3
    costs = np.ones(20)
    idx_c = select_batch(sur, X_pool, n_select=3, policy="cost_aware", X_test=X_test, costs=costs)
    assert len(set(idx_c)) == 3
