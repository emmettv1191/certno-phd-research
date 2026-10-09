"""Diagnostics tests (Problem D)."""

from __future__ import annotations

import numpy as np

from certno.diagnostics import residual_features, LogisticClassifier, identifiability_note


def test_residual_features_shape_and_range():
    r = np.sin(np.linspace(0, 1, 100))
    f = residual_features(r, h=0.01)
    assert f.shape == (6,)
    assert np.all(np.isfinite(f))


def test_logistic_classifier_trains():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    X[:25] += 1
    y = np.concatenate([np.zeros(25), np.ones(25)])
    clf = LogisticClassifier(n_features=4, n_classes=2)
    clf.fit(X, y, epochs=200, lr=0.5)
    assert clf.accuracy(X, y) > 0.85


def test_identifiability_note_exists():
    assert "scaling" in identifiability_note().lower()
