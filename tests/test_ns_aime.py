"""
Unit tests for NS-AIME core mathematics (paper compliance).

Run:  pytest -q   (from the ns_aime_pkg directory)

These tests verify the properties the paper relies on:
  1. Eq. (1) inverse surrogate shape & determinism.
  2. A_dagger is IMMUTABLE through fit() and optimize() (required for the
     dual-reporting protocol, Section 3.4).
  3. Rule Satisfaction (RS) does not decrease after re-constraint (Eq. 2).
  4. RD / GA metric formulas (Eq. 3, Eq. 4).
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ns_aime_xai2026 import NSAIME, compute_metrics, calculate_metrics  # noqa: E402


def _toy(n=400, d=6, m=2, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    X[:, 0] = (X[:, 0] > 0).astype(float)        # a binary feature
    logit = 1.5 * X[:, 0] + 1.0 * X[:, 2] - 1.1 * X[:, 4] + 0.5 * rng.randn(n)
    p = 1.0 / (1.0 + np.exp(-logit))
    Yhat = np.c_[1 - p, p]
    names = [f"x{i}" for i in range(d)]
    return X, Yhat, names


def test_inverse_operator_shape_and_determinism():
    X, Yhat, names = _toy()
    ex = NSAIME().fit(X, Yhat, feature_names=names)
    d, m = X.shape[1], Yhat.shape[1]
    assert ex.A_dagger_.shape == (d, m)
    # Eq. (1) is deterministic: recomputation is identical.
    A2 = ex._inverse_operator(ex._Xp_cache, Yhat)
    assert np.allclose(A2, ex.A_dagger_)


def test_a_dagger_is_immutable():
    """fit() + optimize() must not modify the raw inverse surrogate."""
    X, Yhat, names = _toy()
    ex = NSAIME().fit(X, Yhat, feature_names=names)
    A_before = ex.A_dagger_.copy()
    ex.optimize()
    assert np.allclose(A_before, ex.A_dagger_), "A_dagger was mutated!"


def test_rule_satisfaction_does_not_decrease():
    X, Yhat, names = _toy()
    ex = NSAIME(beta=8.0).fit(X, Yhat, feature_names=names)
    B = ex.optimize()
    raw = compute_metrics(ex.A_dagger_, ex.A_dagger_, ex._Xp_cache, ex.rules_)
    ns = compute_metrics(B, ex.A_dagger_, ex._Xp_cache, ex.rules_)
    assert ns["rs"] >= raw["rs"] - 1e-9


def test_metric_formulas():
    X, Yhat, names = _toy()
    ex = NSAIME().fit(X, Yhat, feature_names=names)
    B = ex.optimize()
    rd, ga, sp, rs = calculate_metrics(B, ex.A_dagger_, ex._Xp_cache, ex.rules_)
    # Eq. (3) RD
    expect_rd = np.linalg.norm(B - ex.A_dagger_) / np.linalg.norm(ex.A_dagger_)
    assert abs(rd - expect_rd) < 1e-9
    # GA is a mean cosine -> within [-1, 1]
    assert -1.0 - 1e-9 <= ga <= 1.0 + 1e-9
    # beta=0 -> B == A_dagger -> RD 0, GA 1
    ex0 = NSAIME(beta=0.0, alpha=0.0).fit(X, Yhat, feature_names=names)
    B0 = ex0.optimize()
    rd0, ga0, _, _ = calculate_metrics(B0, ex0.A_dagger_, ex0._Xp_cache, ex0.rules_)
    assert rd0 < 1e-6 and ga0 > 1.0 - 1e-6
