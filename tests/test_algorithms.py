"""Hand-computed smoke tests for the four learning algorithms.

For each algorithm we set up a 2-feature toy problem, run one update,
and compare against the closed-form expected (theta, w) computed
directly from the equations in Maei (2011) Chapter 5.
"""

from __future__ import annotations

import numpy as np
import pytest

from algorithms import GTD1, GTD2, TD0, TDC, make_algorithm


PHI   = np.array([1.0, 0.0])
PHI_P = np.array([0.0, 1.0])
REWARD = 1.0
ALPHA = 0.1
BETA  = 0.05
GAMMA = 1.0


def test_td0_one_step():
    alg = TD0(feature_dim=2, gamma=GAMMA)
    alg.update(PHI, PHI_P, REWARD, ALPHA)
    np.testing.assert_allclose(alg.theta, [0.1, 0.0])
    np.testing.assert_allclose(alg.w, [0.0, 0.0])


def test_tdc_one_step():
    alg = TDC(feature_dim=2, gamma=GAMMA)
    alg.update(PHI, PHI_P, REWARD, ALPHA, BETA)
    np.testing.assert_allclose(alg.theta, [0.1, 0.0])  # delta=1, w_phi=0
    np.testing.assert_allclose(alg.w, [0.05, 0.0])     # beta * 1 * phi


def test_gtd2_one_step():
    alg = GTD2(feature_dim=2, gamma=GAMMA)
    alg.update(PHI, PHI_P, REWARD, ALPHA, BETA)
    # First step: w_phi=0, so theta unchanged.
    np.testing.assert_allclose(alg.theta, [0.0, 0.0])
    np.testing.assert_allclose(alg.w, [0.05, 0.0])


def test_gtd1_one_step():
    alg = GTD1(feature_dim=2, gamma=GAMMA)
    alg.update(PHI, PHI_P, REWARD, ALPHA, BETA)
    # First step: w_phi=0, so theta unchanged. w update: beta*(delta*phi - 0).
    np.testing.assert_allclose(alg.theta, [0.0, 0.0])
    np.testing.assert_allclose(alg.w, [0.05, 0.0])


def test_factory_returns_correct_classes():
    assert isinstance(make_algorithm("td0",  2), TD0)
    assert isinstance(make_algorithm("tdc",  2), TDC)
    assert isinstance(make_algorithm("gtd2", 2), GTD2)
    assert isinstance(make_algorithm("gtd1", 2), GTD1)
    assert isinstance(make_algorithm("gtd",  2), GTD1)
    assert isinstance(make_algorithm("td",   2), TD0)
    with pytest.raises(KeyError):
        make_algorithm("nonsense", 2)


def test_reset_zeros_state():
    alg = TDC(feature_dim=2, gamma=GAMMA)
    alg.update(PHI, PHI_P, REWARD, ALPHA, BETA)
    alg.reset()
    np.testing.assert_allclose(alg.theta, 0.0)
    np.testing.assert_allclose(alg.w, 0.0)


def test_dim_mismatch_raises():
    alg = TDC(feature_dim=2, gamma=GAMMA)
    with pytest.raises(ValueError):
        TDC(feature_dim=2, gamma=GAMMA, theta_init=np.zeros(3))
