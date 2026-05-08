"""Tests for the closed-form RMSPBE evaluator."""

from __future__ import annotations

import numpy as np

from environments import BoyanChainEnv, RandomWalkEnv
from evaluation import RMSPBEEvaluator
from features import (
    BoyanTriangleFeatures,
    DependentFeatures,
    InvertedFeatures,
    TabularFeatures,
)


def test_random_walk_tabular_theta_star_zeros_rmspbe():
    env = RandomWalkEnv(n_states=5)
    fm = TabularFeatures(n_states=5)
    eval = RMSPBEEvaluator(env, fm, gamma=1.0)
    theta_star = eval.theta_star()
    # tabular -> exact representation, theta* = (1/6, 2/6, 3/6, 4/6, 5/6)
    np.testing.assert_allclose(theta_star,
                               np.arange(1, 6) / 6, atol=1e-12)
    assert eval(theta_star) < 1e-10


def test_random_walk_dependent_rmspbe_decreases_in_solution():
    env = RandomWalkEnv(n_states=5)
    fm = DependentFeatures()
    eval = RMSPBEEvaluator(env, fm, gamma=1.0)
    rmspbe_zero = eval(np.zeros(3))
    rmspbe_star = eval(eval.theta_star())
    assert rmspbe_star < rmspbe_zero
    assert rmspbe_star < 1e-10


def test_random_walk_inverted_rmspbe_at_solution_is_zero():
    env = RandomWalkEnv(n_states=5)
    fm = InvertedFeatures(n_states=5)
    eval = RMSPBEEvaluator(env, fm, gamma=1.0)
    assert eval(eval.theta_star()) < 1e-10


def test_boyan_rmspbe_at_solution_is_zero():
    env = BoyanChainEnv(n_states=14)
    fm = BoyanTriangleFeatures(n_states=14, feature_dim=4)
    eval = RMSPBEEvaluator(env, fm, gamma=1.0)
    assert eval(eval.theta_star()) < 1e-10


def test_dimension_mismatch_raises():
    env = RandomWalkEnv(n_states=5)
    fm = BoyanTriangleFeatures()
    try:
        RMSPBEEvaluator(env, fm, gamma=1.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched n_states")
