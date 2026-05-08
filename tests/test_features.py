"""Smoke tests for the feature representations."""

from __future__ import annotations

import numpy as np
import pytest

from features import (
    BoyanTriangleFeatures,
    DependentFeatures,
    InvertedFeatures,
    TabularFeatures,
)


def test_tabular_features_zeros_terminals_and_eye_interior():
    fm = TabularFeatures(n_states=5)
    Phi = fm.phi_matrix()
    assert Phi.shape == (7, 5)
    assert np.allclose(Phi[0], 0.0)
    assert np.allclose(Phi[-1], 0.0)
    assert np.allclose(Phi[1:6], np.eye(5))
    for s in range(7):
        np.testing.assert_array_equal(fm(s), Phi[s])


def test_dependent_features_match_thesis_p48():
    fm = DependentFeatures()
    Phi = fm.phi_matrix()
    assert Phi.shape == (7, 3)
    np.testing.assert_allclose(Phi[1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(Phi[2], [1 / np.sqrt(2), 1 / np.sqrt(2), 0.0])
    np.testing.assert_allclose(Phi[3], [1 / np.sqrt(3)] * 3)
    np.testing.assert_allclose(Phi[4], [0.0, 1 / np.sqrt(2), 1 / np.sqrt(2)])
    np.testing.assert_allclose(Phi[5], [0.0, 0.0, 1.0])
    assert np.allclose(Phi[0], 0.0)
    assert np.allclose(Phi[-1], 0.0)


def test_inverted_features_zero_in_state_column():
    fm = InvertedFeatures(n_states=5)
    Phi = fm.phi_matrix()
    assert Phi.shape == (7, 5)
    for s in range(1, 6):
        row = Phi[s]
        zero_idx = np.where(row == 0.0)[0]
        assert zero_idx.tolist() == [s - 1]
        non_zero = np.delete(row, zero_idx)
        assert np.allclose(non_zero, 0.5)


def test_boyan_triangle_features_shape_and_terminals():
    fm = BoyanTriangleFeatures(n_states=14, feature_dim=4)
    Phi = fm.phi_matrix()
    assert Phi.shape == (14, 4)
    assert np.allclose(Phi[0], 0.0)
    np.testing.assert_allclose(Phi[1],  [0.00, 0.00, 0.00, 1.00])
    np.testing.assert_allclose(Phi[5],  [0.00, 0.00, 1.00, 0.00])
    np.testing.assert_allclose(Phi[9],  [0.00, 1.00, 0.00, 0.00])
    np.testing.assert_allclose(Phi[13], [1.00, 0.00, 0.00, 0.00])


def test_boyan_unsupported_config_raises():
    with pytest.raises(NotImplementedError):
        BoyanTriangleFeatures(n_states=10, feature_dim=4)
