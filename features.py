"""Linear feature representations used in Maei (2011) Fig 5.1.

Each FeatureMap maps an environment state index to a fixed-length
feature vector phi(s) in R^d. Terminal states return the all-zero
vector by convention; this is what the existing python_codes/ reference
sets via `phi_p = zeros(d)` when the next state is absorbing.

Feature maps also expose the full Phi matrix (one row per state,
including zeros on the terminal rows) so the closed-form RMSPBE
evaluator can pre-compute A, b, C without extra knowledge of the
environment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


# --------------------------------------------------------------------------- #
# Base class                                                                   #
# --------------------------------------------------------------------------- #

class FeatureMap(ABC):
    """Stateless map state_index -> feature vector."""

    n_states: int     # total state count (including terminals)
    feature_dim: int  # d - dimensionality of phi

    def __init__(self, n_states: int, feature_dim: int) -> None:
        self.n_states = int(n_states)
        self.feature_dim = int(feature_dim)
        self._matrix = self._build_matrix()
        if self._matrix.shape != (self.n_states, self.feature_dim):
            raise ValueError(
                f"feature matrix has shape {self._matrix.shape}; "
                f"expected ({self.n_states}, {self.feature_dim})"
            )

    @abstractmethod
    def _build_matrix(self) -> np.ndarray:
        """Construct the (n_states, feature_dim) Phi matrix."""

    # -- public surface ----------------------------------------------------- #

    def phi_matrix(self) -> np.ndarray:
        """Return the full (n_states, feature_dim) feature matrix."""
        return self._matrix

    def __call__(self, state: int) -> np.ndarray:
        if not 0 <= state < self.n_states:
            raise IndexError(f"state index {state} out of range "
                             f"[0, {self.n_states})")
        return self._matrix[state]

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(n_states={self.n_states}, "
                f"feature_dim={self.feature_dim})")


# --------------------------------------------------------------------------- #
# Random-walk feature representations                                          #
# --------------------------------------------------------------------------- #

class TabularFeatures(FeatureMap):
    """Identity (table-lookup) features for the K-state random walk.

    Non-terminal state j (j = 1..K) gets the standard basis vector
    e_{j-1} in R^K. Terminals (states 0 and K+1) get the zero vector.
    """

    def __init__(self, n_states: int = 5) -> None:
        K = int(n_states)
        if K < 1:
            raise ValueError("n_states (number of non-terminal states) must be >= 1")
        self.K = K
        super().__init__(n_states=K + 2, feature_dim=K)

    def _build_matrix(self) -> np.ndarray:
        K = self.K
        N = K + 2
        Phi = np.zeros((N, K))
        Phi[1:N - 1, :] = np.eye(K)
        return Phi


class DependentFeatures(FeatureMap):
    """Hand-crafted "dependent" features for the 5-state random walk.

    From Maei (2011), p. 48:
        phi(1) = (1, 0, 0)
        phi(2) = (1/sqrt(2), 1/sqrt(2), 0)
        phi(3) = (1/sqrt(3), 1/sqrt(3), 1/sqrt(3))
        phi(4) = (0, 1/sqrt(2), 1/sqrt(2))
        phi(5) = (0, 0, 1)

    With 3 features and 5 states the representation is insufficient to
    represent the true value function exactly. Terminals get zeros.
    """

    def __init__(self) -> None:
        super().__init__(n_states=7, feature_dim=3)

    def _build_matrix(self) -> np.ndarray:
        Phi = np.zeros((7, 3))
        Phi[1, 0] = 1.0
        Phi[2, 0] = 1.0 / np.sqrt(2.0)
        Phi[2, 1] = 1.0 / np.sqrt(2.0)
        Phi[3, :] = 1.0 / np.sqrt(3.0)
        Phi[4, 1] = 1.0 / np.sqrt(2.0)
        Phi[4, 2] = 1.0 / np.sqrt(2.0)
        Phi[5, 2] = 1.0
        return Phi


class InvertedFeatures(FeatureMap):
    """Inverted features for the K-state random walk.

    Each non-terminal state gets a length-K vector that is 1/2 in every
    coordinate except for a single 0 - chosen so that consecutive states
    look almost identical, causing pathological generalisation. The j-th
    middle state (j = 1..K) has a 0 in coordinate j-1 (0-based).

    From `get_randWalk_corr.m`: MPhi(j, j-1) = 0, all others 1/2.
    """

    def __init__(self, n_states: int = 5) -> None:
        K = int(n_states)
        if K < 1:
            raise ValueError("n_states (non-terminal states) must be >= 1")
        self.K = K
        super().__init__(n_states=K + 2, feature_dim=K)

    def _build_matrix(self) -> np.ndarray:
        K = self.K
        N = K + 2
        Phi = np.zeros((N, K))
        for j in range(1, N - 1):  # absolute state index
            Phi[j, :] = 0.5
            Phi[j, j - 1] = 0.0
        return Phi


# --------------------------------------------------------------------------- #
# Boyan chain features                                                         #
# --------------------------------------------------------------------------- #

class BoyanTriangleFeatures(FeatureMap):
    """Boyan-chain triangle features.

    Hardcoded for the standard 14-state, 4-feature configuration from
    Boyan (2002). State 0 (the terminal) gets zeros; states 1..13 get
    the overlapping triangle features.
    """

    _FEATURE_TABLE_14_4 = np.array([
        [0.00, 0.00, 0.00, 0.00],   # state 0 (terminal)
        [0.00, 0.00, 0.00, 1.00],
        [0.00, 0.00, 0.25, 0.75],
        [0.00, 0.00, 0.50, 0.50],
        [0.00, 0.00, 0.75, 0.25],
        [0.00, 0.00, 1.00, 0.00],
        [0.00, 0.25, 0.75, 0.00],
        [0.00, 0.50, 0.50, 0.00],
        [0.00, 0.75, 0.25, 0.00],
        [0.00, 1.00, 0.00, 0.00],
        [0.25, 0.75, 0.00, 0.00],
        [0.50, 0.50, 0.00, 0.00],
        [0.75, 0.25, 0.00, 0.00],
        [1.00, 0.00, 0.00, 0.00],
    ])

    def __init__(self, n_states: int = 14, feature_dim: int = 4) -> None:
        if (n_states, feature_dim) != (14, 4):
            raise NotImplementedError(
                "BoyanTriangleFeatures is only implemented for the "
                "standard (n_states=14, feature_dim=4) configuration.")
        super().__init__(n_states=n_states, feature_dim=feature_dim)

    def _build_matrix(self) -> np.ndarray:
        return self._FEATURE_TABLE_14_4.copy()


__all__ = [
    "FeatureMap",
    "TabularFeatures",
    "DependentFeatures",
    "InvertedFeatures",
    "BoyanTriangleFeatures",
]
