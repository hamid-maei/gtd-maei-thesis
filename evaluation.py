"""Closed-form RMSPBE evaluator.

Given an environment and a feature map, this module precomputes the
matrices

    A = Phi^T D (gamma P - I) Phi
    b = Phi^T D r_bar
    C = Phi^T D Phi

restricted to the non-terminal states, where D is the diagonal matrix
of expected non-terminal visits starting from the environment's initial
state distribution mu. Then for any theta:

    RMSPBE(theta) = sqrt( (A theta + b)^T C^{-1} (A theta + b) )

This mirrors the per-script `compute_A_b_C_*` helpers in
`archive/Fast_Gradient_TD_Codes_ICML09paper/python_codes/common.py` but is written
once in environment-agnostic form, by taking advantage of the
`Environment.transition_matrix` and `FeatureMap.phi_matrix` interfaces.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from environments import Environment
from features import FeatureMap


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _expected_visits(P_non_terminal: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Diagonal of the visit-count distribution (normalised to sum to 1).

    visits = mu^T (I - Q)^{-1}
    D_diag = visits / |visits|_1

    where Q is the sub-matrix of P restricted to non-terminal states.
    """
    n = P_non_terminal.shape[0]
    visits = mu @ np.linalg.inv(np.eye(n) - P_non_terminal)
    total = np.linalg.norm(visits, ord=1)
    if total <= 0.0:
        raise ValueError("Visit distribution sums to 0; check mu and P.")
    return visits / total


# --------------------------------------------------------------------------- #
# Evaluator                                                                    #
# --------------------------------------------------------------------------- #

class RMSPBEEvaluator:
    """Closed-form sqrt(MSPBE) evaluator for a fixed (env, feature_map).

    Parameters
    ----------
    env : Environment
        Source of the transition matrix, reward vector, initial state
        distribution, and the index list of non-terminal states.
    feature_map : FeatureMap
        Provides the (n_states, feature_dim) Phi matrix. Terminal rows
        are expected to be zero (which the standard FeatureMap classes
        in features.py guarantee).
    gamma : float, default 1.0
        Discount factor.
    """

    def __init__(self, env: Environment, feature_map: FeatureMap,
                 gamma: float = 1.0) -> None:
        if env.n_states != feature_map.n_states:
            raise ValueError(
                f"Environment has {env.n_states} states but feature map "
                f"declares {feature_map.n_states}.")
        self.env = env
        self.feature_map = feature_map
        self.gamma = float(gamma)
        self._build()

    # -- precompute --------------------------------------------------------- #

    def _build(self) -> None:
        N = self.env.n_states
        P = self.env.transition_matrix()
        r_bar = self.env.reward_vector()
        Phi_full = self.feature_map.phi_matrix()

        non_terminal = self.env.non_terminal_states()
        Q = P[np.ix_(non_terminal, non_terminal)]
        mu = self.env.mu()
        D_non_terminal = _expected_visits(Q, mu)

        # Lift D_non_terminal back to a diagonal of size N (zeros on terminals).
        DD = np.zeros((N, N))
        DD[non_terminal, non_terminal] = D_non_terminal

        gamma = self.gamma
        A = Phi_full.T @ DD @ (gamma * P - np.eye(N)) @ Phi_full
        b = Phi_full.T @ DD @ r_bar
        C = Phi_full.T @ DD @ Phi_full

        # Cache
        self.A: np.ndarray = A
        self.b: np.ndarray = b
        self.C: np.ndarray = C
        self.inv_C: np.ndarray = np.linalg.inv(C)
        self.D_diag: np.ndarray = D_non_terminal

    # -- public surface ----------------------------------------------------- #

    def __call__(self, theta: np.ndarray) -> float:
        """Return RMSPBE(theta)."""
        v = self.A @ theta + self.b
        val = float(v @ self.inv_C @ v)
        return float(np.sqrt(val)) if val > 0.0 else 0.0

    def msbe_components(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (A, b, C) for callers that want them directly."""
        return self.A, self.b, self.C

    def theta_star(self) -> np.ndarray:
        """The fixed-point: theta* = -A^{-1} b. Useful for tests."""
        return np.linalg.solve(self.A, -self.b)

    def __repr__(self) -> str:
        return (f"RMSPBEEvaluator(env={type(self.env).__name__}, "
                f"feature_map={self.feature_map!r}, gamma={self.gamma})")


__all__ = ["RMSPBEEvaluator"]
