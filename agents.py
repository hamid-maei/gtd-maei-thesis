"""LinearTDAgent - composes a feature map with a learning algorithm.

Following the OAK split documented in the project's CLAUDE.md (and
mirroring src/agents.py at the repo root), the agent:

  * holds a `FeatureMap` to turn environment state indices into
    feature vectors;
  * holds a `LinearTDAlgorithm` that owns theta and w and implements
    one update rule;
  * stores per-run step sizes (alpha, beta) so a sweep can change them
    between sweep points without rebuilding the agent.

The environment never sees feature vectors, and the algorithm never
sees state indices - the agent is the only object that crosses that
boundary.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from algorithms import (
    LinearTDAlgorithm,
    make_algorithm,
)
from features import FeatureMap


class LinearTDAgent:
    """Glue between a feature representation and a TD-style algorithm.

    Parameters
    ----------
    algorithm : LinearTDAlgorithm
        The learning rule (TD0 / TDC / GTD2 / GTD1).
    feature_map : FeatureMap
        Maps state indices to feature vectors. The agent assumes that
        terminal states are encoded as the zero vector inside the
        feature map; no explicit "done" handling is needed in `update`.
    alpha : float, default 0.1
        Primary step size.
    beta : float, default 0.0
        Secondary step size (passed through to TDC / GTD2 / GTD1).
        TD0 ignores it.
    """

    def __init__(self, algorithm: LinearTDAlgorithm, feature_map: FeatureMap,
                 alpha: float = 0.1, beta: float = 0.0) -> None:
        if algorithm.feature_dim != feature_map.feature_dim:
            raise ValueError(
                f"Algorithm feature_dim={algorithm.feature_dim} does not "
                f"match feature_map feature_dim={feature_map.feature_dim}.")
        self.algorithm = algorithm
        self.feature_map = feature_map
        self.alpha = float(alpha)
        self.beta = float(beta)

    # -- factory ------------------------------------------------------------ #

    @classmethod
    def make(cls, algorithm_name: str, feature_map: FeatureMap, *,
             alpha: float = 0.1, beta: float = 0.0,
             gamma: float = 1.0) -> "LinearTDAgent":
        """Build an agent from a short algorithm name.

        >>> from features import TabularFeatures
        >>> agent = LinearTDAgent.make("tdc", TabularFeatures(5), alpha=0.1)
        """
        algo = make_algorithm(algorithm_name,
                              feature_dim=feature_map.feature_dim,
                              gamma=gamma)
        return cls(algo, feature_map, alpha=alpha, beta=beta)

    # -- step-size control -------------------------------------------------- #

    def set_step_sizes(self, alpha: float, beta: float) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)

    # -- standard RL interface --------------------------------------------- #

    def reset(self) -> None:
        """Re-zero the algorithm's parameters (theta and w)."""
        self.algorithm.reset()

    def act(self, obs: int) -> Optional[int]:
        """Policy is fixed for these problems, so no action is chosen."""
        return None

    def observe(self, obs: int) -> np.ndarray:
        """Map a state index to its feature vector."""
        return self.feature_map(obs)

    def update(self, obs: int, action: Optional[int], reward: float,
               next_obs: int, done: bool) -> None:
        """Apply one transition's update."""
        phi = self.feature_map(obs)
        phi_p = self.feature_map(next_obs)
        # The feature map already returns zeros for terminal states, so
        # phi_p naturally vanishes on absorbing transitions. The `done`
        # flag is accepted for interface compatibility but not needed.
        del action, done
        self.algorithm.update(phi, phi_p, reward, self.alpha, self.beta)

    # -- forwarded parameter views ----------------------------------------- #

    @property
    def theta(self) -> np.ndarray:
        return self.algorithm.theta

    @property
    def w(self) -> np.ndarray:
        return self.algorithm.w

    @property
    def name(self) -> str:
        return self.algorithm.name

    # -- conveniences ------------------------------------------------------- #

    def __repr__(self) -> str:
        return (f"LinearTDAgent(algorithm={self.algorithm!r}, "
                f"feature_map={self.feature_map!r}, "
                f"alpha={self.alpha}, beta={self.beta})")


__all__ = ["LinearTDAgent"]
