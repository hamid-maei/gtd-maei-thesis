"""Linear gradient-TD algorithms (Maei, 2011, Chapter 5).

Each concrete subclass implements one paper equation pair:

  TD(0)  : eq. (2.10)
  TDC    : eq. (5.10) + (5.8b)
  GTD2   : eq. (5.8a) + (5.8b)
  GTD1   : eq. (5.2a) + (5.2b)   (the original "GTD")

All algorithms share the same shape:

    delta = reward + gamma * (theta . phi') - (theta . phi)

then a (theta, w) update parametrised by (alpha, beta = eta * alpha).

The classes hold mutable state (theta, w) and expose:

  * `update(phi, phi_p, reward, alpha, beta)` - one-step learning rule;
  * `reset()`                                  - re-zero theta and w.

They are intentionally trajectory-agnostic: the agent decides which
state -> phi mapping to apply. With `theta_0 = w_0 = 0` (the default),
results match Maei (2011) Fig 5.1 exactly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np


_VectorOrScalar = Union[np.ndarray, float]


# --------------------------------------------------------------------------- #
# Base class                                                                   #
# --------------------------------------------------------------------------- #

class LinearTDAlgorithm(ABC):
    """Abstract base for linear TD-style algorithms with an auxiliary `w`."""

    name: str = "base"

    def __init__(self, feature_dim: int, gamma: float = 1.0,
                 theta_init: _VectorOrScalar = 0.0,
                 w_init: _VectorOrScalar = 0.0) -> None:
        self.feature_dim = int(feature_dim)
        self.gamma = float(gamma)
        self._theta_init = theta_init
        self._w_init = w_init
        self.theta: np.ndarray = np.zeros(self.feature_dim)
        self.w: np.ndarray = np.zeros(self.feature_dim)
        self.reset()

    # -- helpers ------------------------------------------------------------ #

    @staticmethod
    def _materialise(value: _VectorOrScalar, dim: int) -> np.ndarray:
        if np.isscalar(value):
            return float(value) * np.ones(dim)
        arr = np.asarray(value, dtype=float).copy()
        if arr.shape != (dim,):
            raise ValueError(f"initial vector must have shape ({dim},), "
                             f"got {arr.shape}")
        return arr

    def reset(self) -> None:
        self.theta = self._materialise(self._theta_init, self.feature_dim)
        self.w = self._materialise(self._w_init, self.feature_dim)

    def td_error(self, phi: np.ndarray, phi_p: np.ndarray,
                 reward: float) -> float:
        return float(reward + self.gamma * (self.theta @ phi_p) -
                     (self.theta @ phi))

    # -- subclass interface ------------------------------------------------- #

    @abstractmethod
    def update(self, phi: np.ndarray, phi_p: np.ndarray, reward: float,
               alpha: float, beta: float) -> None:
        """Apply one transition's update in place."""

    # -- conveniences ------------------------------------------------------- #

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(feature_dim={self.feature_dim}, "
                f"gamma={self.gamma})")


# --------------------------------------------------------------------------- #
# Concrete algorithms                                                          #
# --------------------------------------------------------------------------- #

class TD0(LinearTDAlgorithm):
    """Conventional linear TD(0) (no auxiliary parameter).

    Update:
        theta <- theta + alpha * delta * phi
    """

    name = "TD"

    def update(self, phi, phi_p, reward, alpha, beta=0.0):
        delta = self.td_error(phi, phi_p, reward)
        self.theta = self.theta + alpha * delta * phi


class TDC(LinearTDAlgorithm):
    """Linear TDC - TD with gradient correction term (Maei, 2011 eq. 5.10).

    Updates:
        theta <- theta + alpha * (delta * phi - gamma * phi' * (w . phi))
        w     <- w     + beta  * (delta - (w . phi)) * phi
    """

    name = "TDC"

    def update(self, phi, phi_p, reward, alpha, beta):
        delta = self.td_error(phi, phi_p, reward)
        w_phi = self.w @ phi
        self.theta = (self.theta
                      + alpha * (delta * phi - self.gamma * phi_p * w_phi))
        self.w = self.w + beta * (delta - w_phi) * phi


class GTD2(LinearTDAlgorithm):
    """Linear GTD2 (Maei, 2011 eq. 5.8a-5.8b).

    Updates:
        theta <- theta + alpha * (phi - gamma * phi') * (w . phi)
        w     <- w     + beta  * (delta - (w . phi)) * phi
    """

    name = "GTD2"

    def update(self, phi, phi_p, reward, alpha, beta):
        delta = self.td_error(phi, phi_p, reward)
        w_phi = self.w @ phi
        self.theta = (self.theta
                      + alpha * (phi - self.gamma * phi_p) * w_phi)
        self.w = self.w + beta * (delta - w_phi) * phi


class GTD1(LinearTDAlgorithm):
    """Original GTD a.k.a. GTD1 (Maei, 2011 eq. 5.2a-5.2b).

    Note the distinctive `w` rule: an exponential-moving-average style
    update that is *not* of the form `(scalar) * phi`.

    Updates:
        theta <- theta + alpha * (phi - gamma * phi') * (w . phi)
        w     <- w     + beta  * (delta * phi - w)
    """

    name = "GTD"

    def update(self, phi, phi_p, reward, alpha, beta):
        delta = self.td_error(phi, phi_p, reward)
        w_phi = self.w @ phi
        self.theta = (self.theta
                      + alpha * (phi - self.gamma * phi_p) * w_phi)
        self.w = self.w + beta * (delta * phi - self.w)


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #

_REGISTRY = {
    "td0":  TD0,
    "td":   TD0,
    "tdc":  TDC,
    "gtd2": GTD2,
    "gtd1": GTD1,
    "gtd":  GTD1,   # the thesis labels GTD1 as just "GTD"
}


def make_algorithm(name: str, feature_dim: int, *,
                   gamma: float = 1.0,
                   theta_init: _VectorOrScalar = 0.0,
                   w_init: _VectorOrScalar = 0.0) -> LinearTDAlgorithm:
    """Look up an algorithm class by short name and instantiate it.

    >>> alg = make_algorithm("tdc", feature_dim=5, gamma=1.0)
    >>> isinstance(alg, TDC)
    True
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown algorithm '{name}'. "
                       f"Choices: {sorted(set(_REGISTRY))}")
    cls = _REGISTRY[key]
    return cls(feature_dim, gamma=gamma,
               theta_init=theta_init, w_init=w_init)


__all__ = [
    "LinearTDAlgorithm",
    "TD0", "TDC", "GTD2", "GTD1",
    "make_algorithm",
]
