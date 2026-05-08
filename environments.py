"""Episodic Markov reward processes used in Maei (2011) Fig 5.1.

The environments here only provide dynamics (state index, reward, done)
and the closed-form transition / reward matrices required by the
RMSPBE evaluator. They are deliberately feature-agnostic; the agent is
responsible for turning state indices into feature vectors via a
FeatureMap.

State conventions:
* RandomWalkEnv: states 0 and N-1 are absorbing terminals. The
  non-terminal middle states are 1..N-2 (with N = K + 2 where K is the
  user-visible "number of states", default 5).
* BoyanChainEnv: state 0 is the absorbing terminal. Non-terminal states
  are 1..N-1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Base class                                                                   #
# --------------------------------------------------------------------------- #

class Environment(ABC):
    """Episodic MRP base class.

    Subclasses must define `n_states` (total number of states including
    terminals), `transition_matrix()`, `reward_vector()`, expose the
    initial-state distribution `mu()` over non-terminal states, and
    advance dynamics in `step()`.
    """

    n_states: int

    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed = seed
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self._state: int = -1
        self._done: bool = True

    # -- subclass interface -------------------------------------------------- #

    @abstractmethod
    def transition_matrix(self) -> np.ndarray:
        """Return the (N, N) row-stochastic transition matrix P."""

    @abstractmethod
    def reward_vector(self) -> np.ndarray:
        """Return the per-state expected reward r_bar of length N."""

    @abstractmethod
    def mu(self) -> np.ndarray:
        """Initial-state distribution over non-terminal states.

        The vector has one entry per non-terminal state and sums to 1.
        """

    @abstractmethod
    def non_terminal_states(self) -> np.ndarray:
        """Indices of the non-terminal states (used by the evaluator)."""

    @abstractmethod
    def is_terminal(self, state: int) -> bool:
        """Return True if `state` is an absorbing terminal."""

    @abstractmethod
    def initial_state(self) -> int:
        """Sample (or return the deterministic) starting state index."""

    @abstractmethod
    def _transition(self, state: int) -> tuple[int, float]:
        """Sample one transition: returns (next_state, reward)."""

    # -- standard RL interface ---------------------------------------------- #

    def reset(self, seed: Optional[int] = None) -> int:
        """Begin a new episode and return the initial state index."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._state = self.initial_state()
        self._done = self.is_terminal(self._state)
        return self._state

    def step(self) -> tuple[int, float, bool, dict]:
        """Advance one step. Returns (next_state, reward, done, info)."""
        if self._done:
            raise RuntimeError("step() called after the episode terminated; "
                               "call reset() first.")
        next_state, reward = self._transition(self._state)
        self._done = self.is_terminal(next_state)
        self._state = next_state
        return next_state, reward, self._done, {}

    # -- conveniences ------------------------------------------------------- #

    @property
    def state(self) -> int:
        return self._state

    @property
    def done(self) -> bool:
        return self._done


# --------------------------------------------------------------------------- #
# Concrete environments                                                        #
# --------------------------------------------------------------------------- #

class RandomWalkEnv(Environment):
    """Symmetric K-state random walk with two absorbing terminals.

    State indexing follows MATLAB's convention from
    `get_randWalk_tab.m`:

      * Total states  N = K + 2.
      * State 0       = left terminal (reward 0).
      * States 1..K   = non-terminal middle states.
      * State N - 1   = right terminal (reward 1 on entry).

    Each non-terminal state transitions 50/50 to its neighbour.
    """

    def __init__(self, n_states: int = 5, seed: Optional[int] = None) -> None:
        super().__init__(seed)
        if n_states < 1:
            raise ValueError("n_states (number of middle states) must be >= 1")
        self.K = int(n_states)
        self.n_states = self.K + 2

    # -- closed-form helpers ------------------------------------------------ #

    def transition_matrix(self) -> np.ndarray:
        N = self.n_states
        P = np.zeros((N, N))
        for i in range(1, N - 1):
            P[i, i - 1] = 0.5
            P[i, i + 1] = 0.5
        P[0, 0] = 1.0
        P[N - 1, N - 1] = 1.0
        return P

    def reward_vector(self) -> np.ndarray:
        """E[R | S = s], length N. Reward 1 only on entry to the right
        terminal; absorbing self-loops have reward 0.
        """
        N = self.n_states
        R = np.zeros((N, N))
        R[N - 2, N - 1] = 1.0
        P = self.transition_matrix()
        return np.sum(P * R, axis=1)

    def non_terminal_states(self) -> np.ndarray:
        return np.arange(1, self.n_states - 1, dtype=int)

    def mu(self) -> np.ndarray:
        """Start in the middle non-terminal state with probability 1."""
        mu = np.zeros(self.K)
        mu[self.K // 2] = 1.0  # MATLAB's ceil(K/2) (1-based) == K // 2 (0-based)
        return mu

    def is_terminal(self, state: int) -> bool:
        return state == 0 or state == self.n_states - 1

    def initial_state(self) -> int:
        # 0-based middle of the K non-terminal states. MATLAB uses
        # ceil(K/2) (1-based) which maps to (K // 2) + 1 in 1-based or
        # (K // 2) + 1 in MATLAB index space, i.e. middle state.
        return 1 + self.K // 2

    def _transition(self, state: int) -> tuple[int, float]:
        if self.rng.random() >= 0.5:
            next_state = state + 1
        else:
            next_state = state - 1
        # Reward is 1 only on entry to the right terminal.
        reward = 1.0 if next_state == self.n_states - 1 else 0.0
        return next_state, reward


class BoyanChainEnv(Environment):
    """14-state Boyan chain (Boyan, 2002; Maei, 2011 Fig 5.1).

    Single absorbing terminal at state 0. Episodes start in state N - 1
    (state 13 with the default 14 states).

      * From state s > 2: 50/50 transition to s - 1 or s - 2; reward -3.
      * From state 2:     deterministic transition to state 1; reward -2.
      * From state 1:     deterministic transition to terminal 0;
                          reward 0.
    """

    def __init__(self, n_states: int = 14, seed: Optional[int] = None) -> None:
        super().__init__(seed)
        if n_states < 3:
            raise ValueError("n_states must be >= 3 for the Boyan chain")
        self.n_states = int(n_states)

    # -- closed-form helpers ------------------------------------------------ #

    def transition_matrix(self) -> np.ndarray:
        N = self.n_states
        P = np.zeros((N, N))
        # state 0 = terminal, absorbing
        P[0, 0] = 1.0
        # state 1 -> 0
        P[1, 0] = 1.0
        # state 2 -> 1
        P[2, 1] = 1.0
        # state s >= 3: 50/50 to s-1 / s-2
        for s in range(3, N):
            P[s, s - 1] = 0.5
            P[s, s - 2] = 0.5
        return P

    def reward_vector(self) -> np.ndarray:
        N = self.n_states
        R = np.zeros((N, N))
        R[1, 0] = 0.0  # explicit
        R[2, 1] = -2.0
        for s in range(3, N):
            R[s, s - 1] = -3.0
            R[s, s - 2] = -3.0
        P = self.transition_matrix()
        return np.sum(P * R, axis=1)

    def non_terminal_states(self) -> np.ndarray:
        return np.arange(1, self.n_states, dtype=int)

    def mu(self) -> np.ndarray:
        """Start in the last non-terminal state with probability 1."""
        mu = np.zeros(self.n_states - 1)
        mu[-1] = 1.0
        return mu

    def is_terminal(self, state: int) -> bool:
        return state == 0

    def initial_state(self) -> int:
        return self.n_states - 1

    def _transition(self, state: int) -> tuple[int, float]:
        if state > 2:
            next_state = state - 1 if self.rng.random() < 0.5 else state - 2
            reward = -3.0
        elif state == 2:
            next_state, reward = 1, -2.0
        elif state == 1:
            next_state, reward = 0, 0.0
        else:
            raise RuntimeError("Cannot transition from terminal state 0")
        return next_state, reward


__all__ = ["Environment", "RandomWalkEnv", "BoyanChainEnv"]
