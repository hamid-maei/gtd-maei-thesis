"""Trainer + parameter-sweep utilities.

The `Trainer` runs a single episode end-to-end against one agent. The
`ParameterSweep` does the heavy lifting required by Maei (2011) Fig 5.1:
for each `(eta, alpha)` cell of the sweep grid it spins up one agent
per algorithm, sharing the same environment trajectory across all of
them (paired comparison), and records per-episode RMSPBE.

Saving uses the same npz schema as
`archive/Fast_Gradient_TD_Codes_ICML09paper/python_codes/common.py`
`save_results`, so the plotting code reads either folder identically.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence, Type

import numpy as np

from agents import LinearTDAgent
from algorithms import GTD1, GTD2, LinearTDAlgorithm, TDC
from common import best_i_j
from environments import Environment
from evaluation import RMSPBEEvaluator
from features import FeatureMap


# --------------------------------------------------------------------------- #
# Single-episode trainer                                                       #
# --------------------------------------------------------------------------- #

class Trainer:
    """Run one episode of an `Environment` x `LinearTDAgent`.

    The trainer is intentionally tiny: it walks the environment until
    `done` and asks the agent to update on every transition. It exposes
    `run_episode` (returns the trajectory length) and
    `evaluate_episode` (returns the post-episode RMSPBE).
    """

    def __init__(self, env: Environment, agent: LinearTDAgent,
                 evaluator: Optional[RMSPBEEvaluator] = None) -> None:
        self.env = env
        self.agent = agent
        self.evaluator = evaluator

    def run_episode(self, seed: Optional[int] = None) -> int:
        """Walk one episode end-to-end. Returns the number of steps."""
        obs = self.env.reset(seed=seed)
        steps = 0
        while not self.env.done:
            action = self.agent.act(obs)
            next_obs, reward, done, _ = self.env.step()
            self.agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            steps += 1
        return steps

    def evaluate(self) -> float:
        """Return RMSPBE of the agent's current theta."""
        if self.evaluator is None:
            raise RuntimeError("Trainer was created without an evaluator.")
        return self.evaluator(self.agent.theta)


# --------------------------------------------------------------------------- #
# Sweep result dataclass                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class SweepResult:
    """Container for the output of a `ParameterSweep`.

    Mirrors the npz schema written by the reference implementation in
    `python_codes/common.py::save_results` (under the ICML09 archive): each algorithm gets its
    own `RMSPBE_<NAME>` array of shape `(L_eta, L_alpha, episodes)`.
    """
    rmspbe: dict[str, np.ndarray]
    alphas: np.ndarray
    etas: np.ndarray
    runs: int
    episodes: int
    elapsed_seconds: float = 0.0
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    def save(self, path: str) -> str:
        """Write a `.npz` matching the reference schema."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "alphaStepSizeVec": np.asarray(self.alphas, dtype=float),
            "stepSizeRatioVec": np.asarray(self.etas, dtype=float),
            "runs": np.asarray([self.runs]),
            "nEpisodes": np.asarray([self.episodes]),
        }
        for name, arr in self.rmspbe.items():
            payload[f"RMSPBE_{name}"] = arr
        payload.update(self.extra)
        np.savez(path, **payload)
        return path

    # -- conveniences ------------------------------------------------------- #

    def best_indices(self, name: str, exclude_eta_zero: bool = True
                     ) -> tuple[int, int]:
        """Return the (i_eta, j_alpha) cell minimising the average
        RMSPBE for a given algorithm.
        """
        arr = self.rmspbe[name].mean(axis=2)  # (L_eta, L_alpha)
        if exclude_eta_zero:
            i, j = best_i_j(arr[1:, :])
            return i + 1, j
        return best_i_j(arr)


# --------------------------------------------------------------------------- #
# Parameter sweep                                                              #
# --------------------------------------------------------------------------- #

class ParameterSweep:
    """Sweep over `(eta, alpha)` and record per-episode RMSPBE.

    All algorithms registered with the sweep share the same environment
    trajectory at each step (paired comparison) - this is critical for
    reproducing the variance / ranking properties of Fig 5.1 and is
    exactly how the reference implementation in `python_codes/` (ICML09 archive)
    works.

    Parameters
    ----------
    env : Environment
        Concrete environment instance. Its `rng` is reseeded once at
        the start of `run()` so that runs are deterministic.
    feature_map : FeatureMap
        Shared by all agents; controls which feature representation is
        used.
    evaluator : RMSPBEEvaluator
        Computes per-episode RMSPBE from the algorithm's `theta`.
    algorithms : sequence of `LinearTDAlgorithm` subclasses
        One agent per class is built per `(eta, alpha)` cell. The class
        name (`cls.__name__`) becomes the key in the result dict (e.g.
        `"TDC"`, `"GTD2"`, `"GTD1"`).
    alphas, etas : 1-D iterables of float
        Sweep grids. `eta = beta / alpha`. Including `eta = 0` is
        useful because the `eta = 0` row of the TDC sweep is precisely
        TD(0).
    runs : int
        Number of independent runs to average over.
    episodes : int
        Number of episodes per run.
    gamma : float
        Discount factor (default 1.0 - episodic, undiscounted, as in
        the thesis).
    seed : int
        Master seed for the environment's RNG.
    verbose : bool
        If True, print progress per `(eta, alpha)` cell.
    """

    DEFAULT_ALGORITHMS: tuple[type[LinearTDAlgorithm], ...] = (TDC, GTD2, GTD1)

    def __init__(self,
                 env: Environment,
                 feature_map: FeatureMap,
                 evaluator: RMSPBEEvaluator,
                 algorithms: Sequence[Type[LinearTDAlgorithm]] = DEFAULT_ALGORITHMS,
                 *,
                 alphas: Sequence[float],
                 etas: Sequence[float],
                 runs: int,
                 episodes: int,
                 gamma: float = 1.0,
                 seed: int = 0,
                 verbose: bool = True) -> None:
        self.env = env
        self.feature_map = feature_map
        self.evaluator = evaluator
        self.algorithms = list(algorithms)
        self.alphas = np.asarray(alphas, dtype=float)
        self.etas = np.asarray(etas, dtype=float)
        self.runs = int(runs)
        self.episodes = int(episodes)
        self.gamma = float(gamma)
        self.seed = int(seed)
        self.verbose = bool(verbose)

    # -- main entry point --------------------------------------------------- #

    def run(self) -> SweepResult:
        env = self.env
        # Reseed the env's RNG once at the start of the sweep so that the
        # entire (eta, alpha, run, episode, step) sequence is deterministic.
        env.rng = np.random.default_rng(self.seed)

        L_eta = len(self.etas)
        L_alpha = len(self.alphas)
        E = self.episodes

        rmspbe = {cls.__name__: np.zeros((L_eta, L_alpha, E))
                  for cls in self.algorithms}

        # Build one agent per algorithm class once and reuse them across
        # the sweep; we just re-zero theta and w via `agent.reset()` each
        # run, and update step sizes via `agent.set_step_sizes(...)`.
        agents = [
            LinearTDAgent(cls(self.feature_map.feature_dim, gamma=self.gamma),
                          self.feature_map, alpha=0.0, beta=0.0)
            for cls in self.algorithms
        ]

        t0 = time.time()
        for i, eta in enumerate(self.etas):
            for j, alpha in enumerate(self.alphas):
                beta = alpha * eta
                for agent in agents:
                    agent.set_step_sizes(alpha, beta)

                for _ in range(self.runs):
                    for agent in agents:
                        agent.reset()

                    for ep in range(E):
                        obs = env.reset()
                        while not env.done:
                            next_obs, reward, done, _ = env.step()
                            for agent in agents:
                                agent.update(obs, None, reward, next_obs, done)
                            obs = next_obs

                        for agent in agents:
                            rmspbe[agent.algorithm.__class__.__name__][i, j, ep] += \
                                self.evaluator(agent.theta)

                if self.verbose:
                    elapsed = time.time() - t0
                    print(f"  eta={eta:>7.4f}  alpha={alpha:>7.4f}  "
                          f"(elapsed {elapsed:6.1f}s)")

        # Average over runs.
        for key in rmspbe:
            rmspbe[key] /= self.runs

        return SweepResult(
            rmspbe=rmspbe,
            alphas=self.alphas,
            etas=self.etas,
            runs=self.runs,
            episodes=self.episodes,
            elapsed_seconds=time.time() - t0,
        )


__all__ = ["Trainer", "SweepResult", "ParameterSweep"]
