"""Smoke tests for environments.RandomWalkEnv and BoyanChainEnv."""

from __future__ import annotations

import numpy as np
import pytest

from environments import BoyanChainEnv, RandomWalkEnv


# --------------------------------------------------------------------------- #
# RandomWalkEnv                                                                #
# --------------------------------------------------------------------------- #

class TestRandomWalkEnv:
    def test_state_count_and_initial_state(self):
        env = RandomWalkEnv(n_states=5, seed=0)
        assert env.n_states == 7
        assert list(env.non_terminal_states()) == [1, 2, 3, 4, 5]
        assert env.initial_state() == 3   # middle non-terminal of 1..5

    def test_terminal_predicate(self):
        env = RandomWalkEnv(n_states=5, seed=0)
        assert env.is_terminal(0)
        assert env.is_terminal(env.n_states - 1)
        for s in range(1, env.n_states - 1):
            assert not env.is_terminal(s)

    def test_transition_matrix_is_row_stochastic(self):
        env = RandomWalkEnv(n_states=5, seed=0)
        P = env.transition_matrix()
        assert P.shape == (7, 7)
        assert np.allclose(P.sum(axis=1), 1.0)

    def test_reward_only_on_right_terminal(self):
        env = RandomWalkEnv(n_states=5, seed=0)
        r_bar = env.reward_vector()
        assert np.allclose(r_bar[:-2], 0.0)        # no reward in the interior
        assert r_bar[-2] == pytest.approx(0.5)     # P=0.5 to right terminal w/ r=1
        assert r_bar[-1] == 0.0                    # absorbing

    def test_episodes_terminate(self):
        env = RandomWalkEnv(n_states=5, seed=0)
        for _ in range(5):
            obs = env.reset()
            steps = 0
            while not env.done:
                obs, reward, done, _ = env.step()
                steps += 1
                assert steps < 1000, "episode failed to terminate"

    def test_step_after_done_raises(self):
        env = RandomWalkEnv(n_states=5, seed=0)
        env.reset()
        while not env.done:
            env.step()
        with pytest.raises(RuntimeError):
            env.step()


# --------------------------------------------------------------------------- #
# BoyanChainEnv                                                                #
# --------------------------------------------------------------------------- #

class TestBoyanChainEnv:
    def test_state_count_and_initial_state(self):
        env = BoyanChainEnv(n_states=14, seed=0)
        assert env.n_states == 14
        assert list(env.non_terminal_states()) == list(range(1, 14))
        assert env.initial_state() == 13

    def test_transition_matrix_layout(self):
        env = BoyanChainEnv(n_states=14, seed=0)
        P = env.transition_matrix()
        assert P[0, 0] == 1.0          # terminal absorbing
        assert P[1, 0] == 1.0          # state 1 -> terminal
        assert P[2, 1] == 1.0          # state 2 -> 1
        for s in range(3, 14):
            assert P[s, s - 1] == 0.5
            assert P[s, s - 2] == 0.5
        assert np.allclose(P.sum(axis=1), 1.0)

    def test_reward_layout(self):
        env = BoyanChainEnv(n_states=14, seed=0)
        r_bar = env.reward_vector()
        assert r_bar[0] == 0.0
        assert r_bar[1] == 0.0
        assert r_bar[2] == pytest.approx(-2.0)
        for s in range(3, 14):
            assert r_bar[s] == pytest.approx(-3.0)

    def test_full_episode_terminates(self):
        env = BoyanChainEnv(n_states=14, seed=0)
        env.reset()
        steps = 0
        while not env.done:
            env.step()
            steps += 1
            assert steps < 1000
