"""Numerical-equivalence tests against the reference implementation.

For each of the four problems, we run a small sweep with the new
ParameterSweep and a matching sweep against the legacy scripts in
`archive/Fast_Gradient_TD_Codes_ICML09paper/python_codes/`, then assert that every
RMSPBE entry agrees within `atol=1e-12`.

Bit-identical results require:

  1. Same RNG seed.
  2. Same `(eta, alpha, run, episode)` traversal order.
  3. Same algorithm-update order at each step (TDC -> GTD2 -> GTD1).
  4. Same `phi' = 0` convention on terminal transitions.

All four are guaranteed by `training.ParameterSweep`.

The reference is invoked in a subprocess so its `algorithms.py` module
does not collide with the new package's `algorithms.py`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest

from algorithms import GTD1, GTD2, TDC
from environments import BoyanChainEnv, RandomWalkEnv
from evaluation import RMSPBEEvaluator
from features import (
    BoyanTriangleFeatures,
    DependentFeatures,
    InvertedFeatures,
    TabularFeatures,
)
from training import ParameterSweep


# Reference lives next to this package under gtd-algos/archive/...
_REFERENCE_ROOT = Path(__file__).resolve().parents[2]  # repo root (gtd-algos)
REFERENCE_DIR = (
    _REFERENCE_ROOT
    / "archive"
    / "Fast_Gradient_TD_Codes_ICML09paper"
    / "python_codes"
)


# Sweep grids: small enough to keep the test fast (<10 s).
SMALL_RW_ETAS  = [0.0, 1 / 4, 1.0]
SMALL_RW_ALPHAS = [0.5 / 8, 0.5 / 4, 0.5 / 2]
SMALL_BOYAN_ETAS = [0.0, 1 / 8, 1 / 2]
SMALL_BOYAN_ALPHAS = [0.5 / 4, 0.5 / 2, 0.5]
RUNS = 2
EPISODES = 20
SEED = 0
ATOL = 1e-12


# --------------------------------------------------------------------------- #
# Reference subprocess                                                         #
# --------------------------------------------------------------------------- #

_RUN_TEMPLATE = textwrap.dedent("""
    import json, sys, numpy as np
    sys.path.insert(0, {ref_dir!r})
    from {module} import run

    kwargs = json.loads({kwargs_json!r})
    out = run(verbose=False, **kwargs)
    np.savez({out_path!r}, **out)
""").lstrip()


def _run_reference(module: str, kwargs: dict) -> dict[str, np.ndarray]:
    """Invoke python_codes/<module>.py::run in a subprocess and load the
    resulting RMSPBE arrays.
    """
    if not REFERENCE_DIR.exists():
        pytest.skip(f"Reference folder not present: {REFERENCE_DIR}")

    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "ref.npz")
        script = _RUN_TEMPLATE.format(
            ref_dir=str(REFERENCE_DIR),
            module=module,
            kwargs_json=json.dumps(kwargs),
            out_path=out_path,
        )
        # Strip the new package from PYTHONPATH so the subprocess only
        # imports the reference modules.
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        subprocess.run([sys.executable, "-c", script],
                       check=True, env=env, cwd=tmp)
        with np.load(out_path) as npz:
            return {k: npz[k] for k in npz.files}


def _sweep_new(env, feature_map, alphas, etas) -> dict[str, np.ndarray]:
    evaluator = RMSPBEEvaluator(env, feature_map, gamma=1.0)
    sweep = ParameterSweep(
        env=env,
        feature_map=feature_map,
        evaluator=evaluator,
        algorithms=(TDC, GTD2, GTD1),
        alphas=alphas,
        etas=etas,
        runs=RUNS,
        episodes=EPISODES,
        gamma=1.0,
        seed=SEED,
        verbose=False,
    )
    return sweep.run().rmspbe


def _assert_close(new: dict, ref: dict) -> None:
    """Assert RMSPBE_TDC / RMSPBE_GTD2 / RMSPBE_GTD1 agree."""
    for new_key, ref_key in [("TDC", "RMSPBE_TDC"),
                              ("GTD2", "RMSPBE_GTD2"),
                              ("GTD1", "RMSPBE_GTD1")]:
        n = new[new_key]
        r = ref[ref_key]
        assert n.shape == r.shape, (new_key, n.shape, r.shape)
        diff = float(np.abs(n - r).max())
        assert np.allclose(n, r, atol=ATOL), (
            f"{new_key} differs: max abs diff = {diff:.3e}"
        )


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #

def test_random_walk_tabular_matches_reference():
    ref = _run_reference("randwalk_tab_exp", dict(
        runs=RUNS, n_episodes=EPISODES,
        step_size_ratio_vec=SMALL_RW_ETAS,
        alpha_step_size_vec=SMALL_RW_ALPHAS,
        n_features=5, gamma=1.0, seed=SEED,
    ))
    new = _sweep_new(
        env=RandomWalkEnv(n_states=5, seed=SEED),
        feature_map=TabularFeatures(n_states=5),
        alphas=SMALL_RW_ALPHAS, etas=SMALL_RW_ETAS,
    )
    _assert_close(new, ref)


def test_random_walk_dependent_matches_reference():
    ref = _run_reference("randwalk_fa_exp", dict(
        runs=RUNS, n_episodes=EPISODES,
        step_size_ratio_vec=SMALL_RW_ETAS,
        alpha_step_size_vec=SMALL_RW_ALPHAS,
        gamma=1.0, seed=SEED,
    ))
    new = _sweep_new(
        env=RandomWalkEnv(n_states=5, seed=SEED),
        feature_map=DependentFeatures(),
        alphas=SMALL_RW_ALPHAS, etas=SMALL_RW_ETAS,
    )
    _assert_close(new, ref)


def test_random_walk_inverted_matches_reference():
    ref = _run_reference("randwalk_invF_exp", dict(
        runs=RUNS, n_episodes=EPISODES,
        step_size_ratio_vec=SMALL_RW_ETAS,
        alpha_step_size_vec=SMALL_RW_ALPHAS,
        n_features=5, gamma=1.0, seed=SEED,
    ))
    new = _sweep_new(
        env=RandomWalkEnv(n_states=5, seed=SEED),
        feature_map=InvertedFeatures(n_states=5),
        alphas=SMALL_RW_ALPHAS, etas=SMALL_RW_ETAS,
    )
    _assert_close(new, ref)


def test_boyan_matches_reference():
    ref = _run_reference("boyan_exp", dict(
        runs=RUNS, n_episodes=EPISODES,
        step_size_ratio_vec=SMALL_BOYAN_ETAS,
        alpha_step_size_vec=SMALL_BOYAN_ALPHAS,
        n_features=4, n_states=14, gamma=1.0, seed=SEED,
    ))
    new = _sweep_new(
        env=BoyanChainEnv(n_states=14, seed=SEED),
        feature_map=BoyanTriangleFeatures(n_states=14, feature_dim=4),
        alphas=SMALL_BOYAN_ALPHAS, etas=SMALL_BOYAN_ETAS,
    )
    _assert_close(new, ref)
