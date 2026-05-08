"""Reproduce Maei (2011) Fig 5.1, bottom-right panel (Boyan chain).

14-state Boyan chain with 4 overlapping triangle features; Boyan adds
one extra (smaller) eta value to the random-walk grid as documented in
the thesis.
"""

from __future__ import annotations

import argparse
import os

from figure_51_config import THESIS_EPISODES, THESIS_RUNS
from algorithms import GTD1, GTD2, TDC
from environments import BoyanChainEnv
from evaluation import RMSPBEEvaluator
from features import BoyanTriangleFeatures
from training import ParameterSweep


# Thesis Fig 5.1 Boyan grid (one extra power of two below the random
# walk eta range, plus a wider alpha grid up to 2.0).
ETA_VEC = [0.0, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0]
ALPHA_VEC = [0.5 / 32, 0.5 / 16, 0.5 / 8, 0.5 / 4, 0.5 / 2, 0.5, 1.0, 2.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    if args.quick and args.paper:
        raise SystemExit("Pass either --quick or --paper, not both.")

    runs = args.runs if args.runs is not None else (5 if args.quick else THESIS_RUNS)
    episodes = args.episodes if args.episodes is not None else (
        20 if args.quick else THESIS_EPISODES["boyan"])

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out_dir or os.path.join(here, "data", "boyan")
    os.makedirs(out_dir, exist_ok=True)

    env = BoyanChainEnv(n_states=14, seed=args.seed)
    feature_map = BoyanTriangleFeatures(n_states=14, feature_dim=4)
    evaluator = RMSPBEEvaluator(env, feature_map, gamma=1.0)

    print(f"== boyan_exp ==  runs={runs}  episodes={episodes}")
    sweep = ParameterSweep(
        env=env,
        feature_map=feature_map,
        evaluator=evaluator,
        algorithms=(TDC, GTD2, GTD1),
        alphas=ALPHA_VEC,
        etas=ETA_VEC,
        runs=runs,
        episodes=episodes,
        gamma=1.0,
        seed=args.seed,
    )
    result = sweep.run()
    out_path = os.path.join(out_dir, "results.npz")
    result.save(out_path)
    print(f"Saved: {out_path}  (took {result.elapsed_seconds:.1f}s)")


if __name__ == "__main__":
    main()
