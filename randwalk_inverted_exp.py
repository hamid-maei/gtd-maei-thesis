"""Reproduce Maei (2011) Fig 5.1, top-right panel (Random Walk - Inverted).

Five non-terminal states; "inverted" features (1/2 in every coordinate
except a 0 in the column matching the state) chosen to cause
pathological generalisation between consecutive states.
"""

from __future__ import annotations

import argparse
import os

from figure_51_config import THESIS_EPISODES, THESIS_RUNS
from algorithms import GTD1, GTD2, TDC
from environments import RandomWalkEnv
from evaluation import RMSPBEEvaluator
from features import InvertedFeatures
from training import ParameterSweep


ETA_VEC = [0.0, 1 / 4, 1 / 2, 1.0, 2.0]
ALPHA_VEC = [0.5 / 16, 0.5 / 8, 0.5 / 4, 0.5 / 2, 0.5]


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
        20 if args.quick else THESIS_EPISODES["randwalk_invF"])

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out_dir or os.path.join(here, "data", "randwalk_invF")
    os.makedirs(out_dir, exist_ok=True)

    env = RandomWalkEnv(n_states=5, seed=args.seed)
    feature_map = InvertedFeatures(n_states=5)
    evaluator = RMSPBEEvaluator(env, feature_map, gamma=1.0)

    print(f"== randwalk_inverted_exp ==  runs={runs}  episodes={episodes}")
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
