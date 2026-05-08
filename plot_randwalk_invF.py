"""Plot Random Walk - Inverted features (Fig 5.1 top-right)."""

from __future__ import annotations

import argparse
import os

from figure_51_config import epis_b_from_npz, warn_short_episodes
from plot_common import plot_param_study_and_best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--fig-dir", default=None)
    parser.add_argument(
        "--epis-b",
        type=int,
        default=None,
        metavar="N",
        help="Episodes for averaging / best curves (default: min(thesis, T)).",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or os.path.join(here, "data", "randwalk_invF")
    fig_dir = args.fig_dir or os.path.join(here, "figures", "randwalk_invF")
    npz_path = os.path.join(data_dir, "results.npz")

    epis_b, T = epis_b_from_npz(npz_path, "randwalk_invF", args.epis_b)
    warn_short_episodes("randwalk_invF", T)

    out = plot_param_study_and_best(
        npz_path=npz_path,
        fig_dir=fig_dir,
        exp_name="randwalkInvFeat",
        epis_b=epis_b,
        x_lim=(0.03, 0.5),
        y_lim=(0.0, 0.20),
        y_lim_curves=(0.0, 0.20),
        x_ticks=[0.03, 0.06, 0.12, 0.25, 0.5],
        show=args.show,
    )
    print(f"Saved {out['param_study_path']}")
    print(f"Saved {out['best_curves_path']}")


if __name__ == "__main__":
    main()
