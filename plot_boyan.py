"""Plot Boyan chain (Fig 5.1 bottom-right)."""

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
    data_dir = args.data_dir or os.path.join(here, "data", "boyan")
    fig_dir = args.fig_dir or os.path.join(here, "figures", "boyan")
    npz_path = os.path.join(data_dir, "results.npz")

    epis_b, T = epis_b_from_npz(npz_path, "boyan", args.epis_b)
    warn_short_episodes("boyan", T)

    out = plot_param_study_and_best(
        npz_path=npz_path,
        fig_dir=fig_dir,
        exp_name="boyan",
        epis_b=epis_b,
        x_lim=(0.015, 2.0),
        y_lim=(0.0, 2.8),
        y_lim_curves=(0.0, 2.8),
        x_ticks=[0.015, 0.03, 0.06, 0.12, 0.25, 0.5, 1.0, 2.0],
        show=args.show,
    )
    print(f"Saved {out['param_study_path']}")
    print(f"Saved {out['best_curves_path']}")


if __name__ == "__main__":
    main()
