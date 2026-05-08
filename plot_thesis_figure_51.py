"""Reproduce Figure 5.1 of Maei (2011) PhD thesis as a single 2x2 panel figure.

Each panel contains a left sub-panel (parameter study, RMSPBE vs alpha, log-x)
and a right sub-panel (best learning curve, RMSPBE vs episodes). Curves are
labelled in-figure (no legend box), matching the thesis style.

Reads pre-computed RMSPBE arrays from data/<env>/results.npz produced by the
four *_exp.py scripts; does not run any new simulations.
"""

from __future__ import annotations

import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from common import best_i_j
from figure_51_config import THESIS_EPISODES, warn_short_episodes


HERE = os.path.dirname(os.path.abspath(__file__))

# Mutable so `main()` can point loads at another package root (e.g. when
# orchestrated by run_all.py --reference-plots).
_PACKAGE_ROOT: list[str] = [HERE]


# Colours used on the panels - match plot_common.py / thesis style
TD_COLOR   = "red"
TDC_COLOR  = "blue"
GTD2_COLOR = "green"
GTD1_COLOR = "black"

# Per-eta marker shape (matches MATLAB plot_all_*.m)
_ETA_MARKER = {
    0.0:    "",
    1 / 16: "*",
    1 / 8:  "v",
    1 / 4:  "<",
    1 / 2:  "o",
    1.0:    "x",
    2.0:    "+",
    4.0:    "d",
}


def _marker_for_eta(eta: float) -> str:
    for key, m in _ETA_MARKER.items():
        if abs(eta - key) < 1e-9:
            return m
    return "."


def _load(env: str) -> dict:
    root = _PACKAGE_ROOT[0]
    npz_path = os.path.join(root, "data", env, "results.npz")
    return dict(np.load(npz_path))


def _draw_panel(ax_left, ax_right, env_data: dict, *, title: str,
                epis_b: int, x_ticks: Iterable[float],
                y_max: float, episodes_max: int,
                show_xlabels: bool = True, show_ylabel_left: bool = True):
    """Render the (param-study | learning-curves) pair for one problem.

    Adds in-figure text labels next to each algorithm curve (thesis style).
    """
    rmspbe_tdc  = env_data["RMSPBE_TDC"]
    rmspbe_gtd2 = env_data["RMSPBE_GTD2"]
    rmspbe_gtd1 = env_data["RMSPBE_GTD1"]
    alpha_vec   = env_data["alphaStepSizeVec"]
    eta_vec     = env_data["stepSizeRatioVec"]

    L_eta, L_alpha, T = rmspbe_tdc.shape
    epis_b = min(epis_b, T)

    avg_tdc  = rmspbe_tdc[:, :, :epis_b].mean(axis=2)
    avg_gtd2 = rmspbe_gtd2[:, :, :epis_b].mean(axis=2)
    avg_gtd1 = rmspbe_gtd1[:, :, :epis_b].mean(axis=2)

    # eta = 0 row  -> TD(0); eta > 0 rows used for TDC / GTD2 / GTD
    td0_curve = avg_tdc[0, :].copy()
    tdc_grid  = avg_tdc[1:, :].copy()
    gtd2_grid = avg_gtd2[1:, :].copy()
    gtd1_grid = avg_gtd1[1:, :].copy()

    # ---- Left sub-panel: parameter study --------------------------------- #
    ax = ax_left
    m0 = _marker_for_eta(eta_vec[0]) or None
    ax.plot(alpha_vec, td0_curve,
            color=TD_COLOR, linewidth=4, marker=m0, label="TD")
    for i, eta in enumerate(eta_vec):
        if eta == 0.0:
            continue
        m = _marker_for_eta(eta) or None
        ax.plot(alpha_vec, avg_tdc[i],  color=TDC_COLOR, marker=m, linewidth=1.5)
        ax.plot(alpha_vec, avg_gtd2[i], color=GTD2_COLOR, marker=m, linewidth=1.5)
        ax.plot(alpha_vec, avg_gtd1[i], color=GTD1_COLOR, marker=m, linewidth=1.5)

    ax.set_xscale("log")
    ax.set_xlim(min(x_ticks), max(x_ticks))
    ax.set_ylim(0.0, y_max)
    ax.set_xticks(list(x_ticks))
    ax.set_xticklabels([f"{t:g}" for t in x_ticks])
    if show_ylabel_left:
        ax.set_ylabel("RMSPBE", fontsize=12)
    if show_xlabels:
        ax.set_xlabel(r"$\alpha$", fontsize=12)
    ax.set_title(title, fontsize=12, loc="center")

    # ---- In-figure labels: place near best column for each algorithm ----- #
    bj_td  = int(np.argmin(td0_curve))
    bi_t,  bj_t  = best_i_j(tdc_grid)
    bi_g2, bj_g2 = best_i_j(gtd2_grid)
    bi_g1, bj_g1 = best_i_j(gtd1_grid)

    def _annotate(x, y, txt, color):
        ax.text(x, y, txt, color=color, fontsize=10,
                ha="center", va="bottom",
                bbox=dict(facecolor="white", edgecolor="none",
                          pad=0.5, alpha=0.8))

    _annotate(alpha_vec[bj_td],  td0_curve[bj_td],            "TD",   TD_COLOR)
    _annotate(alpha_vec[bj_t],   avg_tdc[bi_t + 1, bj_t],     "TDC",  TDC_COLOR)
    _annotate(alpha_vec[bj_g2],  avg_gtd2[bi_g2 + 1, bj_g2],  "GTD2", GTD2_COLOR)
    _annotate(alpha_vec[bj_g1],  avg_gtd1[bi_g1 + 1, bj_g1],  "GTD",  GTD1_COLOR)

    ax.grid(True, which="both", alpha=0.25)

    # ---- Right sub-panel: best learning curves --------------------------- #
    bx = ax_right
    x = np.arange(1, epis_b + 1)
    best_td0  = rmspbe_tdc [0,           bj_td,  :epis_b]
    best_tdc  = rmspbe_tdc [bi_t  + 1,   bj_t,   :epis_b]
    best_gtd2 = rmspbe_gtd2[bi_g2 + 1,   bj_g2,  :epis_b]
    best_gtd1 = rmspbe_gtd1[bi_g1 + 1,   bj_g1,  :epis_b]

    bx.plot(x, best_gtd1, color=GTD1_COLOR, linewidth=2)
    bx.plot(x, best_gtd2, color=GTD2_COLOR, linewidth=2)
    bx.plot(x, best_tdc,  color=TDC_COLOR,  linewidth=2)
    bx.plot(x, best_td0,  color=TD_COLOR,   linewidth=3)

    bx.set_xlim(0, epis_b)
    bx.set_ylim(0.0, y_max)
    if show_xlabels:
        bx.set_xlabel("episodes", fontsize=12)
    bx.grid(True, alpha=0.25)
    bx.set_yticklabels([])  # share y with left panel visually

    # On-curve labels for the learning-curve sub-panel
    last = epis_b - 1
    bx.text(x[last] * 0.97, best_gtd1[last], "GTD",
            color=GTD1_COLOR, fontsize=10, ha="right", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.8))
    bx.text(x[last] * 0.97, best_gtd2[last], "GTD2",
            color=GTD2_COLOR, fontsize=10, ha="right", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.8))
    bx.text(x[last] * 0.97, best_tdc[last],  "TDC",
            color=TDC_COLOR, fontsize=10, ha="right", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.8))
    bx.text(x[last] * 0.97, best_td0[last],  "TD",
            color=TD_COLOR, fontsize=10, ha="right", va="top",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.8))


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-root",
        default=None,
        metavar="DIR",
        help="Directory that contains data/<env>/results.npz (default: "
             "this package folder).",
    )
    parser.add_argument(
        "--fig-dir",
        default=None,
        metavar="DIR",
        help="Where to write thesis_figure_5_1.png/pdf (default: "
             "<package-root>/figures).",
    )
    args = parser.parse_args()
    root = os.path.abspath(args.package_root or HERE)
    _PACKAGE_ROOT[0] = root
    fig_dir = os.path.abspath(args.fig_dir or os.path.join(root, "figures"))

    for key in ("randwalk_tab", "randwalk_invF", "randwalk_fa", "boyan"):
        p = os.path.join(root, "data", key, "results.npz")
        with np.load(p) as z:
            t_axis = int(z["RMSPBE_TDC"].shape[2])
        warn_short_episodes(key, t_axis)

    fig, axes = plt.subplots(2, 4, figsize=(16, 9),
                             gridspec_kw={"width_ratios": [1.05, 0.95, 1.05, 0.95]})

    et = THESIS_EPISODES
    # ---- Top-left: random walk - tabular --------------------------------- #
    _draw_panel(axes[0, 0], axes[0, 1], _load("randwalk_tab"),
                title="Random Walk - Tabular features",
                epis_b=et["randwalk_tab"],
                x_ticks=[0.03, 0.06, 0.12, 0.25, 0.5],
                y_max=0.16, episodes_max=et["randwalk_tab"],
                show_xlabels=False, show_ylabel_left=True)

    # ---- Top-right: random walk - inverted features --------------------- #
    _draw_panel(axes[0, 2], axes[0, 3], _load("randwalk_invF"),
                title="Random Walk - Inverted features",
                epis_b=et["randwalk_invF"],
                x_ticks=[0.03, 0.06, 0.12, 0.25, 0.5],
                y_max=0.20, episodes_max=et["randwalk_invF"],
                show_xlabels=False, show_ylabel_left=False)

    # ---- Bottom-left: random walk - dependent features ------------------ #
    _draw_panel(axes[1, 0], axes[1, 1], _load("randwalk_fa"),
                title="Random Walk - Dependent features",
                epis_b=et["randwalk_fa"],
                x_ticks=[0.008, 0.015, 0.03, 0.06, 0.12, 0.25, 0.5],
                y_max=0.14, episodes_max=et["randwalk_fa"],
                show_xlabels=True, show_ylabel_left=True)

    # ---- Bottom-right: Boyan chain --------------------------------------- #
    _draw_panel(axes[1, 2], axes[1, 3], _load("boyan"),
                title="Boyan Chain",
                epis_b=et["boyan"],
                x_ticks=[0.015, 0.03, 0.06, 0.12, 0.25, 0.5, 1.0, 2.0],
                y_max=2.8, episodes_max=et["boyan"],
                show_xlabels=True, show_ylabel_left=False)

    fig.suptitle(
        "Figure 5.1 - Empirical results on the four small problems "
        "(Maei, 2011, PhD thesis). Reproduction.",
        fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_dir = fig_dir
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "thesis_figure_5_1.png")
    out_pdf = os.path.join(out_dir, "thesis_figure_5_1.pdf")
    fig.savefig(out_png, dpi=160)
    fig.savefig(out_pdf)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
