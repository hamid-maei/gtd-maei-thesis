"""
Common plotting utilities for the ICML'09 reproductions.

We render two figures per environment, matching Figure 5.1 of Maei (2011)
and Figure 3 of Sutton, Maei et al. ICML'09:
  (a) Average RMSPBE vs alpha (log-x), one line per eta. TD(0) is the eta=0
      row of the TDC sweep, drawn thicker.
  (b) Best learning curves: per-episode RMSPBE for the (eta*, alpha*)
      minimising the panel-(a) average over the first epis_b episodes,
      with one line per algorithm.
"""

from __future__ import annotations

import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from common import best_i_j


# ---- paper colour scheme (Maei, 2011, Figure 5.1) ------------------------- #
TD_COLOR   = "red"
TDC_COLOR  = "blue"
GTD2_COLOR = "green"
GTD1_COLOR = "black"

# ---- per-eta marker map (matches MATLAB plot_all_*.m) --------------------- #
_ETA_MARKER = {
    0.0:    "",          # solid line, no marker (TD(0))
    1 / 16: "*",
    1 / 8:  "v",
    1 / 4:  "<",
    1 / 2:  "o",
    1.0:    "x",
    2.0:    "+",
    4.0:    "d",
}


def _marker_for_eta(eta: float) -> str:
    """Return MATLAB-style marker for an eta value, with a small tolerance."""
    for key, m in _ETA_MARKER.items():
        if abs(eta - key) < 1e-9:
            return m
    return "."


def plot_param_study_and_best(npz_path: str, fig_dir: str, exp_name: str,
                              epis_b: int,
                              x_lim: tuple | None = None,
                              y_lim: tuple | None = None,
                              y_lim_curves: tuple | None = None,
                              x_ticks: Iterable[float] | None = None,
                              show: bool = False) -> dict:
    """Generate the two paper-style figures and dump ASCII artefacts.

    Returns a dict of useful summary arrays.
    """
    data = np.load(npz_path)
    rmspbe_tdc  = data["RMSPBE_TDC"]
    rmspbe_gtd2 = data["RMSPBE_GTD2"]
    rmspbe_gtd1 = data["RMSPBE_GTD1"]
    alpha_vec = data["alphaStepSizeVec"]
    eta_vec   = data["stepSizeRatioVec"]

    L_eta, L_alpha, T = rmspbe_tdc.shape
    epis_b = min(epis_b, T)

    # Average RMSPBE over the first epis_b episodes
    avg_tdc  = rmspbe_tdc[:, :, :epis_b].mean(axis=2)   # shape (L_eta, L_alpha)
    avg_gtd2 = rmspbe_gtd2[:, :, :epis_b].mean(axis=2)
    avg_gtd1 = rmspbe_gtd1[:, :, :epis_b].mean(axis=2)

    # eta=0 row (TDC w/ beta=0) is plotted as TD(0)
    param_study_td0  = avg_tdc[0, :].copy()
    param_study_tdc  = avg_tdc[1:, :].copy()
    param_study_gtd2 = avg_gtd2[1:, :].copy()
    param_study_gtd1 = avg_gtd1[1:, :].copy()

    os.makedirs(fig_dir, exist_ok=True)

    # ----------------------- Panel (a): parameter study ----------------------
    fig_ps, ax_ps = plt.subplots(figsize=(7.5, 6))

    # Plot TD(0) once
    m0 = _marker_for_eta(eta_vec[0])
    ax_ps.plot(alpha_vec, param_study_td0,
               color=TD_COLOR, linestyle="-",
               marker=(m0 or None),
               linewidth=6,
               label="TD")

    # Plot TDC, GTD2, GTD1 for each non-zero eta
    handles_added = {"TDC": False, "GTD2": False, "GTD": False}
    for i, eta in enumerate(eta_vec):
        if eta == 0.0:
            continue
        m = _marker_for_eta(eta) or None
        h_tdc, = ax_ps.plot(alpha_vec, avg_tdc[i, :],
                            color=TDC_COLOR, linestyle="-",
                            marker=m, linewidth=2,
                            label="TDC" if not handles_added["TDC"] else None)
        h_gtd2, = ax_ps.plot(alpha_vec, avg_gtd2[i, :],
                             color=GTD2_COLOR, linestyle="-",
                             marker=m, linewidth=2,
                             label="GTD2" if not handles_added["GTD2"] else None)
        h_gtd1, = ax_ps.plot(alpha_vec, avg_gtd1[i, :],
                             color=GTD1_COLOR, linestyle="-",
                             marker=m, linewidth=2,
                             label="GTD" if not handles_added["GTD"] else None)
        handles_added["TDC"] = handles_added["GTD2"] = handles_added["GTD"] = True

    ax_ps.set_xscale("log")
    ax_ps.set_xlabel(r"$\alpha$", fontsize=14)
    ax_ps.set_ylabel("Average RMSPBE", fontsize=14)
    ax_ps.set_title(f"{exp_name}: average over {epis_b} episodes")
    if x_lim:
        ax_ps.set_xlim(*x_lim)
    if y_lim:
        ax_ps.set_ylim(*y_lim)
    if x_ticks is not None:
        ax_ps.set_xticks(list(x_ticks))
        ax_ps.set_xticklabels([f"{t:g}" for t in x_ticks])
    ax_ps.grid(True, which="both", alpha=0.3)
    ax_ps.legend(loc="best")
    fig_ps.tight_layout()

    ps_path = os.path.join(fig_dir, f"{exp_name}_param_study.png")
    fig_ps.savefig(ps_path, dpi=160)
    fig_ps.savefig(os.path.join(fig_dir, f"{exp_name}_param_study.pdf"))

    # ----------------------- Panel (b): best learning curves -----------------
    bi_tdc,  bj_tdc  = best_i_j(param_study_tdc)
    bi_gtd2, bj_gtd2 = best_i_j(param_study_gtd2)
    bi_gtd1, bj_gtd1 = best_i_j(param_study_gtd1)
    bj_td0 = int(np.argmin(param_study_td0))

    best_tdc  = rmspbe_tdc[bi_tdc + 1,  bj_tdc,  :epis_b]
    best_gtd2 = rmspbe_gtd2[bi_gtd2 + 1, bj_gtd2, :epis_b]
    best_gtd1 = rmspbe_gtd1[bi_gtd1 + 1, bj_gtd1, :epis_b]
    best_td0  = rmspbe_tdc[0, bj_td0, :epis_b]

    fig_bc, ax_bc = plt.subplots(figsize=(7.5, 6))
    x = np.arange(1, epis_b + 1)
    ax_bc.plot(x, best_td0,  color=TD_COLOR,   linewidth=2, label="TD")
    ax_bc.plot(x, best_tdc,  color=TDC_COLOR,  linewidth=2, label="TDC")
    ax_bc.plot(x, best_gtd2, color=GTD2_COLOR, linewidth=2, label="GTD2")
    ax_bc.plot(x, best_gtd1, color=GTD1_COLOR, linewidth=2, label="GTD")
    ax_bc.set_xlabel("Episode", fontsize=14)
    ax_bc.set_ylabel("RMSPBE", fontsize=14)
    ax_bc.set_title(f"{exp_name}: best learning curves")
    if y_lim_curves:
        ax_bc.set_ylim(*y_lim_curves)
    ax_bc.grid(True, alpha=0.3)
    ax_bc.legend(loc="best")
    fig_bc.tight_layout()

    bc_path = os.path.join(fig_dir, f"{exp_name}_best_curves.png")
    fig_bc.savefig(bc_path, dpi=160)
    fig_bc.savefig(os.path.join(fig_dir, f"{exp_name}_best_curves.pdf"))

    # ----------------------- ASCII dumps (match MATLAB) ----------------------
    np.savetxt(os.path.join(fig_dir, f"{exp_name}_study_td0.txt"),  param_study_td0)
    np.savetxt(os.path.join(fig_dir, f"{exp_name}_study_tdc.txt"),  param_study_tdc)
    np.savetxt(os.path.join(fig_dir, f"{exp_name}_study_gtd2.txt"), param_study_gtd2)
    np.savetxt(os.path.join(fig_dir, f"{exp_name}_study_gtd1.txt"), param_study_gtd1)
    np.savetxt(os.path.join(fig_dir, f"{exp_name}_best_td0.txt"),   best_td0)
    np.savetxt(os.path.join(fig_dir, f"{exp_name}_best_tdc.txt"),   best_tdc)
    np.savetxt(os.path.join(fig_dir, f"{exp_name}_best_gtd2.txt"),  best_gtd2)
    np.savetxt(os.path.join(fig_dir, f"{exp_name}_best_gtd1.txt"),  best_gtd1)

    if show:
        plt.show()
    plt.close(fig_ps)
    plt.close(fig_bc)

    return {
        "param_study_td0":  param_study_td0,
        "param_study_tdc":  param_study_tdc,
        "param_study_gtd2": param_study_gtd2,
        "param_study_gtd1": param_study_gtd1,
        "best_td0":  best_td0,
        "best_tdc":  best_tdc,
        "best_gtd2": best_gtd2,
        "best_gtd1": best_gtd1,
        "best_indices": {
            "tdc":  (bi_tdc + 1,  bj_tdc),
            "gtd2": (bi_gtd2 + 1, bj_gtd2),
            "gtd1": (bi_gtd1 + 1, bj_gtd1),
            "td0":  (0, bj_td0),
        },
        "param_study_path": ps_path,
        "best_curves_path": bc_path,
    }


__all__ = ["plot_param_study_and_best", "TD_COLOR", "TDC_COLOR",
           "GTD2_COLOR", "GTD1_COLOR"]
