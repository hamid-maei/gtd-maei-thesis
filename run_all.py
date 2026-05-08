"""Run all four ICML'09 / Maei (2011) Fig 5.1 experiments end-to-end.

**Thesis numerics:** By default (no ``--quick``), each experiment uses
100 runs and the episode counts required for thesis figure 5.1 (see
``figure_51_config.py``). ``figures/*.png`` are only consistent with the
thesis after ``data/*/results.npz`` has been regenerated at that scale.

Usage:
    python reproduce_figure_51.py
    python run_all.py          # runs each *_exp.py with explicit --runs 100
                               # and thesis --episodes (200/400/500/100)
    python run_all.py --quick  # 5×20 smoke test only

Optional: ``--reference-plots`` uses archive ``plot_all_*.py`` for per-env figures.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from figure_51_config import THESIS_EPISODES, THESIS_RUNS


HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(HERE)
_ARCHIVE_PY = os.path.join(
    _REPO_ROOT,
    "archive",
    "Fast_Gradient_TD_Codes_ICML09paper",
    "python_codes",
)

# (data subfolder under gtd_phd_maei/data/, archive plot_all_*.py name).
_REFERENCE_PLOT_SCRIPTS: list[tuple[str, str]] = [
    ("randwalk_tab", "plot_all_randwalk_tab.py"),
    ("randwalk_fa", "plot_all_randwalk_fa.py"),
    ("randwalk_invF", "plot_all_randwalk_invF.py"),
    ("boyan", "plot_all_boyan.py"),
]


# (experiment_script, plot_script). The plot scripts are run after the
# corresponding experiments, then plot_thesis_figure_51.py renders the
# combined 2x4 figure once at the end.
EXPERIMENTS = [
    ("randwalk_tabular_exp.py",   "plot_randwalk_tab.py"),
    ("randwalk_dependent_exp.py", "plot_randwalk_fa.py"),
    ("randwalk_inverted_exp.py",  "plot_randwalk_invF.py"),
    ("boyan_exp.py",              "plot_boyan.py"),
]

# Map each experiment script to ``figure_51_config.THESIS_EPISODES`` key.
_EXP_ENV_KEY: dict[str, str] = {
    "randwalk_tabular_exp.py": "randwalk_tab",
    "randwalk_dependent_exp.py": "randwalk_fa",
    "randwalk_inverted_exp.py": "randwalk_invF",
    "boyan_exp.py": "boyan",
}


def _python(script: str, *flags: str) -> None:
    cmd = [sys.executable, os.path.join(HERE, script), *flags]
    print("\n$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=HERE)
    print(f"-- finished {script} in {time.time() - t0:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="Smoke only: 5 runs and 20 episodes per problem "
                             "(does NOT match thesis Fig 5.1).")
    parser.add_argument("--paper", action="store_true",
                        help="Ignored (kept for compatibility): full runs "
                             "already use thesis scale via explicit "
                             "--runs/--episodes unless --quick.")
    parser.add_argument("--skip-experiments", action="store_true",
                        help="Only re-render plots from existing data.")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Only run experiments, skip plotting.")
    parser.add_argument("--reference-plots", action="store_true",
                        help="Render per-env figures via archive "
                             "python_codes/plot_all_*.py (identical to "
                             "pre-refactor stack); thesis via this folder's "
                             "plot_thesis_figure_51.py.")
    args = parser.parse_args()

    if args.quick and args.paper:
        raise SystemExit("Pass either --quick or --paper, not both.")

    if not args.skip_experiments:
        if args.quick:
            print(
                "\n=== QUICK smoke: 5 runs x 20 episodes — NOT thesis Fig 5.1 ===\n",
                flush=True,
            )
        else:
            print(
                "\n=== Thesis Fig 5.1: each experiment is invoked as "
                f"--runs {THESIS_RUNS} --episodes <...> with episodes "
                f"{THESIS_EPISODES['randwalk_tab']} / {THESIS_EPISODES['randwalk_fa']} / "
                f"{THESIS_EPISODES['randwalk_invF']} / {THESIS_EPISODES['boyan']} "
                "(tabular / dep / inv / boyan) ===\n",
                flush=True,
            )

    _PLOT_TO_SUBFOLDER = {
        "plot_randwalk_tab.py": "randwalk_tab",
        "plot_randwalk_fa.py": "randwalk_fa",
        "plot_randwalk_invF.py": "randwalk_invF",
        "plot_boyan.py": "boyan",
    }

    for exp_script, plot_script in EXPERIMENTS:
        if not args.skip_experiments:
            if args.quick:
                exp_flags: tuple[str, ...] = ("--quick",)
            else:
                key = _EXP_ENV_KEY[exp_script]
                exp_flags = (
                    "--runs",
                    str(THESIS_RUNS),
                    "--episodes",
                    str(THESIS_EPISODES[key]),
                )
            _python(exp_script, *exp_flags)
        if args.skip_plots:
            continue
        if args.reference_plots:
            if not os.path.isdir(_ARCHIVE_PY):
                raise SystemExit(
                    f"Archive plot scripts not found: {_ARCHIVE_PY}")
            pair = next(
                (s, sc) for s, sc in _REFERENCE_PLOT_SCRIPTS
                if s == _PLOT_TO_SUBFOLDER[plot_script])
            cmd = [
                sys.executable,
                os.path.join(_ARCHIVE_PY, pair[1]),
                "--data-dir",
                os.path.join(HERE, "data", pair[0]),
                "--fig-dir",
                os.path.join(HERE, "figures", pair[0]),
            ]
            print("\n$ " + " ".join(cmd))
            subprocess.run(cmd, check=True)
        else:
            _python(plot_script)

    if not args.skip_plots:
        if args.reference_plots:
            thesis_cmd = [
                sys.executable,
                os.path.join(HERE, "plot_thesis_figure_51.py"),
                "--package-root",
                HERE,
                "--fig-dir",
                os.path.join(HERE, "figures"),
            ]
            print("\n$ " + " ".join(thesis_cmd))
            subprocess.run(thesis_cmd, check=True, cwd=HERE)
        else:
            _python("plot_thesis_figure_51.py")

    print("\nAll done. Figures live in:", os.path.join(HERE, "figures"))


if __name__ == "__main__":
    main()
