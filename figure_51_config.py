"""Canonical scale for thesis figure 5.1 (Maei 2011, Sec. 5.4).

These match the sweep sizes documented in the package README and in
`archive/.../python_codes/*_exp.py` for the full (non--quick) runs.
"""

from __future__ import annotations

# Independent runs averaged in each (eta, alpha) cell.
THESIS_RUNS: int = 100

# Episodes per run (third axis of RMSPBE_* arrays); also the horizon
# used in the thesis panels for averaged RMSPBE and best learning curves.
THESIS_EPISODES: dict[str, int] = {
    "randwalk_tab": 200,
    "randwalk_fa": 400,
    "randwalk_invF": 500,
    "boyan": 100,
}

__all__ = ["THESIS_RUNS", "THESIS_EPISODES", "epis_b_from_npz", "warn_short_episodes"]


def epis_b_from_npz(npz_path: str, env_key: str, epis_b_cli: int | None) -> tuple[int, int]:
    """Return ``(epis_b, T)`` for ``plot_param_study_and_best``.

    If ``epis_b_cli`` is None, use ``min(thesis_episodes, T)`` (same as
    clamping inside ``plot_common`` when the CLI matched the thesis).
    """
    import numpy as np

    with np.load(npz_path) as z:
        T = int(z["RMSPBE_TDC"].shape[2])
    want = THESIS_EPISODES[env_key]
    if epis_b_cli is not None:
        return min(epis_b_cli, T), T
    return min(want, T), T


def warn_short_episodes(env_key: str, T: int) -> None:
    """Stderr notice when ``results.npz`` is shorter than the thesis run."""
    import sys

    want = THESIS_EPISODES[env_key]
    if T < want:
        print(
            f"[figure 5.1 / {env_key}] results.npz has T={T} episodes along the "
            f"sweep axis; the thesis uses {want}. Curves will not match the "
            "thesis until you regenerate `data/` at full scale (omit `--quick`; "
            "see `reproduce_figure_51.py` or `run_all.py`).",
            file=sys.stderr,
        )
