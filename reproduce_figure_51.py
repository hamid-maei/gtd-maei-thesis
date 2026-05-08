"""Regenerate thesis figure 5.1 **numerics and plots** at full scale.

The curves in the thesis are produced from ``data/*/results.npz`` after
running the four sweeps at **100 runs** and **200 / 400 / 500 / 100**
episodes (tabular / dependent / inverted / Boyan). Opening files under
``figures/`` alone does not recreate those numbers—you must rebuild
``data/`` first.

This script is a thin wrapper around ``run_all.py`` without ``--quick``.
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    run_all = os.path.join(here, "run_all.py")
    env = {**os.environ, "PYTHONPATH": here}
    print(
        "Regenerating thesis figure 5.1 data + plots (full scale; ~10 min).\n",
        flush=True,
    )
    subprocess.run([sys.executable, run_all], check=True, cwd=here, env=env)


if __name__ == "__main__":
    main()
