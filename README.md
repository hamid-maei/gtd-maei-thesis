# gtd-maei-thesis

Code to reproduce **Figure 5.1** from **Hamid Maei’s 2011 PhD thesis** on gradient temporal-difference learning. The thesis figure summarizes thesis-scale parameter studies for several classic off-policy / linear function approximation problems (random walks and Boyan’s chain), comparing **TD(0)** with **TDC**, **GTD2**, and **GTD1** under the same RMSPBE metric used in the thesis.

The experiments build on the corrected setup described in Sutton, Maei, et al. (ICML 2009); this repository is a self-contained Python project with runnable drivers, plotting scripts, and checks.

## What’s included

- **Experiments** that sweep step sizes and secondary step-size ratios (`eta = beta / alpha`), save tensors to `data/*/results.npz`, and generate per-problem and combined figures under `figures/`.
- **Unit and consistency tests** in `tests/` (run with `pytest`).
- **Checked-in `data/` and `figures/`** so you can inspect thesis-style outputs without rerunning the full sweeps (full runs are still reproducible from the scripts).

## Repository layout

```
gtd-maei-thesis/
  environments.py          Random walk + Boyan environments
  features.py              Tabular / dependent / inverted / Boyan feature maps
  algorithms.py            TD(0), TDC, GTD2, GTD1
  agents.py                Agent: features + algorithm + step sizes
  evaluation.py            RMSPBE via closed-form A, b, C
  training.py              Episode training + parameter sweeps
  figure_51_config.py      Canonical 100 runs, episode counts per problem
  reproduce_figure_51.py   Full-scale reproduction (same as thesis numerics)
  plot_*.py, plot_common.py
  randwalk_*_exp.py, boyan_exp.py
  run_all.py               Driver: experiments + plots
  tests/
  data/                    Saved sweep results (npz)
  figures/                 Plots (png/pdf) including thesis_figure_5_1.*
```

## Quick start

Thesis Figure 5.1 is defined by the **RMSPBE arrays** in `data/*/results.npz` (100 runs; 200 / 400 / 500 / 100 episodes for the four problems). The bundled `figures/` match that full-scale data. If you regenerate with `--quick`, curves will **not** match the thesis.

```bash
cd gtd-maei-thesis

# Full thesis-scale reproduction (writes data/ and figures/)
PYTHONPATH=. python reproduce_figure_51.py
# same as:
PYTHONPATH=. python run_all.py

# Small smoke run only (5×20); for debugging, not for thesis-matching plots
PYTHONPATH=. python run_all.py --quick

# Tests
PYTHONPATH=. python -m pytest tests -q
```

## How the pieces connect

```
Environment  →  observations, rewards, done
Feature map  →  φ(s), φ(s′)
Agent  →  feeds φ, φ′, r and step sizes into the chosen linear TD algorithm
Linear algorithm  →  parameter vector θ (and auxiliary weights for gradient TD)
RMSPBE evaluator  →  scalar error curve over episodes
Parameter sweep  →  grids over α and η, multiple runs, saved arrays
```

Environments return **state indices**; feature maps turn them into **vectors**. Algorithms only see φ, φ′, and rewards, which keeps the experiment code aligned with the usual RL factorization (world vs representation vs update).

## Algorithms (thesis Chapter 5)

| Class  | Update |
| ------ | ------ |
| `TD0`  | `θ += α δ φ` |
| `TDC`  | `θ += α (δ φ − γ φ′ (w·φ));  w += β (δ − w·φ) φ` |
| `GTD2` | `θ += α (φ − γ φ′) (w·φ);       w += β (δ − w·φ) φ` |
| `GTD1` | `θ += α (φ − γ φ′) (w·φ);       w += β (δ φ − w)` |

`δ = r + γ (θ·φ′) − (θ·φ)`.  
`η = β / α`. Setting **`η = 0`** collapses TDC / GTD2 / GTD1 to TD(0)-style behavior (auxiliary weights stay zero).

## Reproducing Figure 5.1

`PYTHONPATH=. python reproduce_figure_51.py` (or `python run_all.py` without `--quick`) writes:

- `data/<problem>/results.npz` for each problem
- `figures/<problem>/` PNG and PDF panels
- `figures/thesis_figure_5_1.png` and `.pdf` — 2×4 layout like the thesis

Sweep grids match the thesis:

| Problem       | runs | episodes | α grid | η grid |
| ------------- | ---- | -------- | ------ | ------ |
| RW tabular    | 100  | 200      | {.03, .06, .12, .25, .5} | {¼, ½, 1, 2} |
| RW dependent  | 100  | 400      | {.008, .015, .03, .06, .12, .25, .5} | {¼, ½, 1, 2} |
| RW inverted   | 100  | 500      | {.03, .06, .12, .25, .5} | {¼, ½, 1, 2} |
| Boyan chain   | 100  | 100      | {.015, .03, .06, .12, .25, .5, 1, 2} | {⅛, ¼, ½, 1, 2} |

`η = 0` is included so the `η = 0` row for TDC doubles as the TD(0) baseline (thesis p. 49, footnote 3).

## Numerical cross-check

`tests/test_consistency.py` compares small sweeps from this codebase against the **legacy scripts** under  
`archive/Fast_Gradient_TD_Codes_ICML09paper/python_codes/` when that tree is present beside the project (`np.allclose`, tight tolerance), using the same seeds and traversal order. If you don’t have that archive checkout, skip or ignore those tests; the rest of `tests/` still exercises the implementation directly.
