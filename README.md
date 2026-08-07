# ml-experiments

A running notebook collection — model comparisons, dataset explorations, and paper replications.

## Structure

| Folder | Contents |
|--------|----------|
| `comparisons/` | Model A vs B vs C on a dataset |
| `explorations/` | Dataset deep dives and EDA |
| `replications/` | Paper or technique implementations |
| `scratch/` | Quick one-offs and experiments |

## Notebooks

### comparisons/
| Notebook | Description |
|----------|-------------|
| [2026-03-02-xgboost-lgbm-rf-wine-quality](comparisons/2026-03-02-xgboost-lgbm-rf-wine-quality.ipynb) | XGBoost vs LightGBM vs Random Forest on UCI Wine Quality — accuracy, AUC, training time, feature importance |

---

---

**Stack:** Python · scikit-learn · XGBoost · LightGBM · PyTorch · pandas · matplotlib

> _Note: building the C++ extensions requires a C++17 compiler and pybind11 development headers. On macOS the system clang is sufficient; on Linux install `python3-dev` and `g++` ≥ 9. Run `pip install -e .` from the repo root to compile and install in editable mode._
