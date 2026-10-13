# ml-experiments

A small notebook collection: one model comparison and a benchmark for a C++ extension.

## Structure

| Folder | Contents |
|--------|----------|
| `comparisons/` | Model A vs B vs C on a dataset |
| `cpp_ext/` | `rolling_stats`, a pybind11 C++ extension with rolling mean, rolling std (Welford) and rolling z-score. Build steps in [`cpp_ext/README.md`](cpp_ext/README.md) |
| `notebooks/` | Benchmark for `cpp_ext/` |

## Notebooks

| Notebook | Description |
|----------|-------------|
| [comparisons/2026-03-02-xgboost-lgbm-rf-wine-quality](comparisons/2026-03-02-xgboost-lgbm-rf-wine-quality.ipynb) | XGBoost vs LightGBM vs Random Forest on UCI Wine Quality — accuracy, AUC, training time, feature importance |
| [notebooks/cpp_rolling_benchmark](notebooks/cpp_rolling_benchmark.ipynb) | Builds `cpp_ext/`, checks its output against pandas, then times C++ vs pandas (and numpy for the mean) at n=1,000,000 with windows 10, 50 and 200. Committed without outputs, so run it to get timings |

---

**Stack:** Python · scikit-learn · XGBoost · LightGBM · pandas · matplotlib · C++17 / pybind11

> _Note: building the C++ extension requires a C++17 compiler and pybind11. On macOS the system clang is sufficient; on Linux install `python3-dev` and `g++` ≥ 9. Build it with `cd cpp_ext && pip install pybind11 && python setup.py build_ext --inplace`._
