# rolling_stats — C++ Rolling Window Statistics via pybind11

A compiled C++ extension exposing three rolling window functions to Python:

| Function | Description |
|---|---|
| `rolling_mean(arr, window)` | Sliding-window mean using running sum (O(n)) |
| `rolling_std(arr, window)` | Rolling sample std using Welford's online algorithm |
| `rolling_zscore(arr, window)` | Rolling z-score: `(x - mean) / std` per window |

All functions return a `numpy.ndarray` of `float64` with `NaN` filled for the
first `window - 1` positions, matching pandas `.rolling()` behavior.

## Why C++?

pandas `.rolling().std()` computes variance with a two-pass algorithm (compute
mean, then sum squared deviations). **Welford's online algorithm** does it in a
single pass with better numerical stability — no catastrophic cancellation from
subtracting large numbers. For streaming or tick-data pipelines where you're
processing windows one element at a time, a single-pass approach avoids
re-reading data.

The C++ layer also eliminates Python interpreter overhead per element, which
matters most for large arrays (n >= 100 000) and small windows where pandas'
internal Cython path has less room to amortize setup cost.

## Build

```bash
cd cpp_ext
pip install pybind11
python setup.py build_ext --inplace
```

This produces a `.so` file in `cpp_ext/` (e.g. `rolling_stats.cpython-311-darwin.so`).

## Usage

```python
import sys
sys.path.insert(0, "path/to/cpp_ext")

import numpy as np
import rolling_stats

arr = np.random.randn(1_000_000)

mean   = rolling_stats.rolling_mean(arr, window=50)
std    = rolling_stats.rolling_std(arr, window=50)
zscore = rolling_stats.rolling_zscore(arr, window=50)
```

Output arrays are the same length as `arr`. The first `window - 1` values are
`NaN`, identical to `pandas.Series(arr).rolling(window).mean()` etc.

## Expected output

After building you should see a file like:

```
cpp_ext/rolling_stats.cpython-311-darwin.so
```

Import it directly:

```python
import rolling_stats
print(rolling_stats.__doc__)
# C++ rolling window statistics via pybind11
```

## Benchmark

See `notebooks/cpp_rolling_benchmark.ipynb` for a full benchmark comparing
C++, pandas, and numpy across `n=1_000_000` with window sizes 10, 50, and 200.
