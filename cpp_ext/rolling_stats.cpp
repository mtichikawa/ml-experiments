#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

py::array_t<double> rolling_mean(py::array_t<double> input, int window) {
    auto buf = input.request();
    if (buf.ndim != 1)
        throw std::runtime_error("Input must be 1-D");
    int n = buf.shape[0];
    auto result = py::array_t<double>(n);
    auto rbuf = result.request();
    double* in_ptr = static_cast<double*>(buf.ptr);
    double* out_ptr = static_cast<double*>(rbuf.ptr);

    // Fill first window-1 elements with NaN
    for (int i = 0; i < window - 1 && i < n; ++i)
        out_ptr[i] = std::numeric_limits<double>::quiet_NaN();

    if (n < window) return result;

    // Compute first window sum
    double sum = 0.0;
    for (int i = 0; i < window; ++i)
        sum += in_ptr[i];
    out_ptr[window - 1] = sum / window;

    // Slide the window
    for (int i = window; i < n; ++i) {
        sum += in_ptr[i] - in_ptr[i - window];
        out_ptr[i] = sum / window;
    }
    return result;
}

py::array_t<double> rolling_std(py::array_t<double> input, int window) {
    auto buf = input.request();
    if (buf.ndim != 1)
        throw std::runtime_error("Input must be 1-D");
    int n = buf.shape[0];
    auto result = py::array_t<double>(n);
    auto rbuf = result.request();
    double* in_ptr = static_cast<double*>(buf.ptr);
    double* out_ptr = static_cast<double*>(rbuf.ptr);

    for (int i = 0; i < window - 1 && i < n; ++i)
        out_ptr[i] = std::numeric_limits<double>::quiet_NaN();

    if (n < window) return result;

    for (int start = 0; start <= n - window; ++start) {
        // Welford's online algorithm over this window
        double mean = 0.0, M2 = 0.0;
        for (int k = 0; k < window; ++k) {
            double x = in_ptr[start + k];
            double delta = x - mean;
            mean += delta / (k + 1);
            double delta2 = x - mean;
            M2 += delta * delta2;
        }
        double variance = (window > 1) ? M2 / (window - 1) : 0.0;
        out_ptr[start + window - 1] = std::sqrt(variance);
    }
    return result;
}

py::array_t<double> rolling_zscore(py::array_t<double> input, int window) {
    auto buf = input.request();
    if (buf.ndim != 1)
        throw std::runtime_error("Input must be 1-D");
    int n = buf.shape[0];
    auto result = py::array_t<double>(n);
    auto rbuf = result.request();
    double* in_ptr = static_cast<double*>(buf.ptr);
    double* out_ptr = static_cast<double*>(rbuf.ptr);

    for (int i = 0; i < window - 1 && i < n; ++i)
        out_ptr[i] = std::numeric_limits<double>::quiet_NaN();

    if (n < window) return result;

    for (int start = 0; start <= n - window; ++start) {
        double mean = 0.0, M2 = 0.0;
        for (int k = 0; k < window; ++k) {
            double x = in_ptr[start + k];
            double delta = x - mean;
            mean += delta / (k + 1);
            double delta2 = x - mean;
            M2 += delta * delta2;
        }
        double variance = (window > 1) ? M2 / (window - 1) : 0.0;
        double std_dev = std::sqrt(variance);
        double last_val = in_ptr[start + window - 1];
        // Clamp to avoid division by zero
        out_ptr[start + window - 1] = (std_dev > 1e-10) ? (last_val - mean) / std_dev : 0.0;
    }
    return result;
}

PYBIND11_MODULE(rolling_stats, m) {
    m.doc() = "C++ rolling window statistics via pybind11";
    m.def("rolling_mean",   &rolling_mean,   "Rolling mean (sliding window)",
          py::arg("arr"), py::arg("window"));
    m.def("rolling_std",    &rolling_std,    "Rolling std (Welford online algorithm)",
          py::arg("arr"), py::arg("window"));
    m.def("rolling_zscore", &rolling_zscore, "Rolling z-score: (x - mean) / std",
          py::arg("arr"), py::arg("window"));
}
