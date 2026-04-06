from setuptools import setup, Extension
import pybind11

ext = Extension(
    "rolling_stats",
    sources=["rolling_stats.cpp"],
    include_dirs=[pybind11.get_include()],
    extra_compile_args=["-O3", "-march=native", "-std=c++17"],
    language="c++",
)

setup(
    name="rolling_stats",
    version="0.1.0",
    ext_modules=[ext],
)
