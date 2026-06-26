# Changelog

## 0.6.5

- Aligned MCMC proposal and acceptance logic with the updated R package implementation.
- Added per-iteration `trace_stats` output from the C++ backend during fitting.
- Added `traceplots()` for burn-in-aware MCMC trace diagnostics.

## 0.6.1

- Migrated AddiVortes to a Python-only package named `addivortes`.
- Added a Pythonic `AddiVortesRegressor` estimator API.
- Added a C++20 pybind11 backend for fitting and prediction support.
- Added numpy and pandas preprocessing, including categorical covariate encoding.
- Added Python packaging metadata, tests, wheel build support, and CI.
