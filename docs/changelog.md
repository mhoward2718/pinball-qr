# Changelog

All notable changes to this project are documented here.

## 0.1.0 (Unreleased)

### Added

- **Linear quantile regression** (`QuantileRegressor`)
  - Barrodale-Roberts simplex solver (`"br"`) with rank-inversion CI
  - Frisch-Newton interior-point solver (`"fn"` / `"fnb"`)
  - Preprocessing + Frisch-Newton solver (`"pfn"`) for large datasets
  - L1-penalised Lasso solver (`"lasso"`)
  - Extensible solver registry with `get_solver()` / `register_solver()`

- **Inference**
  - `summary()` with 5 SE methods: rank, iid, nid, ker, boot
  - `bootstrap()` with 3 strategies: xy-pair, wild, mcmb

- **Nonparametric estimation** (`QuantizationQuantileEstimator`)
  - CLVQ optimal quantization grid construction
  - Voronoi cell assignment and conditional quantile estimation
  - Bootstrap averaging over multiple grids

- **sklearn compatibility**
  - Full `check_estimator` compliance (46/46 for linear, 52/52 for nonparametric)
  - Pipeline, cross-validation, and grid search support
  - `BaseQuantileEstimator` abstract base class

- **Datasets**
  - `load_engel()` — Engel food expenditure (235 obs, 1 predictor)
  - `load_barro()` — Barro economic growth (161 obs, 13 predictors)

- **Documentation**
  - GitHub Pages site with MkDocs Material
  - Theory guides, API reference, and example notebooks
