# Installation

## From PyPI (recommended)

```bash
pip install pinball-qr
```

## From Source

Pinball includes compiled Fortran extensions, so building from source
requires a Fortran compiler (e.g. `gfortran`) and the
[Meson](https://mesonbuild.com/) build system.

```bash
git clone https://github.com/mhoward2718/pinball-qr.git
cd pinball-qr
python -m venv .venv
source .venv/bin/activate
pip install meson-python meson ninja numpy
pip install -e . --no-build-isolation
```

### macOS

```bash
brew install gcc  # provides gfortran
```

### Ubuntu / Debian

```bash
sudo apt install gfortran
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| NumPy | ≥ 1.23 | Array operations |
| SciPy | ≥ 1.10 | Statistical distributions, sparse linear algebra |
| scikit-learn | ≥ 1.3 | Base estimator classes, validation utilities |

## Verify Installation

```python
import pinball
print(pinball.__version__)

# Quick smoke test — fit the Engel dataset
from pinball import QuantileRegressor, load_engel
X, y = load_engel(return_X_y=True)
model = QuantileRegressor(tau=0.5).fit(X, y)
print(f"Intercept: {model.intercept_:.2f}, Slope: {model.coef_[0]:.4f}")
```
