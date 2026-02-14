"""Pinball — fast quantile regression for Python.

Provides an sklearn-compatible :class:`QuantileRegressor` backed by
high-performance Fortran solvers (Barrodale-Roberts simplex, Frisch-Newton
interior point) ported from R's ``quantreg`` package.

Quick start::

    from pinball import QuantileRegressor
    model = QuantileRegressor(tau=0.5, method="fn")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
"""

import os as _os
import sys as _sys


def _setup_windows_dll_dirs():
    """On Windows, register DLL search directories.

    Python 3.8+ no longer searches PATH for DLL dependencies of
    extension modules (.pyd) or ctypes-loaded DLLs.  We must call
    ``os.add_dll_directory()`` so that runtime libraries such as
    ``libopenblas.dll`` and ``libgfortran-5.dll`` can be found.
    """
    if _sys.platform != "win32":
        return

    _added = set()

    # 1. The package directory itself (delvewheel puts vendored DLLs here
    #    or in a .libs sub-directory).
    pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
    for d in (pkg_dir, _os.path.join(pkg_dir, ".libs")):
        if _os.path.isdir(d) and d not in _added:
            try:
                _os.add_dll_directory(d)
                _added.add(d)
            except OSError:
                pass

    # 2. Directories from PATH — needed for development / CI installs
    #    where OpenBLAS or MinGW runtime DLLs live in external dirs
    #    (e.g. C:\openblas\bin, MinGW bin).
    for entry in _os.environ.get("PATH", "").split(_os.pathsep):
        entry = entry.strip()
        if entry and _os.path.isdir(entry) and entry not in _added:
            try:
                _os.add_dll_directory(entry)
                _added.add(entry)
            except OSError:
                pass


_setup_windows_dll_dirs()

__author__ = """Michael Howard"""
__email__ = "mah38900@gmail.com"
__version__ = "0.1.0"

from pinball.datasets import load_engel
from pinball.estimators._base import BaseQuantileEstimator
from pinball.linear._bootstrap import BootstrapResult, bootstrap
from pinball.linear._estimator import QuantileRegressor
from pinball.linear._inference import InferenceResult, summary
from pinball.linear.solvers import get_solver, list_solvers, register_solver
from pinball.linear.solvers.base import BaseSolver, SolverResult

__all__ = [
    "BaseQuantileEstimator",
    "QuantileRegressor",
    "BaseSolver",
    "SolverResult",
    "InferenceResult",
    "BootstrapResult",
    "summary",
    "bootstrap",
    "get_solver",
    "list_solvers",
    "register_solver",
    "load_engel",
]
