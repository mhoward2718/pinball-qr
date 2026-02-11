"""Barro economic growth dataset.

Cross-country growth data used by Barro (1991) and widely referenced
in quantile regression examples.  161 countries, 13 predictors,
target is net GDP growth rate.

Source: Barro, R. and Lee, J.-W. (1994), via ``quantreg::barro``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.utils import Bunch

_DATA_DIR = Path(__file__).parent

_FEATURE_NAMES = [
    "lgdp2", "mse2", "fse2", "fhe2", "mhe2", "lexp2", "lintr2",
    "gedy2", "Iy2", "gcony2", "lblakp2", "pol2", "ttrad2",
]


def load_barro() -> Bunch:
    """Load the Barro cross-country growth dataset.

    Returns
    -------
    sklearn.utils.Bunch
        Dictionary-like with keys:

        - ``data``  : ndarray, shape (161, 13) — predictor variables
        - ``target`` : ndarray, shape (161,) — net GDP growth (y.net)
        - ``feature_names`` : list of str
        - ``DESCR`` : str — dataset description
    """
    csv_path = _DATA_DIR / "barro.csv"
    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    y = raw[:, 0]
    X = raw[:, 1:]

    return Bunch(
        data=X,
        target=y,
        feature_names=_FEATURE_NAMES,
        DESCR=(
            "Barro (1991) cross-country economic growth data.  161 countries, "
            "13 predictors including initial GDP, schooling enrolment ratios, "
            "life expectancy, investment, government consumption, and more.  "
            "Target is the net GDP growth rate."
        ),
    )
