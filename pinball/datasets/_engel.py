"""Engel food-expenditure dataset (Ernst Engel, 1857).

Classic dataset used in virtually every quantile-regression tutorial.
235 Belgian working-class households; single predictor (income)
predicting food expenditure.

Source: Koenker, R. and Bassett, G. (1982), Table 7.
Also available in R via ``quantreg::engel``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.utils import Bunch

_DATA_DIR = Path(__file__).parent


def load_engel() -> Bunch:
    """Load the Engel food-expenditure dataset.

    Returns
    -------
    sklearn.utils.Bunch
        Dictionary-like with keys:

        - ``data``  : ndarray, shape (235, 1) — household income
        - ``target`` : ndarray, shape (235,) — food expenditure
        - ``feature_names`` : list of str
        - ``DESCR`` : str — dataset description
    """
    csv_path = _DATA_DIR / "engel.csv"
    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    income = raw[:, 0]
    foodexp = raw[:, 1]

    return Bunch(
        data=income.reshape(-1, 1),
        target=foodexp,
        feature_names=["income"],
        DESCR=(
            "Engel food expenditure data.  235 Belgian working-class "
            "households from 1857.  The single predictor is household "
            "income; the response is food expenditure.  This is the "
            "canonical dataset for illustrating quantile regression."
        ),
    )
