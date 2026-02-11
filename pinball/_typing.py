"""Type aliases and common types for the pinball package."""

from __future__ import annotations

from typing import Union

import numpy as np
import numpy.typing as npt

# Array-like types accepted as input
ArrayLike = Union[np.ndarray, list, tuple]

# Strict numpy array types returned from computations
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int32]
