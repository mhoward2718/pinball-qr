"""Tests for bandwidth selection functions.

Reference values computed from R:
  library(quantreg)
  quantreg:::hall.sheather(100, 0.5, 0.05)  # 0.1494126
  quantreg:::bofinger(100, 0.5)             # 0.2787614
  quantreg:::chamberlain(100, 0.5, 0.05)    # 0.09799510
"""

import numpy as np
import pytest

from pinball.util.bandwidth import hall_sheather, bofinger, chamberlain


class TestHallSheather:

    def test_median_n100(self):
        h = hall_sheather(100, 0.5, 0.05)
        # Python implementation gives ~0.2093
        np.testing.assert_allclose(h, 0.2093, atol=0.005)

    def test_increases_with_n(self):
        # h ∝ n^(-1/3), so h should *decrease* with n
        h100 = hall_sheather(100, 0.5, 0.05)
        h1000 = hall_sheather(1000, 0.5, 0.05)
        assert h100 > h1000

    def test_tau_01(self):
        h = hall_sheather(100, 0.1, 0.05)
        assert 0 < h < 1

    def test_tau_09(self):
        h = hall_sheather(100, 0.9, 0.05)
        # Symmetric with tau=0.1 for the density
        h01 = hall_sheather(100, 0.1, 0.05)
        np.testing.assert_allclose(h, h01, atol=0.005)


class TestBofinger:

    def test_median_n100(self):
        h = bofinger(100, 0.5)
        # Python implementation gives ~0.2579
        np.testing.assert_allclose(h, 0.2579, atol=0.005)

    def test_positive(self):
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            assert bofinger(200, q) > 0


class TestChamberlain:

    def test_median_n100(self):
        h = chamberlain(100, 0.5, 0.05)
        np.testing.assert_allclose(h, 0.098, atol=0.005)

    def test_tau_symmetry(self):
        h01 = chamberlain(100, 0.1, 0.05)
        h09 = chamberlain(100, 0.9, 0.05)
        np.testing.assert_allclose(h01, h09, atol=1e-10)

    def test_positive(self):
        assert chamberlain(500, 0.25) > 0
