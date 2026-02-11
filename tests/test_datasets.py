"""Tests for dataset loaders."""

import numpy as np
import pytest

from pinball.datasets import load_engel, load_barro


class TestLoadEngel:

    def test_returns_bunch(self):
        data = load_engel()
        assert hasattr(data, "data")
        assert hasattr(data, "target")
        assert hasattr(data, "feature_names")
        assert hasattr(data, "DESCR")

    def test_shape(self):
        data = load_engel()
        assert data.data.shape == (235, 1)
        assert data.target.shape == (235,)

    def test_feature_names(self):
        data = load_engel()
        assert data.feature_names == ["income"]

    def test_values_positive(self):
        data = load_engel()
        assert np.all(data.data > 0)
        assert np.all(data.target > 0)

    def test_known_income_range(self):
        """Income ranges roughly 420–5800 in the original data."""
        data = load_engel()
        assert data.data.min() > 300
        assert data.data.max() < 6000

    def test_descr_is_string(self):
        data = load_engel()
        assert isinstance(data.DESCR, str)
        assert len(data.DESCR) > 10


class TestLoadBarro:

    def test_returns_bunch(self):
        data = load_barro()
        assert hasattr(data, "data")
        assert hasattr(data, "target")
        assert hasattr(data, "feature_names")

    def test_shape(self):
        data = load_barro()
        assert data.data.shape == (161, 13)
        assert data.target.shape == (161,)

    def test_feature_names_count(self):
        data = load_barro()
        assert len(data.feature_names) == 13
        assert "lgdp2" in data.feature_names

    def test_target_has_variation(self):
        data = load_barro()
        assert data.target.std() > 0
