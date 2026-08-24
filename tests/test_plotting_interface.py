import inspect

import pytest

from HomoloMap import plotting


def test_plot_left_exposes_region_labels():
    """NumPy inputs need an explicit route for non-consecutive atlas labels."""
    assert "data_labels" in inspect.signature(plotting.plot_left).parameters


def test_missing_surfplot_is_reported_only_when_plotting(monkeypatch):
    monkeypatch.setattr(plotting, "SURFPLOT_AVAILABLE", False)
    with pytest.raises(ImportError, match=r"HomoloMap\[surface\]"):
        plotting.plot_left([1.0, 2.0], atlas="BN")
