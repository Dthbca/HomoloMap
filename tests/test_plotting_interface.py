import inspect

from HomoloMap import plotting


def test_plot_left_exposes_region_labels():
    """NumPy inputs need an explicit route for non-consecutive atlas labels."""
    assert "data_labels" in inspect.signature(plotting.plot_left).parameters
