import numpy as np
import pandas as pd

from HomoloMap.datasets import layers


def _raw_layers():
    return {
        "l1": pd.DataFrame({"A": [2.0, 1.0], "B": [2.0, 3.0]}, index=[1, 2]),
        "l2": pd.DataFrame({"A": [6.0, 3.0], "B": [2.0, 1.0]}, index=[1, 2]),
    }


def _patch_layer_io(monkeypatch):
    mapping = {"mapping_column": "subclass", "coverage": 1.0}
    monkeypatch.setattr(
        layers, "load_layer_counts",
        lambda **kwargs: (_raw_layers(), mapping),
    )

    def relabel(data, **kwargs):
        # Mimic atlas interpolation/coverage loss without changing labels.
        result = {key: frame * 0.75 for key, frame in data.items()}
        return result, {"dropped_labels": {key: [] for key in data}}

    monkeypatch.setattr(layers, "relabel_layer_counts", relabel)


def test_fetch_layer_ratio_recloses_within_layer(monkeypatch):
    _patch_layer_io(monkeypatch)
    result, audit = layers.fetch_layer_ratio(
        normalization="within_layer", normalization_order="before_relabel",
        return_mapping=True, as_dict=True,
    )
    for frame in result.values():
        np.testing.assert_allclose(frame.sum(axis=1), 1.0)
    assert audit["reclosed_after_relabel"] is True
    assert audit["max_abs_closure_error"] < 1e-12


def test_fetch_layer_ratio_recloses_across_layers(monkeypatch):
    _patch_layer_io(monkeypatch)
    result = layers.fetch_layer_ratio(
        normalization="within_region_cross_layer",
        normalization_order="before_relabel", as_dict=True,
    )
    total = result["l1"].copy()
    for layer in list(result)[1:]:
        total = total.add(result[layer])
    np.testing.assert_allclose(total, 1.0)
