from pathlib import Path

import numpy as np
import pandas as pd

from HomoloMap.datasets import fetch_ctype_ratio


ROOT = Path(__file__).parents[1]


def test_released_bn_maps_are_closed_and_finite():
    expected = {
        "ctype_ratio_BN_23_subclass.csv": (105, 23),
        "ctype_ratio_BN_71_cluster.csv": (105, 71),
    }
    for filename, shape in expected.items():
        data = pd.read_csv(ROOT / "data" / "maps" / "BN" / filename, index_col=0)
        assert data.shape == shape
        assert np.isfinite(data.to_numpy()).all()
        assert np.allclose(data.sum(axis=1), 1.0)
        assert list(data.index) == list(range(1, 210, 2))


def test_original_d99_map_shapes():
    ratio = pd.read_csv(
        ROOT / "data" / "maps" / "D99" / "ctype_ratio_plot_D99.csv",
        index_col=0,
    )
    density = pd.read_csv(
        ROOT / "data" / "maps" / "D99" / "ctype_density_plot_D99.csv",
        index_col=0,
    )
    assert ratio.shape == (132, 226)
    assert density.shape == (141, 257)


def test_packaged_bn_loaders_use_current_released_maps():
    subclass = fetch_ctype_ratio(level="subclass", atlas="BN")
    cluster = fetch_ctype_ratio(level="cluster", atlas="BN")
    assert subclass.shape == (105, 23)
    assert cluster.shape == (105, 71)
    assert np.allclose(subclass.sum(axis=1), 1.0)
    assert np.allclose(cluster.sum(axis=1), 1.0)
    assert subclass.attrs["celltype_mapping"]["mapping_column"] == "subclass"


def test_packaged_d99_loader_uses_cluster_mapping_dict():
    subclass, audit = fetch_ctype_ratio(
        level="subclass", atlas="D99", return_mapping=True
    )
    assert subclass.shape == (132, 23)
    assert np.allclose(subclass.sum(axis=1), 1.0)
    assert audit["n_original_types"] == 226
    assert audit["n_mapped_types"] == 191
    assert audit["n_unresolved_types"] == 35
