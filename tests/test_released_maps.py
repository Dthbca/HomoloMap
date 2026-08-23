from pathlib import Path

import numpy as np
import pandas as pd


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

