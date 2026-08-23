"""Load and validate the released HomoloMap BN cell-composition maps."""

from pathlib import Path

import numpy as np
import pandas as pd


root = Path(__file__).parents[1]
subclass = pd.read_csv(
    root / "data/maps/BN/ctype_ratio_BN_23_subclass.csv", index_col=0
)
cluster = pd.read_csv(
    root / "data/maps/BN/ctype_ratio_BN_71_cluster.csv", index_col=0
)

assert subclass.shape == (105, 23)
assert cluster.shape == (105, 71)
assert np.allclose(subclass.sum(axis=1), 1.0)
assert np.allclose(cluster.sum(axis=1), 1.0)

print("subclass", subclass.shape)
print("cluster", cluster.shape)

