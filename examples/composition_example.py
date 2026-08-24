"""Minimal compositional-data example using synthetic ROI ratios."""

import pandas as pd

from HomoloMap.transforms import transform_composition


ratios = pd.DataFrame(
    {
        "Excitatory": [0.60, 0.45, 0.30],
        "Inhibitory": [0.25, 0.35, 0.40],
        "Non-neuronal": [0.15, 0.20, 0.30],
    },
    index=["ROI-1", "ROI-2", "ROI-3"],
)

clr = transform_composition(ratios, method="clr")
print(clr.round(3))
