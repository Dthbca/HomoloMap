"""Analyze external brain IDP maps with HomoloMap cell-type predictors."""

import pandas as pd

from HomoloMap.utils import run_analysis


# Rows are BN region labels; columns are user-provided imaging-derived phenotypes.
idps = pd.read_csv("brain_idps_bn.csv", index_col=0)
idps.index = idps.index.astype(int)

result = run_analysis(
    data=idps,
    atlas="BN",
    feature_type="ratio",
    ctype_level="subclass",
    n_spins=1000,
    cumulative=True,
    mode="linear",
    explanations="shap",
    unmapped="drop",
    renormalize=True,
    random_state=42,
)

result["correlation"].to_csv("idp_celltype_spin_results.csv")
result["cumulative_effects"].to_csv("idp_celltype_total_models.csv")
