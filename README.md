# HomoloMap

HomoloMap is a Python package for relating regional human brain imaging-derived phenotypes (IDPs) to cell-type-resolved cortical maps. Its main external input is a region-by-IDP table; the package provides released cell-type maps, atlas transformations, spatial null models, multivariable analyses, and visualization utilities.

The released maps combine the whole-cortex macaque spatial-transcriptomic dataset of [Chen *et al.* (2023)](https://doi.org/10.1016/j.cell.2023.06.009) with the integrated adult human brain single-nucleus RNA-sequencing taxonomy of [Siletti *et al.* (2023)](https://doi.org/10.1126/science.add7046). Macaque spatial cell labels are harmonized to the human reference at subclass and cluster levels. Cortical areal correspondence follows the cross-species joint-embedding method of [Xu *et al.* (2020)](https://doi.org/10.1016/j.neuroimage.2020.117346), which extracts matched functional-connectivity gradients in a shared macaque–human space and uses them to guide cortical alignment. Regional source maps in the [D99 macaque atlas](https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/nonhuman/macaque_tempatl/atlas_d99v2.html) are then summarized in the [Human Brainnetome Atlas](https://doi.org/10.1093/cercor/bhw157).

HomoloMap records unmapped source types and retained mapping coverage rather than silently discarding them. Ratio maps are reclosed after cell-type aggregation and spatial relabeling. The resulting homologous labels represent transcriptomic correspondence and should not be interpreted as evidence of identical abundance or molecular state across species.


## Main capabilities

- Load released human-aligned maps at subclass or cluster resolution with coverage audits.
- Transform cortical data between supported atlas spaces.
- Work with ratio, density, CLR, and ILR feature representations.
- Generate spatial null models and run spin-based association tests.
- Fit linear, random-forest, and support-vector models.
- Summarize total model performance, dominance contribution, and SHAP attribution.
- Analyze whole-cortex or laminar cell composition against user-supplied IDPs.
- Produce cortical surface plots, heatmaps, and contribution summaries.

## Included cell-type maps

The repository includes the spatial maps used to demonstrate HomoloMap:

| File | Atlas | Shape | Meaning |
|---|---|---:|---|
| `data/maps/D99/ctype_ratio_plot_D99.csv` | D99 | 132 × 226 | Original fine cell-type ratio map |
| `data/maps/D99/ctype_density_plot_D99.csv` | D99 | 141 × 257 | Original fine cell-type density map |
| `data/maps/BN/ctype_ratio_BN_23_subclass.csv` | Brainnetome | 105 × 23 | HomoloMap mapped and reclosed subclass composition |
| `data/maps/BN/ctype_ratio_BN_71_cluster.csv` | Brainnetome | 105 × 71 | HomoloMap mapped and reclosed cluster composition |

The Brainnetome files use the 105 cortical labels from one hemisphere (`1, 3, ..., 209`). They were generated from the D99 ratio map using the exact `plot` key in `data/mappings/cluster_mapping_dict.csv`. Of 226 source features, 191 were mapped; 35 unresolved features accounted for approximately 11.29% of the original ratio mass and were removed before reclosure. D99 labels 106, 118, and 194 had no target contribution during regional relabeling. See [`data/README.md`](data/README.md) and the audit JSON files for full provenance and interpretation.

## Installation

```bash
git clone https://github.com/Dthbca/HomoloMap.git
cd HomoloMap
python -m pip install -e .
```

Optional functionality can be installed with:

```bash
python -m pip install -e ".[all]"
```

Python 3.9 or newer is required.

## Tutorial

Start with [`tutorials/01_brain_idp_analysis.ipynb`](tutorials/01_brain_idp_analysis.ipynb). It walks through the intended workflow from an external region-by-IDP table to atlas-label quality control, spatial spin tests, total multivariable contribution, optional SHAP attribution, and a CLR sensitivity analysis. The notebook includes deterministic toy IDPs for an execution check; replace them with your own BN-indexed brain maps for scientific analysis.

## What you provide

The usual external input is a `pandas.DataFrame` containing brain IDPs:

- rows: atlas regions;
- columns: IDPs, such as cortical thickness effects, functional measures, MEG frequency maps, or other regional imaging phenotypes;
- index: numeric region labels matching the selected atlas;
- values: one regional value per IDP (missing values are allowed only where the chosen analysis supports them).

Cell-type ratios or densities are **not** the usual external input to the analysis workflow. They are predictors supplied by HomoloMap. You may nevertheless load the released cell-type tables directly for a custom analysis.

## Tutorial: relate brain IDPs to cell-type maps

### 1. Prepare an IDP table

```python
import pandas as pd

# CSV layout:
# region,myelin_IDP,MEG_alpha_IDP
# 1,0.42,-0.18
# 3,0.37,-0.11
# ...
idps = pd.read_csv("brain_idps_bn.csv", index_col=0)
idps.index = idps.index.astype(int)
```

The example uses Brainnetome (BN) labels. The distributed BN maps contain 105 cortical regions with labels `1, 3, ..., 209`, so the IDP table should use the same labels or a subset of them.

### 2. Inspect and align the supplied cell-type map

```python
from pathlib import Path
import pandas as pd

celltypes = pd.read_csv(
    Path("data/maps/BN/ctype_ratio_BN_23_subclass.csv"),
    index_col=0,
)
celltypes.index = celltypes.index.astype(int)

shared = celltypes.index.intersection(idps.index, sort=False)
X = celltypes.loc[shared]
Y = idps.loc[shared]

print(f"Using {len(shared)} shared BN regions")
print(f"Predictors: {X.shape[1]} cell-type subclasses")
print(f"Outcomes: {Y.shape[1]} brain IDPs")
```

Use `ctype_ratio_BN_71_cluster.csv` instead when finer cluster-level interpretation is required. Subclass and cluster analyses should be reported as distinct feature resolutions, not pooled into one multiple-testing family.

### 3. Run the standard analysis workflow

For package-managed cell-type predictors, pass the external IDP table through `data`:

```python
from HomoloMap.utils import run_analysis

result = run_analysis(
    data=idps,                 # external outcomes, not cell-type data
    atlas="BN",
    feature_type="ratio",
    ctype_level="subclass",
    n_spins=1000,
    metric="pearsonr",
    cumulative=True,
    mode="linear",
    explanations="shap",     # or "dominance" for linear models
    unmapped="drop",
    renormalize=True,
    random_state=42,
)

spin_results = result["correlation"]
total_model = result["cumulative_effects"]
celltype_explanations = result["explanations"]
```

The outputs answer complementary questions:

- `correlation`: which individual cell-type maps are spatially associated with each IDP, using spin-based nulls and adjusted p-values;
- `cumulative_effects`: how much spatial variation in each IDP is explained jointly by all selected cell-type predictors, with spatial permutation inference;
- `explanations`: how the fitted multivariable model distributes contribution across individual cell types (SHAP) or predictors (dominance analysis).

### 4. CLR sensitivity analysis for ratio maps

Raw mapped ratios are the primary interpretable representation. Because ratios are compositional, repeat the analysis with a centered log-ratio (CLR) transform as a sensitivity analysis:

```python
clr_result = run_analysis(
    data=idps,
    atlas="BN",
    feature_type="ratio",
    ctype_level="subclass",
    n_spins=1000,
    cumulative=True,
    explanations="shap",
    renormalize=True,
    correlation_transform="clr",
    cumulative_transform="clr",
    explanation_transform="clr",
    zero_method="multiplicative",
    random_state=42,
)
```

CLR coefficients and SHAP values describe log-ratio contrasts, so they should not be interpreted as absolute abundance effects. CLR is not appropriate for density features unless those values have first been given a defensible compositional meaning.

### 5. Reproducibility checks

Before interpreting results, verify that:

1. the IDP and cell-type tables use the same atlas and label convention;
2. the number and identity of shared regions are recorded;
3. mapping coverage and unresolved source cell types are reported;
4. each stated multiple-testing family is corrected separately;
5. atlas, hemisphere, spin method, number of rotations, random seed, feature resolution, and compositional transform are saved with the results.

See the docstrings in `HomoloMap.datasets`, `HomoloMap.transforms`, `HomoloMap.stats`, and `HomoloMap.utils` for the current API. The package is in alpha development; validate results before scientific interpretation.

## Repository layout

- `src/HomoloMap/datasets`: dataset loaders and mapping utilities.
- `src/HomoloMap/transforms`: atlas, geometry, smoothing, and composition transforms.
- `src/HomoloMap/stats`: spatial nulls, model analysis, and laminar statistics.
- `src/HomoloMap/plotting.py`: surface and statistical visualization.
- `examples`: small examples for loading released maps and preparing external IDPs.
- `tests`: lightweight package tests.
- `data`: original D99 maps, HomoloMap-derived BN maps, mapping table, and provenance audits.

The repository includes released D99 and BN cell-type maps. Project-specific brain IDPs are not included and are expected to be supplied by the user.

## Statistical interpretation

Spatial autocorrelation can inflate ordinary parametric significance. HomoloMap therefore supports spatially constrained null models. Multiple-comparison families, atlas coverage, preprocessing choices, and held-out prediction design remain the responsibility of the analyst. Correlation, dominance, and SHAP results describe spatial association or model dependence, not causality.

## Citation

If you use HomoloMap, cite the repository and the specific release or commit. A formal software citation is provided in [`CITATION.cff`](CITATION.cff).

## License

HomoloMap is released under the [MIT License](LICENSE).
