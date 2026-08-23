# HomoloMap

HomoloMap is a Python toolkit for relating spatial cell-type maps to human brain imaging phenotypes in a common atlas space. It combines atlas relabeling, compositional transformations, spatially constrained null models, multivariable contribution analysis, and publication-oriented visualization.

The package was developed for analyses in which cell types measured in one biological reference are harmonized by transcriptomic homology and then evaluated against regional human neuroimaging maps. HomoloMap does not assume that homologous labels imply identical abundance or molecular state across species; mapping coverage and unresolved features should always be reported.

## Main capabilities

- Aggregate fine cell labels to subclass or cluster resolution with mapping audits.
- Relabel surface and volumetric data between supported atlas spaces.
- Work with ratio, density, CLR, and ILR feature representations.
- Generate spatial null models and run spin-based association tests.
- Fit linear, random-forest, and support-vector models.
- Summarize total model performance, dominance contribution, and SHAP attribution.
- Analyze laminar cell composition and cortical layer thickness.
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

## Quick example: compositional transformation

```python
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

clr = transform_composition(
    ratios,
    method="clr",
    zero_method="multiplicative",
)
print(clr)
```

## Typical imaging workflow

```python
from HomoloMap.utils import prepare_analysis_data, run_analysis

# Data can be supplied as a DataFrame or CSV path. Mapping policy,
# compositional transform, atlas alignment, and coverage thresholds
# should be specified explicitly for a reproducible analysis.
prepared = prepare_analysis_data(
    data="celltype_ratio.csv",
    feature_type="ratio",
    ctype_level="subclass",
    composition="clr",
    unmapped="drop",
    min_mapping_coverage=0.95,
)
```

See the docstrings in `HomoloMap.datasets`, `HomoloMap.transforms`, `HomoloMap.stats`, and `HomoloMap.utils` for the current API. The package is in alpha development; validate results and record package version, atlas, mapping table, null model, and random seed in every analysis.

## Repository layout

- `src/HomoloMap/datasets`: dataset loaders and mapping utilities.
- `src/HomoloMap/transforms`: atlas, geometry, smoothing, and composition transforms.
- `src/HomoloMap/stats`: spatial nulls, model analysis, and laminar statistics.
- `src/HomoloMap/plotting.py`: surface and statistical visualization.
- `examples`: small runnable examples using synthetic data.
- `tests`: lightweight package tests.
- `data`: original D99 maps, HomoloMap-derived BN maps, mapping table, and provenance audits.

Only small support surfaces and parcellation resources required by the package are bundled. Project-specific raw spatial-transcriptomic and neuroimaging datasets are not included.

## Statistical interpretation

Spatial autocorrelation can inflate ordinary parametric significance. HomoloMap therefore supports spatially constrained null models. Multiple-comparison families, atlas coverage, preprocessing choices, and held-out prediction design remain the responsibility of the analyst. Correlation, dominance, and SHAP results describe spatial association or model dependence, not causality.

## Citation

If you use HomoloMap, cite the repository and the specific release or commit. A formal software citation is provided in [`CITATION.cff`](CITATION.cff).

## License

HomoloMap is released under the [MIT License](LICENSE).
