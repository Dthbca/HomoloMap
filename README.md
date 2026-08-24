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

The documentation is split into short, task-oriented notebooks:

1. [`Prepare brain IDPs`](tutorials/01_prepare_brain_idps.ipynb): input format, BN label alignment, quality control, and saved analysis tables.
2. [`Spatial spin tests`](tutorials/02_spin_test.ipynb): individual cell-type associations and FDR correction.
3. [`Total contribution and SHAP`](tutorials/03_total_contribution_and_shap.ipynb): joint model performance and optional feature attribution.
4. [`CLR sensitivity`](tutorials/04_clr_sensitivity.ipynb): comparison of mapped ratios with a centered log-ratio representation.

See the [`tutorial index`](tutorials/README.md) for prerequisites, expected outputs, recommended execution order, and the visual summary produced by each notebook. Each notebook is focused on one analytical question and can be opened independently. Deterministic toy IDPs are provided only to check execution; replace them with your own BN-indexed brain maps for scientific analysis.

## What you provide

The usual external input is a `pandas.DataFrame` containing brain IDPs:

- rows: atlas regions;
- columns: IDPs, such as cortical thickness effects, functional measures, MEG frequency maps, or other regional imaging phenotypes;
- index: numeric region labels matching the selected atlas;
- values: one regional value per IDP (missing values are allowed only where the chosen analysis supports them).

Cell-type ratios or densities are **not** the usual external input to the analysis workflow. They are predictors supplied by HomoloMap. You may nevertheless load the released cell-type tables directly for a custom analysis.

## Repository layout

- `src/HomoloMap/datasets`: dataset loaders and mapping utilities.
- `src/HomoloMap/transforms`: atlas, geometry, smoothing, and composition transforms.
- `src/HomoloMap/stats`: spatial nulls, model analysis, and laminar statistics.
- `src/HomoloMap/plotting.py`: surface and statistical visualization.
- `tutorials`: ordered, task-oriented notebooks for the complete IDP workflow.
- `examples`: small examples for loading released maps and preparing external IDPs.
- `tests`: lightweight package tests.
- `data`: original D99 maps, HomoloMap-derived BN maps, mapping table, and provenance audits.
- `src/HomoloMap/datasets`: installation-ready annotations, surfaces, atlas volumes, and the canonical D99/BN resources used by the loaders.

The repository includes released D99 and BN cell-type maps. Project-specific brain IDPs are not included and are expected to be supplied by the user.

## Statistical interpretation

Spatial autocorrelation can inflate ordinary parametric significance. HomoloMap therefore supports spatially constrained null models. Multiple-comparison families, atlas coverage, preprocessing choices, and held-out prediction design remain the responsibility of the analyst. Correlation, dominance, and SHAP results describe spatial association or model dependence, not causality.

## Citation

If you use HomoloMap, cite the repository and the specific release or commit. A formal software citation is provided in [`CITATION.cff`](CITATION.cff).

## License

HomoloMap is released under the [MIT License](LICENSE).
