# HomoloMap tutorials

The tutorials are short and task-oriented. Run them in order for a complete
workflow, or open the notebook matching the analysis you need.

| Tutorial | Question | Main output |
|---|---|---|
| [`01_prepare_brain_idps.ipynb`](01_prepare_brain_idps.ipynb) | How should external brain IDPs be formatted and aligned? | aligned tables, composition heatmap, and IDP quality-control profiles |
| [`02_spin_test.ipynb`](02_spin_test.ipynb) | Which individual cell types are spatially associated with each IDP? | spatial correlations, FDR-adjusted p-values, and ranked association plot |
| [`03_total_contribution_and_shap.ipynb`](03_total_contribution_and_shap.ipynb) | How much do all cell types explain jointly, and which features contribute? | total-performance plot and optional ranked SHAP contribution plot |
| [`04_clr_sensitivity.ipynb`](04_clr_sensitivity.ipynb) | Are ratio-map findings robust to compositional transformation? | ratio-versus-CLR effect table and agreement plot |

Tutorial 1 creates `tutorial_outputs/`; later notebooks use those files when
available and otherwise fall back to deterministic toy IDPs. Toy results only
verify execution and have no scientific interpretation.

The notebooks use 100 rotations for speed. Use at least 1,000 rotations and a
predefined multiple-testing plan for scientific analyses.

The spin tutorial may fetch an fsLR sphere through `neuromaps` on first use.
Set `HOMOLOMAP_DATA` to control the shared cache location. Later runs reuse the
cached file; for offline execution, pre-populate the cache or pass a local
surface directory to the loader.

Every notebook ends with a compact visualization generated from the current
inputs. Replacing the toy IDPs therefore updates the figures automatically.
GitHub also displays a deterministic preview near the top of each notebook so
the expected visual output remains visible before the notebook is executed.
