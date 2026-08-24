# HomoloMap tutorials

The tutorials are short and task-oriented. Run them in order for a complete
workflow, or open the notebook matching the analysis you need.

| Tutorial | Question | Main output |
|---|---|---|
| [`01_prepare_brain_idps.ipynb`](01_prepare_brain_idps.ipynb) | How should external brain IDPs be formatted and aligned? | validated `X` and `Y` tables |
| [`02_spin_test.ipynb`](02_spin_test.ipynb) | Which individual cell types are spatially associated with each IDP? | spin correlations and FDR-adjusted p-values |
| [`03_total_contribution_and_shap.ipynb`](03_total_contribution_and_shap.ipynb) | How much do all cell types explain jointly, and which features contribute? | total models and optional SHAP summaries |
| [`04_clr_sensitivity.ipynb`](04_clr_sensitivity.ipynb) | Are ratio-map findings robust to compositional transformation? | ratio-versus-CLR comparison |

Tutorial 1 creates `tutorial_outputs/`; later notebooks use those files when
available and otherwise fall back to deterministic toy IDPs. Toy results only
verify execution and have no scientific interpretation.

The notebooks use 100 rotations for speed. Use at least 1,000 rotations and a
predefined multiple-testing plan for scientific analyses.
