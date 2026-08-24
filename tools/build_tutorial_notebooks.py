"""Build the small, task-oriented HomoloMap tutorial notebooks."""

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
OUT = ROOT / "tutorials"


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.9"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


notebooks = {
    "01_prepare_brain_idps.ipynb": [
        md("""# Tutorial 1 — Prepare brain IDPs

**Goal:** create label-aligned predictor (`X`) and outcome (`Y`) tables. The external input is a BN region-by-IDP table; HomoloMap supplies the cell-type predictors. This notebook performs only data preparation and quality control."""),
        code("""from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

HERE = Path.cwd() if (Path.cwd() / 'tutorial_utils.py').exists() else Path.cwd() / 'tutorials'
sys.path.insert(0, str(HERE))
from tutorial_utils import find_repo_root, load_celltype_map, load_idps, align_and_validate

ROOT = find_repo_root()
CELLTYPE_LEVEL = 'subclass'  # or 'cluster'
IDP_PATH = None  # e.g. ROOT / 'my_data' / 'brain_idps_bn.csv'"""),
        md("""## Input contract

The IDP CSV must have BN region labels as rows and imaging phenotypes as columns. Labels—not row position—define alignment. The fallback creates deterministic toy IDPs only to verify execution."""),
        code("""X_source = load_celltype_map(CELLTYPE_LEVEL, ROOT)
Y_source, is_toy = load_idps(IDP_PATH, X_source.index, seed=42)
X, Y = align_and_validate(X_source, Y_source)

print('Toy IDPs:', is_toy)
print('Cell types:', X.shape, 'IDPs:', Y.shape)
print('BN labels:', X.index.min(), 'to', X.index.max())
display(Y.head())"""),
        code("""fig, ax = plt.subplots(figsize=(10, 3.5))
sns.heatmap(X.T, cmap='mako', xticklabels=False, ax=ax)
ax.set(xlabel='BN regions', ylabel='Cell types', title='Aligned cell-type composition')
fig.tight_layout()"""),
        md("""## Save validated inputs

Later tutorials load these files. Record the atlas, hemisphere, feature resolution, retained ROI labels, and mapping audit with a scientific analysis."""),
        code("""OUTPUT = ROOT / 'tutorial_outputs'
OUTPUT.mkdir(exist_ok=True)
X.to_csv(OUTPUT / 'aligned_celltype_predictors.csv')
Y.to_csv(OUTPUT / 'aligned_brain_idps.csv')
print('Saved to', OUTPUT.resolve())"""),
    ],
    "02_spin_test.ipynb": [
        md("""# Tutorial 2 — Spatial associations with spin tests

**Goal:** test each cell-type–IDP association while preserving cortical spatial autocorrelation in the null model. This follows the logic of the [BrainSpace spin tutorial](https://brainspace.readthedocs.io/en/latest/python_doc/auto_examples/plot_tutorial3.html)."""),
        code("""from pathlib import Path
import sys
import pandas as pd

HERE = Path.cwd() if (Path.cwd() / 'tutorial_utils.py').exists() else Path.cwd() / 'tutorials'
sys.path.insert(0, str(HERE))
from tutorial_utils import find_repo_root, load_prepared_or_example, align_and_validate
from HomoloMap.stats import SpinTest
from HomoloMap.utils import run_spin_correlations

ROOT = find_repo_root()
N_SPINS = 100  # use >=1,000 for scientific analysis
SEED = 42
X, Y = load_prepared_or_example(ROOT, level='subclass')
X, Y = align_and_validate(X, Y, require_complete_bn=True)"""),
        md("""## Generate one reusable spatial null model

The same rotations are reused across cell types. Benjamini–Hochberg correction is applied separately within each IDP across the selected cell-type resolution."""),
        code("""spinner = SpinTest(atlas='BN', n_spins=N_SPINS, method='Alexander-Bloch', seed=SEED)
spin = run_spin_correlations(
    X, Y, spinner, metric='pearsonr', FDR='fdr_bh', n_jobs=1,
    composition_transform='none',
)
spin.head()"""),
        code("""idp = Y.columns[0]
r_col = f'{idp}_ratio_spin_r'
q_col = f'{idp}_ratio_spin_p_adj'
display(spin[[r_col, q_col]].sort_values(q_col).head(10))"""),
        md("""## Interpretation

The coefficient gives direction and magnitude; the spin p-value evaluates spatial correspondence. FDR significance is not evidence of causality. Define the comparison family before examining results."""),
        code("""OUTPUT = ROOT / 'tutorial_outputs'
OUTPUT.mkdir(exist_ok=True)
spin.to_csv(OUTPUT / 'spin_results_ratio.csv')"""),
    ],
    "03_total_contribution_and_shap.ipynb": [
        md("""# Tutorial 3 — Total contribution and SHAP

**Goal:** separate two multivariable questions: how much the complete cell-type set explains jointly, and how fitted predictions are distributed among individual features."""),
        code("""from pathlib import Path
import sys

HERE = Path.cwd() if (Path.cwd() / 'tutorial_utils.py').exists() else Path.cwd() / 'tutorials'
sys.path.insert(0, str(HERE))
from tutorial_utils import find_repo_root, load_prepared_or_example, align_and_validate
from HomoloMap.stats import SpinTest
from HomoloMap.utils import run_cumulative_models, run_explanation_analysis

ROOT = find_repo_root()
N_SPINS = 100  # use >=1,000 for scientific analysis
SEED = 42
X, Y = load_prepared_or_example(ROOT, level='subclass')
X, Y = align_and_validate(X, Y, require_complete_bn=True)
spinner = SpinTest(atlas='BN', n_spins=N_SPINS, seed=SEED)"""),
        md("""## Joint model

The total model tests the complete predictor set against rotated outcomes. It is not obtained by summing univariate correlations or significant cells."""),
        code("""total = run_cumulative_models(
    X, Y, spinner, mode='linear', n_spins=N_SPINS,
    FDR='fdr_bh', n_jobs=1, composition_transform='none',
)
total"""),
        md("""## Individual contributions

SHAP partitions fitted predictions among features. It measures model dependence rather than biological causality. Install `HomoloMap[explain]` and set the flag below to run it."""),
        code("""RUN_SHAP = False

if RUN_SHAP:
    explanations = run_explanation_analysis(
        X, Y, method='shap', mode='linear', n_jobs=1,
        random_state=SEED, composition_transform='none',
    )
    first = explanations[Y.columns[0]]
    print('Total contribution:', first['total_contribution'])
    display(first['individual_ctype_contribution'].head(10))"""),
        code("""OUTPUT = ROOT / 'tutorial_outputs'
OUTPUT.mkdir(exist_ok=True)
total.to_csv(OUTPUT / 'total_models.csv')"""),
    ],
    "04_clr_sensitivity.ipynb": [
        md("""# Tutorial 4 — CLR sensitivity analysis

**Goal:** assess whether conclusions from mapped cell-type ratios depend on compositional closure. Raw mapped ratios remain the primary interpretable representation; centered log-ratio (CLR) results are a sensitivity analysis."""),
        code("""from pathlib import Path
import sys
import pandas as pd

HERE = Path.cwd() if (Path.cwd() / 'tutorial_utils.py').exists() else Path.cwd() / 'tutorials'
sys.path.insert(0, str(HERE))
from tutorial_utils import find_repo_root, load_prepared_or_example, align_and_validate
from HomoloMap.stats import SpinTest
from HomoloMap.utils import run_spin_correlations

ROOT = find_repo_root()
N_SPINS = 100
SEED = 42
X, Y = load_prepared_or_example(ROOT, level='subclass')
X, Y = align_and_validate(X, Y, require_complete_bn=True)
spinner = SpinTest(atlas='BN', n_spins=N_SPINS, seed=SEED)"""),
        code("""ratio = run_spin_correlations(
    X, Y, spinner, metric='pearsonr', FDR='fdr_bh', n_jobs=1,
    composition_transform='none',
)
clr = run_spin_correlations(
    X, Y, spinner, metric='pearsonr', FDR='fdr_bh', n_jobs=1,
    composition_transform='clr',
    composition_params={'zero_method': 'multiplicative'},
)"""),
        code("""idp = Y.columns[0]
r_col = f'{idp}_ratio_spin_r'
comparison = pd.DataFrame({'ratio_r': ratio[r_col], 'clr_r': clr[r_col]})
print('Across-cell-type agreement')
display(comparison.corr())
display(comparison.reindex(comparison.ratio_r.abs().sort_values(ascending=False).index).head(10))"""),
        md("""## Interpretation

CLR effects are relative log-contrasts, not abundance effects. Compare effect direction, rank, and inferential conclusions rather than expecting identical coefficients. Do not apply CLR to density values without a defensible compositional definition."""),
        code("""OUTPUT = ROOT / 'tutorial_outputs'
OUTPUT.mkdir(exist_ok=True)
ratio.to_csv(OUTPUT / 'spin_results_ratio.csv')
clr.to_csv(OUTPUT / 'spin_results_clr_sensitivity.csv')
comparison.to_csv(OUTPUT / 'ratio_clr_effect_comparison.csv')"""),
    ],
}


OUT.mkdir(exist_ok=True)
for filename, cells in notebooks.items():
    path = OUT / filename
    path.write_text(json.dumps(notebook(cells), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(path.relative_to(ROOT))
