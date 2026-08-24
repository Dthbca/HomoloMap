"""Add concise, reproducible visual summaries to the public tutorials."""

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
TUTORIALS = ROOT / "tutorials"


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def markdown(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def update(name, additions):
    path = TUTORIALS / name
    notebook = json.loads(path.read_text(encoding="utf-8"))
    marker = "tutorial-visual-summary"
    notebook["cells"] = [
        cell for cell in notebook["cells"]
        if marker not in "".join(cell.get("source", []))
    ]
    notebook["cells"].extend(additions)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


update("01_prepare_brain_idps.ipynb", [
    markdown("<!-- tutorial-visual-summary -->\n### Visual quality control\nThe upper panel shows the supplied brain IDPs across ordered BN labels; the lower panel confirms compositional closure."),
    code("""
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                         gridspec_kw={'height_ratios': [3, 1]})
Y.plot(ax=axes[0], linewidth=1.5)
axes[0].set(ylabel='IDP value', title='Brain IDPs in BN regional order')
axes[0].legend(frameon=False, ncol=min(4, Y.shape[1]))
axes[1].plot(X.index, X.sum(axis=1), color='#2a9d8f', linewidth=1.5)
axes[1].axhline(1, color='0.25', linestyle='--', linewidth=0.8)
axes[1].set(xlabel='BN region label', ylabel='Row sum', ylim=(0.98, 1.02))
sns.despine()
fig.tight_layout()
"""),
])

update("02_spin_test.ipynb", [
    markdown("<!-- tutorial-visual-summary -->\n### Visualize spatial associations\nBars show spatially corrected correlations for one IDP. Filled circles mark associations passing the selected FDR threshold."),
    code("""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plot_data = spin[[r_col, q_col]].sort_values(r_col)
colors = np.where(plot_data[q_col] < 0.05, '#d1495b', '#9db7c4')
fig, ax = plt.subplots(figsize=(7, max(4, 0.22 * len(plot_data))))
ax.barh(plot_data.index, plot_data[r_col], color=colors, edgecolor='none')
ax.axvline(0, color='0.25', linewidth=0.8)
ax.set(xlabel='Pearson correlation', ylabel='Cell type',
       title=f'Spin-test associations with {idp}')
sns.despine()
fig.tight_layout()
"""),
])

update("03_total_contribution_and_shap.ipynb", [
    markdown("<!-- tutorial-visual-summary -->\n### Visualize total and individual contributions\nThe first panel summarizes joint model performance. When SHAP is enabled, the second panel ranks individual cell-type contributions."),
    code("""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

numeric = total.select_dtypes('number')
score_col = next((c for c in numeric.columns if 'r2' in c.lower() or 'r_sq' in c.lower()), numeric.columns[0])
fig, ax = plt.subplots(figsize=(7, 3.5))
numeric[score_col].sort_values().plot.barh(ax=ax, color='#457b9d')
ax.set(xlabel=score_col, ylabel='Brain IDP', title='Total cell-type contribution')
sns.despine()
fig.tight_layout()

if RUN_SHAP:
    contribution = first['individual_ctype_contribution']
    if isinstance(contribution, pd.DataFrame):
        contribution = contribution.select_dtypes('number').iloc[:, 0]
    contribution = contribution.sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    contribution.plot.barh(ax=ax, color='#e9a03b')
    ax.set(xlabel='Mean absolute SHAP contribution', ylabel='Cell type',
           title=f'Individual contributions to {Y.columns[0]}')
    sns.despine()
    fig.tight_layout()
"""),
])

update("04_clr_sensitivity.ipynb", [
    markdown("<!-- tutorial-visual-summary -->\n### Visualize CLR sensitivity\nPoints near the diagonal retain a similar effect estimate after the centered log-ratio transformation."),
    code("""
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(5, 5))
sns.scatterplot(data=comparison, x='ratio_r', y='clr_r', s=45, ax=ax,
                color='#5b8e7d')
limits = [comparison.min().min(), comparison.max().max()]
ax.plot(limits, limits, '--', color='0.35', linewidth=0.9)
for label in comparison.assign(delta=lambda d: (d.clr_r-d.ratio_r).abs()).nlargest(5, 'delta').index:
    ax.annotate(label, comparison.loc[label], xytext=(3, 3),
                textcoords='offset points', fontsize=8)
ax.set(title='Ratio versus CLR sensitivity', aspect='equal')
sns.despine()
fig.tight_layout()
"""),
])
