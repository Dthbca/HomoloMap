"""Plotting helpers for cortical-layer analyses."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_layer_heatmap(data, pvalues=None, present_mask=None, layer_labels=None,
                       significance=(0.05, 0.01, 0.001), ax=None,
                       output=None, show=False, cmap='RdYlBu_r',
                       vmin=-1, vmax=1, title=None):
    """Plot a layer-by-cell-type heatmap; present_mask True means visible."""
    values = pd.DataFrame(data).copy()
    if layer_labels is not None:
        if len(layer_labels) != len(values):
            raise ValueError("layer_labels length must match data rows")
        values.index = layer_labels
    if present_mask is not None:
        visible = present_mask.reindex(index=values.index, columns=values.columns)
        values = values.mask(~visible.fillna(False))
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, values.shape[1] * .28), 3.2))
    sns.heatmap(values, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                mask=values.isna(), linewidths=.25, linecolor='white')
    if pvalues is not None:
        pvalues = pvalues.reindex_like(values)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                p = pvalues.iat[i, j]
                if not np.isfinite(p) or not np.isfinite(values.iat[i, j]):
                    continue
                stars = sum(p < threshold for threshold in significance)
                if stars:
                    ax.text(j + .5, i + .5, '*' * stars,
                            ha='center', va='center', fontsize=7)
    ax.set_xlabel('Cell type')
    ax.set_ylabel('Cortical layer')
    if title:
        ax.set_title(title)
    if output:
        ax.figure.savefig(output, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return ax
