"""Spatial and layer-label permutation tests for laminar maps."""

from itertools import permutations
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests


_LAYER_ALIASES = {
    'l1': 'Layer I', 'l2': 'Layer II', 'l3': 'Layer III',
    'l4': 'Layer IV', 'l5': 'Layer V', 'l6': 'Layer VI',
}


def filter_layer_inputs(features, thickness, present_mask=None,
                        exclude_layers=None):
    """Return aligned layer inputs after excluding named layers.

    Parameters
    ----------
    features : mapping of str to DataFrame
        Layer-keyed feature maps.
    thickness : DataFrame
        Layer thickness maps with either short (``l4``) or display
        (``Layer IV``) column labels.
    present_mask : DataFrame, optional
        Layer-by-cell-type inclusion mask.
    exclude_layers : iterable of str, optional
        Layer keys or display labels to remove, for example ``('l4',)`` or
        ``('Layer IV',)``.
    """
    excluded = set(exclude_layers or ())
    excluded |= {short for short, label in _LAYER_ALIASES.items()
                 if label in excluded}
    excluded |= {label for short, label in _LAYER_ALIASES.items()
                 if short in excluded}
    kept = [layer for layer in features
            if layer not in excluded and _LAYER_ALIASES.get(layer) not in excluded]
    if len(kept) < 2:
        raise ValueError("At least two layers must remain after exclusion")
    unknown = set(exclude_layers or ()) - {
        value for layer in features for value in (layer, _LAYER_ALIASES.get(layer, layer))
    }
    if unknown:
        raise ValueError(f"Unknown layers in exclude_layers: {sorted(unknown)}")
    filtered_features = {layer: features[layer] for layer in kept}
    columns = []
    for layer in kept:
        if layer in thickness.columns:
            columns.append(layer)
        elif _LAYER_ALIASES.get(layer) in thickness.columns:
            columns.append(_LAYER_ALIASES[layer])
        else:
            raise KeyError(f"Thickness is missing layer {layer!r}")
    filtered_thickness = thickness.loc[:, columns].copy()
    filtered_mask = None
    if present_mask is not None:
        rows = [layer if layer in present_mask.index else _LAYER_ALIASES.get(layer, layer)
                for layer in kept]
        filtered_mask = present_mask.loc[rows].copy()
    return filtered_features, filtered_thickness, filtered_mask


def layer_spin_correlation(features, thickness, spinner, present_mask=None,
                           metric='pearsonr', correction='fdr_bh', n_jobs=1):
    """Test each layer/cell-type map against matching-layer thickness."""
    if metric not in {'pearson', 'pearsonr', 'spearman', 'spearmanr'}:
        raise ValueError("metric must be pearsonr or spearmanr")
    tasks = []
    for layer, frame in features.items():
        target = _target(thickness, layer)
        for cell_type in frame.columns:
            if present_mask is not None and not present_mask.loc[
                    _mask_layer(layer, present_mask), cell_type]:
                continue
            tasks.append((layer, cell_type, frame[cell_type], target))

    def calculate(task):
        layer, cell_type, x, y = task
        r, p = spinner.correlation(x, y, metric=metric)
        return layer, cell_type, r, p

    rows = Parallel(n_jobs=n_jobs)(delayed(calculate)(task) for task in tasks)
    result = pd.DataFrame(rows, columns=['layer', 'ctype', 'correlation', 'p_value'])
    valid = result.p_value.notna()
    result['reject_H0'] = False
    result['p_adjusted'] = np.nan
    if valid.any():
        reject, adjusted, _, _ = multipletests(
            result.loc[valid, 'p_value'], method=correction)
        result.loc[valid, 'reject_H0'] = reject
        result.loc[valid, 'p_adjusted'] = adjusted
    result.attrs['correction'] = correction
    return result


def layer_match_permutation(features, thickness, scheme='whole',
                            alternative='greater', correction='fdr_bh',
                            random_state=None, n_permutations=None):
    """Test whether cell-type layer labels specifically match thickness layers."""
    layers = list(features)
    if len(layers) != thickness.shape[1]:
        raise ValueError("features and thickness must contain the same number of layers")
    permutations_all = list(permutations(range(len(layers))))
    if n_permutations is not None and n_permutations < len(permutations_all):
        rng = np.random.default_rng(random_state)
        chosen = rng.choice(len(permutations_all), n_permutations, replace=False)
        permutations_all = [permutations_all[i] for i in chosen]
    if scheme == 'whole':
        observed = _mean_match(features, thickness, tuple(range(len(layers))))
        null = np.array([_mean_match(features, thickness, p) for p in permutations_all])
        finite = null[np.isfinite(null)]
        if alternative == 'greater':
            extreme = finite >= observed
        elif alternative == 'less':
            extreme = finite <= observed
        elif alternative == 'two-sided':
            extreme = np.abs(finite) >= abs(observed)
        else:
            raise ValueError("alternative must be greater, less, or two-sided")
        # Exact enumeration includes the identity permutation, hence no +1.
        exact_p = float(extreme.sum() / finite.size) if finite.size else np.nan
        return {'observed_stat': observed, 'p_value': exact_p,
                'null_distribution': null, 'n_permutations': len(null)}
    if scheme != 'mismatch':
        raise ValueError("scheme must be 'whole' or 'mismatch'")
    rows = []
    for i, layer in enumerate(layers):
        for cell_type in features[layer].columns:
            x = features[layer][cell_type]
            observed = _safe_corr(x, thickness.iloc[:, i])
            null = np.array([_safe_corr(x, thickness.iloc[:, p[i]])
                             for p in permutations_all if p[i] != i])
            rows.append((layer, cell_type, observed,
                         _empirical_p(null, observed, alternative), len(null)))
    result = pd.DataFrame(rows, columns=['layer', 'ctype', 'r_obs', 'p_value', 'n_perm_used'])
    valid = result.p_value.notna()
    result['p_adjusted'] = np.nan
    result['reject_H0'] = False
    if valid.any():
        reject, adjusted, _, _ = multipletests(result.loc[valid, 'p_value'], method=correction)
        result.loc[valid, 'reject_H0'] = reject
        result.loc[valid, 'p_adjusted'] = adjusted
    return result


def _mean_match(features, thickness, permutation):
    values = []
    for i, layer in enumerate(features):
        target = thickness.iloc[:, permutation[i]]
        values.extend(_safe_corr(features[layer][c], target) for c in features[layer].columns)
    return float(np.nanmean(values))


def _safe_corr(x, y):
    x, y = pd.Series(x).align(pd.Series(y), join='inner')
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return np.nan
    return pearsonr(x[mask], y[mask])[0]


def _empirical_p(null, observed, alternative):
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or null.size == 0:
        return np.nan
    if alternative == 'greater':
        extreme = null >= observed
    elif alternative == 'less':
        extreme = null <= observed
    elif alternative == 'two-sided':
        extreme = np.abs(null) >= abs(observed)
    else:
        raise ValueError("alternative must be greater, less, or two-sided")
    return (int(extreme.sum()) + 1) / (null.size + 1)


def _target(thickness, layer):
    if layer in thickness.columns:
        return thickness[layer]
    return thickness[_LAYER_ALIASES.get(layer, layer)]


def _mask_layer(layer, mask):
    if layer in mask.index:
        return layer
    return _LAYER_ALIASES.get(layer, layer)
