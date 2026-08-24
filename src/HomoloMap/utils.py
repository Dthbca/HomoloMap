import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass

from statsmodels.stats.multitest import multipletests
from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt

from neuromaps import stats

from joblib import Parallel, delayed
from tqdm import tqdm


@dataclass
class LayerAnalysisData:
    proportions: dict
    counts: dict
    thickness: pd.DataFrame
    present_mask: pd.DataFrame
    mapping: dict
    metadata: dict


def prepare_layer_analysis(atlas='BN', mapping_column='subclass',
                           data_dir=None, mask_kind='external',
                           mask_threshold=0.05, composition='clr',
                           thickness_relative=True, unmapped='drop',
                           normalization='within_layer', zero_policy='nan',
                           zero_method='multiplicative', invalid_rows='drop',
                           normalization_order='before_relabel', reclose=True):
    """Prepare aligned laminar cell-type maps and BigBrain thickness."""
    from HomoloMap.datasets import (
        load_layer_counts, relabel_layer_counts, normalize_layer_composition,
        fetch_bigbrain_layer_thickness, fetch_laminar_mask)
    from HomoloMap.transforms import make_layer_subcompositions
    if normalization_order not in {'before_relabel', 'after_relabel'}:
        raise ValueError(
            "normalization_order must be 'before_relabel' or 'after_relabel'"
        )
    normalization_mode = (
        'within_region' if normalization == 'within_region_cross_layer'
        else normalization
    )
    raw, mapping = load_layer_counts(
        data_dir=data_dir, mapping_column=mapping_column,
        unmapped=unmapped, return_mapping=True)
    ratio_input = (
        normalize_layer_composition(
            raw, mode=normalization_mode, zero_policy=zero_policy)
        if normalization_order == 'before_relabel' else raw
    )
    proportions, ratio_relabel_audit = relabel_layer_counts(
        ratio_input, target_atlas=atlas, method='mean', return_audit=True)
    counts, relabel_audit = relabel_layer_counts(
        raw, target_atlas=atlas, method='sum', return_audit=True)
    if normalization_order == 'after_relabel' or reclose:
        proportions = normalize_layer_composition(
            proportions, mode=normalization_mode, zero_policy=zero_policy)
    cell_types = next(iter(proportions.values())).columns
    present_mask, mask_audit = fetch_laminar_mask(
        kind=mask_kind, cell_types=cell_types, threshold=mask_threshold,
        counts=counts, data_dir=data_dir)
    features, composition_audit = make_layer_subcompositions(
        proportions, present_mask=present_mask, transform=composition,
        zero_method=zero_method, invalid_rows=invalid_rows)
    regions = next(iter(features.values())).index
    thickness = fetch_bigbrain_layer_thickness(
        atlas=atlas, data_dir=data_dir, relative=thickness_relative,
        regions=regions)
    metadata = {'atlas': atlas, 'normalization': normalization,
                'normalization_order': normalization_order,
                'reclosed_after_relabel': bool(reclose),
                'zero_policy': zero_policy, 'invalid_rows': invalid_rows,
                'composition': composition, 'thickness_relative': thickness_relative,
                'mask': mask_audit, 'relabel': relabel_audit,
                'ratio_relabel': ratio_relabel_audit,
                'composition_by_layer': composition_audit}
    return LayerAnalysisData(features, counts, thickness, present_mask, mapping, metadata)


def run_layer_analysis(atlas='BN', mapping_column='subclass', data_dir=None,
                       mask_kind='external', mask_threshold=0.05,
                       composition='clr', normalization='within_layer',
                       zero_policy='nan', invalid_rows='drop',
                       thickness_relative=True, unmapped='drop', n_spins=1000,
                       spin_method='Alexander-Bloch', metric='pearsonr',
                       correction='fdr_bh', permutation_scheme='mismatch',
                       n_permutations=None, random_state=42, n_jobs=1,
                       exclude_layers=None,
                       normalization_order='before_relabel', reclose=True):
    """One-stop laminar spatial-coupling and layer-specificity analysis."""
    from HomoloMap.stats import (SpinTest, filter_layer_inputs,
                                 layer_spin_correlation, layer_match_permutation)
    prepared = prepare_layer_analysis(
        atlas=atlas, mapping_column=mapping_column, data_dir=data_dir,
        mask_kind=mask_kind, mask_threshold=mask_threshold,
        composition=composition, thickness_relative=thickness_relative,
        unmapped=unmapped, normalization=normalization,
        zero_policy=zero_policy, invalid_rows=invalid_rows,
        normalization_order=normalization_order, reclose=reclose)
    features, thickness, present_mask = filter_layer_inputs(
        prepared.proportions, prepared.thickness, prepared.present_mask,
        exclude_layers=exclude_layers)
    prepared.proportions = features
    prepared.thickness = thickness
    prepared.present_mask = present_mask
    prepared.metadata['exclude_layers'] = list(exclude_layers or ())
    spinner = SpinTest(atlas=atlas, n_spins=n_spins,
                       method=spin_method, seed=random_state)
    spin = layer_spin_correlation(
        features, thickness, spinner,
        present_mask=present_mask, metric=metric,
        correction=correction, n_jobs=n_jobs)
    match = layer_match_permutation(
        features, thickness, scheme=permutation_scheme,
        correction=correction, random_state=random_state,
        n_permutations=n_permutations)
    return {'data': prepared, 'spin_test': spin,
            'layer_match': match, 'parameters': prepared.metadata}

def inverse(A, k=1, normalize=False):
    '''
    Function that returns the original matrix with the inverse values at
    non-zero indices.

    Parameters
    ----------
    A : (n, n) ndarray
        Matrix for which we want to compute inverse values.
    normalize: bool
        If `True`, then each row of the inverse matrix will be normalize such
        that the values in the row sum to 1.

    Returns
    -------
    w: (n, n) ndarray
        Matrix where each non-zero entries correspond to the inverse of the
        non-zero entries in matrix `A`
    '''

    A = np.array(A, dtype=float, copy=True)  # don't mutate the caller's matrix
    np.fill_diagonal(A, np.inf)
    inv_dist = 1 / A  # Compute the inverse of distances
    if normalize:
        inv_dist /= inv_dist.sum(axis=1, keepdims=True)  # Normalize by row sums
    return inv_dist


def morans_i(dist, y, normalize=False, local=False, invert_dist=False):
    """
    Calculates Moran's I from distance matrix `dist` and brain map `y`

    Parameters
    ----------
    dist : (N, N) array_like
        Distance matrix between `N` regions / vertices / voxels / whatever
    y : (N,) array_like
        Brain map variable of interest
    normalize : bool, optional
        Whether to normalize rows of distance matrix prior to calculation.
        Default: False
    local : bool, optional
        Whether to calculate local Moran's I instead of global. Default: False
    invert_dist : bool, optional
        Whether to invert the distance matrix to generate a weight matrix.
        Default: True
    Returns
    -------
    i : float
        Moran's I, measure of spatial autocorrelation
    """
    mask = ~np.isnan(y)
    y = y[mask]
    dist = dist[mask][:, mask]
    # convert distance matrix to weights
    if invert_dist:
        with np.errstate(divide='ignore'):
            dist = 1 / dist
    np.fill_diagonal(dist, 0)

    # normalize rows, if desired
    if normalize:
        dist /= dist.sum(axis=-1, keepdims=True)

    # calculate Moran's I
    z = y - y.mean()
    if local:
        with np.errstate(all='ignore'):
            z /= y.std()

    zl = np.squeeze(dist @ z[:, None])
    den = (z * z).sum()

    if local:
        return (len(y) - 1) * z * zl / den

    return len(y) / dist.sum() * (z * zl).sum() / den


def _standardize_analysis_array(values, name):
    """Z-score a 1D/2D array and reject constant or non-finite columns."""
    standardized = zscore(np.asarray(values, dtype=float), axis=0)
    if not np.isfinite(standardized).all():
        raise ValueError(f"{name} contains missing, non-finite, or constant values")
    return standardized


def prepare_analysis_data(data=None, feature_type='ratio', ctype_level='subclass',
                          layer=False, mask=True, smooth=None, atlas='BN',
                          mapping_column=None, unmapped='drop', renormalize=False,
                          return_mapping=False, layer_data_dir=None,
                          layer_normalization='within_layer',
                          layer_normalization_order='before_relabel',
                          layer_reclose=True):
    """Load, left-hemisphere filter, reparcellate, and align analysis tables."""
    from HomoloMap.datasets import (
        fetch_ctype_ratio, fetch_layer_ratio, fetch_ctype_density
    )
    from HomoloMap.transforms.parcellation import vol_relabel

    if feature_type not in {'ratio', 'density'}:
        raise ValueError("feature_type must be 'ratio' or 'density'")
    if ctype_level not in {'subclass', 'cluster'}:
        raise ValueError("ctype_level must be 'subclass' or 'cluster'")
    if layer:
        predictors, mapping = fetch_layer_ratio(
            level=ctype_level, mask=mask, mapping_column=mapping_column,
            unmapped=unmapped, return_mapping=True,
            data_dir=layer_data_dir, target_atlas=atlas,
            normalization=layer_normalization,
            normalization_order=layer_normalization_order,
            reclose=layer_reclose)
        predictors = predictors.fillna(0)
    elif feature_type == 'ratio':
        predictors, mapping = fetch_ctype_ratio(
            level=ctype_level, smooth=smooth, mapping_column=mapping_column,
            unmapped=unmapped, renormalize=renormalize, return_mapping=True,
            atlas=atlas)
    else:
        predictors, mapping = fetch_ctype_density(
            level=ctype_level, smooth=smooth, mapping_column=mapping_column,
            unmapped=unmapped, renormalize=renormalize, return_mapping=True,
            atlas='D99')
    if feature_type == 'density' and atlas != 'D99':
        predictors = vol_relabel(
            src='D99', trg=atlas, data=predictors,
            cross_species=True, method='mean'
        )
    if data is None:
        raise ValueError(
            "data is required: provide brain IDPs indexed by the requested "
            "atlas labels"
        )
    outcomes = data
    if isinstance(outcomes, pd.Series):
        outcomes = outcomes.to_frame(name=outcomes.name or 'feature')
    if not isinstance(outcomes, pd.DataFrame) or outcomes.empty:
        raise TypeError("data must be a non-empty pandas DataFrame or Series")
    if not predictors.index.is_unique or not outcomes.index.is_unique:
        raise ValueError("Predictor and phenotype ROI labels must be unique")
    outcomes = outcomes.copy().apply(pd.to_numeric, errors='coerce')
    shared = predictors.index.intersection(outcomes.index, sort=False)
    if len(shared) < 3:
        raise ValueError(f"At least 3 shared left-hemisphere ROIs are required; found {len(shared)}")
    X = predictors.loc[shared].astype(float)
    Y = outcomes.loc[shared].astype(float)
    valid = X.notna().all(axis=1) & Y.notna().any(axis=1)
    X, Y = X.loc[valid], Y.loc[valid]
    mapping = dict(mapping)
    mapping['spatial_mapping'] = {
        'source_space': 'macaque_spatial_transcriptomics',
        'target_space': f'human_{atlas}_left_hemisphere',
        'input_is_precomputed_human_map': feature_type == 'ratio' and atlas == 'BN',
    }
    X.attrs['celltype_mapping'] = mapping
    return (X, Y, mapping) if return_mapping else (X, Y)


def run_spin_correlations(X, Y, spinner, metric='spearmanr', FDR='fdr_bh', n_jobs=-1,
                          composition_transform='none', composition_params=None):
    """Run cell-type/phenotype spin correlations on aligned left-hemi tables."""
    from HomoloMap.transforms import transform_composition
    composition_params = {} if composition_params is None else dict(composition_params)
    X, transform_meta = transform_composition(
        X, method=composition_transform, **composition_params)
    X = X.reindex(spinner.labels)
    Y = Y.reindex(spinner.labels)
    correlation = pd.DataFrame(index=X.columns)
    for outcome in tqdm(Y.columns):
        values = Parallel(n_jobs=n_jobs)(
            delayed(spinner.correlation)(X[feature], Y[outcome], metric=metric)
            for feature in X.columns
        )
        r_values, p_values = map(list, zip(*values))
        correlation[f'{outcome}_ratio_spin_r'] = r_values
        correlation[f'{outcome}_ratio_spin_p'] = p_values
        correlation[f'{outcome}_ratio_spin_p_adj'] = multipletests(
            p_values, method=FDR
        )[1]
    correlation.attrs['composition_transform'] = transform_meta
    return correlation


def run_cumulative_models(X, Y, spinner, mode='linear', n_spins=1000,
                          FDR='fdr_bh', n_jobs=-1, model_kwargs=None,
                          composition_transform='none', composition_params=None):
    """Run multivariable models and spatial permutation inference."""
    from HomoloMap.stats import get_reg_r_pval, get_reg_r_sq
    from HomoloMap.transforms import transform_composition
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    if mode == 'random_forest' and n_jobs != 1:
        model_kwargs.setdefault('n_jobs', 1)
    if len(X) != len(spinner.labels) or set(X.index) != set(spinner.labels):
        raise ValueError("Cumulative models require the complete atlas in label order")
    X = X.reindex(spinner.labels)
    Y = Y.reindex(spinner.labels)
    composition_params = {} if composition_params is None else dict(composition_params)
    X, transform_meta = transform_composition(
        X, method=composition_transform, **composition_params)
    X_values = _standardize_analysis_array(X.values, 'Cell-type predictors')
    rows = []
    for outcome in Y.columns:
        try:
            y = _standardize_analysis_array(Y[outcome].values, outcome)
        except ValueError as exc:
            warnings.warn(str(exc))
            rows.append((outcome, np.nan, np.nan))
            continue
        r2 = get_reg_r_sq(X_values, y, model_type=mode, **model_kwargs)
        p = get_reg_r_pval(
            X_values, y, spinner.spins, n_spins, model_type=mode,
            n_jobs=n_jobs, verbose=False, **model_kwargs
        )
        rows.append((outcome, r2, p))
    result = pd.DataFrame(rows, columns=['disease', 'model_r_sq', 'model_pval']).set_index('disease')
    finite = result.model_pval.notna()
    result['model_pval_adj'] = np.nan
    if finite.any():
        result.loc[finite, 'model_pval_adj'] = multipletests(
            result.loc[finite, 'model_pval'], method=FDR
        )[1]
    result.attrs['composition_transform'] = transform_meta
    return result


def run_explanation_analysis(X, Y, method='shap', mode='linear', n_jobs=-1,
                             random_state=1234, shap_params=None, model_kwargs=None,
                             composition_transform='none', composition_params=None):
    """Run SHAP or linear dominance analysis independently."""
    from HomoloMap.stats import get_shap_stats, get_dominance_stats
    from HomoloMap.transforms import transform_composition
    if method not in {'shap', 'dominance'}:
        raise ValueError("method must be 'shap' or 'dominance'")
    if method == 'dominance' and mode != 'linear':
        raise ValueError("Dominance analysis is defined here only for mode='linear'")
    shap_params = {} if shap_params is None else dict(shap_params)
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    composition_params = {} if composition_params is None else dict(composition_params)
    X, transform_meta = transform_composition(
        X, method=composition_transform, **composition_params)
    if method == 'shap' and composition_transform == 'ilr':
        warnings.warn(
            "ILR SHAP values describe balances, not individual cell types.",
            UserWarning,
        )
    X_values = _standardize_analysis_array(X.values, 'Cell-type predictors')
    output = {}
    for outcome in Y.columns:
        try:
            y = _standardize_analysis_array(Y[outcome].values, outcome)
        except ValueError as exc:
            warnings.warn(str(exc))
            continue
        if method == 'shap':
            summary, values = get_shap_stats(
                X_values, y, model_type=mode, feature_names=X.columns.tolist(),
                random_state=random_state, model_kwargs=model_kwargs, **shap_params
            )
            output[outcome] = {
                'summary': summary, 'shap_values': values,
                'total_mean_absolute_shap': summary.attrs['total_mean_absolute_shap'],
                'total_contribution': summary.attrs['total_contribution'],
                'individual_ctype_contribution': summary,
            }
        else:
            output[outcome] = get_dominance_stats(X_values, y, n_jobs=n_jobs)
    for values in output.values():
        if isinstance(values, dict):
            values['composition_transform'] = transform_meta
    return output


def run_analysis(
    data = None,
    feature_type = 'ratio',
    ctype_level='subclass',
    layer = False,
    mask = True,
    smooth=True,
    smooth_params=None,
    atlas='FGC',
    spin_method = 'Alexander-Bloch',
    metric = 'spearmanr',
    n_spins=1000,
    FDR = 'fdr_bh',
    cumulative=True,
    mode = 'linear',
    explanations = None,
    n_jobs = -1,
    random_state=1234,
    make_plot=False,
    shap_params=None,
    model_kwargs=None,
    mapping_column=None,
    unmapped='drop',
    renormalize=False,
    correlation_transform='none',
    cumulative_transform='none',
    explanation_transform='none',
    zero_method='multiplicative',
    zero_fraction=0.65,
    min_mapping_coverage=None,
):
    """
    Run cell type spatial pattern analysis.
    Parameters
    ----------      
    data : pd.DataFrame, optional
        DataFrame containing brain maps IDPs. Rows should correspond
        to brain regions and columns to features. If None, ENIGMA
        data will be loaded. Default: None.
    feature_type : str, optional
        Type of cell type feature to use. Options are 'ratio' or 'density'.
        Default: 'ratio'.
    ctype_level : str, optional
        Level of cell type classification to use. Options are 'subclass' or
        'cluster'. Default: 'subclass'.
    layer : bool, optional
        Whether to use layer-specific cell type data. Default: False.
    mask : bool, optional
        Whether to apply a mask to the layer-specific data. Default: True.
    smooth : bool, optional
        Whether to smooth the cell type data. Default: True.
    smooth_params : dict, optional
        Parameters for smoothing the cell type data. Should contain keys
        'method', 'radius', and 'roi_disc'. Default: {'method':'mean',
        'radius':10,'roi_disc':None}.
    atlas : str, optional
        Brain atlas to use. Default: 'FGC'.
    spin_method : str, optional
        Method for generating spin permutations. Default: 'original'.
    n_spins : int, optional
        Number of spin permutations to generate. Default: 1000. 
    metric: str, optional
        Metric for calculating correlation. Options are 'pearson', 'spearmanr',
    FDR : str, optional 
        Method for FDR correction. Default: 'fdr_bh'.
    cumulative : bool, optional
        Whether to calculate the cumulative effects. Default: True.
    mode : str, optional
        Type of regression model to use. Options are 'linear' or 'ridge'.
        Default: 'linear'.      
    explanations : str, optional
        Type of model explanation to perform. Options are 'shap' or
        'dominance'. Default: None.
    n_jobs : int, optional
        Number of parallel jobs to run. Default: -1 (use all available cores).
    Returns
    -------
    result : dict
        Dictionary containing analysis results, including:
        - 'features': list of cell type features used
        - 'X': DataFrame of cell type features
        - 'Y': DataFrame of IDPs brain maps
        - 'correlation': DataFrame of spin test correlations
        - 'cumulative_effects': DataFrame of cumulative effect results
        - 'explanations': SHAP or dominance results keyed by phenotype
          (if requested)
    """
    from HomoloMap.stats.nulls import SpinTest

    if feature_type not in {'ratio', 'density'}:
        raise ValueError("feature_type must be 'ratio' or 'density'")
    if ctype_level not in {'subclass', 'cluster'}:
        raise ValueError("ctype_level must be 'subclass' or 'cluster'")
    if mode not in {'linear', 'random_forest', 'svr'}:
        raise ValueError("mode must be 'linear', 'random_forest', or 'svr'")
    if explanations not in {None, 'shap', 'dominance'}:
        raise ValueError("explanations must be 'shap', 'dominance', or None")
    if explanations == 'dominance' and mode != 'linear':
        raise ValueError("Dominance analysis is supported only for mode='linear'")
    if not isinstance(n_spins, (int, np.integer)) or isinstance(n_spins, bool) or n_spins <= 0:
        raise ValueError("n_spins must be a positive integer")
    if n_jobs == 0:
        raise ValueError("n_jobs must not be zero")
    if unmapped not in {'raise', 'drop', 'keep'}:
        raise ValueError("unmapped must be 'raise', 'drop', or 'keep'")
    transforms = {correlation_transform, cumulative_transform, explanation_transform}
    if not transforms.issubset({'none', 'clr', 'ilr'}):
        raise ValueError("composition transforms must be 'none', 'clr', or 'ilr'")
    if feature_type == 'density' and transforms != {'none'}:
        raise ValueError("Log-ratio transforms require feature_type='ratio'")
    smooth_params = {} if smooth_params is None else dict(smooth_params)
    if smooth_params:
        warnings.warn(
            "smooth_params is deprecated and currently ignored; smooth selects "
            "the precomputed left-hemisphere cell-type dataset.",
            DeprecationWarning,
            stacklevel=2,
        )
    shap_params = {} if shap_params is None else dict(shap_params)
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    result = {'parameters': {
        'atlas': atlas, 'feature_type': feature_type, 'ctype_level': ctype_level,
        'n_spins': n_spins, 'spin_method': spin_method, 'metric': metric,
        'mode': mode, 'random_state': random_state,
        'mapping_column': mapping_column or ctype_level,
        'unmapped': unmapped, 'renormalize': bool(renormalize),
        'correlation_transform': correlation_transform,
        'cumulative_transform': cumulative_transform,
        'explanation_transform': explanation_transform,
    }}

    # Step 1-2: prepare aligned left-hemisphere data
    print("Loading cell type ratio data...")
    X_df, Y_df, mapping = prepare_analysis_data(
        data=data, feature_type=feature_type, ctype_level=ctype_level,
        layer=layer, mask=mask, smooth=smooth, atlas=atlas,
        mapping_column=mapping_column, unmapped=unmapped,
        renormalize=renormalize, return_mapping=True
    )
    result['features'] = X_df.columns.tolist()
    result['X'], result['Y'] = X_df, Y_df
    result['roi_labels'] = X_df.index.to_numpy(copy=True)
    result['celltype_mapping'] = mapping
    composition_params = {
        'zero_method': zero_method,
        'zero_fraction': zero_fraction,
        'min_mapping_coverage': min_mapping_coverage,
        'mapping_coverage': mapping['mapping_coverage'],
    }
    # Step 3: 计算旋转测试
    print("Performing spin tests...")
    spinner = SpinTest(atlas=atlas, n_spins=n_spins, method=spin_method, seed=random_state)

    result['correlation'] = run_spin_correlations(
        X_df, Y_df, spinner, metric=metric, FDR=FDR, n_jobs=n_jobs,
        composition_transform=correlation_transform,
        composition_params=composition_params,
    )

    # Step 4: 累积效应
    if cumulative:
        print("Calculating cumulative effects...")
        if len(X_df) != spinner.spins.shape[0] or not np.array_equal(
            X_df.index.to_numpy(), spinner.labels
        ):
            raise ValueError(
                "Cumulative spin regression requires one row per atlas parcel "
                "in atlas-label order; use cumulative=False for partial ROI data"
            )
    
        result['cumulative_effects'] = run_cumulative_models(
            X_df, Y_df, spinner, mode=mode, n_spins=n_spins,
            FDR=FDR, n_jobs=n_jobs, model_kwargs=model_kwargs,
            composition_transform=cumulative_transform,
            composition_params=composition_params,
        )

        if make_plot:
            plt.ion()
            fig = plt.figure()
            disorders = Y_df.columns
            cumulative_df = result['cumulative_effects']
            bars = plt.barh(np.arange(len(disorders)), cumulative_df.model_r_sq,
                tick_label=disorders)
            for i, (p_val, bar) in enumerate(zip(cumulative_df.model_pval_adj, bars)):
                if not np.isfinite(p_val) or p_val >= 0.05:
                    continue
                width = bar.get_width()
                plt.text(width+0.02, bar.get_y(), '*', ha='center', va='bottom',
                         color='red', fontweight='bold', fontsize=12)
            plt.xticks(rotation='vertical')
            sns.despine(top=True, right=True)
            plt.tight_layout()
            result['cumulative_figure'] = fig

    # Step 5: SHAP 分析
    if explanations:
        if explanations not in {'shap', 'dominance'}:
            raise ValueError("explanations must be 'shap', 'dominance', or None")
        print(f"Performing {explanations} analysis...")
        explanation_results = run_explanation_analysis(
            X_df, Y_df, method=explanations, mode=mode, n_jobs=n_jobs,
            random_state=random_state, shap_params=shap_params,
            model_kwargs=model_kwargs,
            composition_transform=explanation_transform,
            composition_params=composition_params,
        )
        result['explanations'] = explanation_results
        if explanations == 'shap':
            result['shap_analysis'] = explanation_results

    result['composition'] = {
        'correlation': result['correlation'].attrs.get('composition_transform'),
        'cumulative': result.get('cumulative_effects', pd.DataFrame()).attrs.get(
            'composition_transform'),
        'explanation_method': explanation_transform if explanations else None,
    }

    print("Analysis complete.")
    return result
