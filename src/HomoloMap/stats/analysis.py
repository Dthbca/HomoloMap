# HomoloMap/stats/analysis.py
"""
Statistical analysis and model interpretation tools.

This module provides regression models, feature importance analysis,
and model explainability methods for brain data.

Optimizations:
- Parallelized p-value computation for speed
- Pre-scaled SVR for faster computation
- Progress bars for long operations
- Caching for repeated operations
"""

import warnings
import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, Dict, List
from itertools import combinations
from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# =============================================================================
# Model Configuration
# =============================================================================

def _get_model(model_type: str, **kwargs):
    """
    Get configured model instance.
    
    Parameters
    ----------
    model_type : str
        Model type
    **kwargs
        Additional model parameters
        
    Returns
    -------
    model : sklearn estimator
        Configured model
    """
    if model_type == 'linear':
        return LinearRegression(**kwargs)
    
    elif model_type == 'random_forest':
        default_params = {
            'random_state': 42,
            'n_estimators': 100,
            'n_jobs': -1,  # Parallel within RF
        }
        default_params.update(kwargs)
        return RandomForestRegressor(**default_params)
    
    elif model_type == 'svr':
        # CRITICAL: SVR is SLOW without proper configuration
        default_params = {
            'kernel': 'rbf',      # Default kernel
            'C': 1.0,             # Regularization
            'epsilon': 0.1,       # Epsilon-tube
            'cache_size': 1000,   # Increase cache (MB)
            'max_iter': 10000,    # Increase iterations if needed
        }
        default_params.update(kwargs)
        
        # WARNING: SVR does not have n_jobs parameter
        if 'n_jobs' in default_params:
            warnings.warn(
                "SVR does not support n_jobs parameter. "
                "Use parallel processing at a higher level instead.",
                UserWarning
            )
            default_params.pop('n_jobs')
        
        return SVR(**default_params)
    
    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. "
            f"Must be 'linear', 'random_forest', or 'svr'"
        )


# =============================================================================
# Regression Models
# =============================================================================

def _validate_regression_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return finite, shape-compatible regression arrays."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array; got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be a 1D array; got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must contain the same number of samples; "
            f"got {X.shape[0]} and {y.shape[0]}"
        )
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must contain at least one sample and one feature")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X and y must contain only finite values")
    return X, y


def _r_squared_from_predictions(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared with deterministic handling of constant targets."""
    ss_residual = float(np.sum((y - y_pred) ** 2))
    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    if np.isclose(ss_total, 0.0):
        return 1.0 if np.isclose(ss_residual, 0.0) else 0.0
    return 1.0 - ss_residual / ss_total


def _adjust_r_squared(r_squared: float, n_samples: int, n_features: int) -> float:
    """Adjust R-squared when sufficient residual degrees of freedom exist."""
    denom = n_samples - n_features - 1
    if denom <= 0:
        warnings.warn(
            f"Cannot compute adjusted R^2: n_samples={n_samples}, "
            f"n_features={n_features} (need n_samples - n_features - 1 > 0). "
            "Returning unadjusted R^2.",
            RuntimeWarning
        )
        return r_squared
    return 1 - (1 - r_squared) * (n_samples - 1) / denom

def get_reg_r_sq(
    X: np.ndarray,
    y: np.ndarray,
    adjust: bool = True,
    model_type: str = 'linear',
    scale_data: bool = None,
    **model_kwargs
) -> float:
    """
    Calculate R-squared or adjusted R-squared for a regression model.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input features (predictors)
    y : np.ndarray, shape (n_samples,)
        Target variable
    adjust : bool, default=True
        Whether to calculate adjusted R-squared.
        Adjusted R² accounts for the number of predictors.
    model_type : {'linear', 'random_forest', 'svr'}, default='linear'
        Type of regression model to fit:
        - 'linear': Linear regression (fastest)
        - 'random_forest': Random forest regressor
        - 'svr': Support vector regression (slowest, needs scaling)
    scale_data : bool, optional
        Whether to standardize X and y before fitting.
        If None, automatically scales for SVR, not for others.
        SVR requires scaling for good performance!
    **model_kwargs
        Additional parameters for the model
        
    Returns
    -------
    r_squared : float
        R-squared or adjusted R-squared value
        
    Notes
    -----
    **Performance Tips:**
    
    1. **Linear models** are fastest (~1ms)
    2. **Random Forest** is moderate (~100ms)
    3. **SVR** is slowest (~1-10s) and REQUIRES scaling
    
    For SVR:
    - Always use scaled data (scale_data=True)
    - Consider using LinearSVR for large datasets
    - Increase cache_size if memory allows
    - Use parallel processing at higher level, not within SVR
    
    Examples
    --------
    Fast linear regression:
    
    >>> from HomoloMap.stats import get_reg_r_sq
    >>> r2 = get_reg_r_sq(X, y, model_type='linear')
    
    SVR with proper scaling (IMPORTANT!):
    
    >>> # Automatic scaling (recommended)
    >>> r2 = get_reg_r_sq(X, y, model_type='svr', scale_data=True)
    >>> 
    >>> # Or manually scale beforehand
    >>> from sklearn.preprocessing import StandardScaler
    >>> scaler_X = StandardScaler()
    >>> scaler_y = StandardScaler()
    >>> X_scaled = scaler_X.fit_transform(X)
    >>> y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    >>> r2 = get_reg_r_sq(X_scaled, y_scaled, model_type='svr')
    
    Custom SVR parameters:
    
    >>> r2 = get_reg_r_sq(
    ...     X, y,
    ...     model_type='svr',
    ...     scale_data=True,
    ...     kernel='linear',     # Linear kernel is faster
    ...     C=10.0,              # Higher regularization
    ...     cache_size=2000      # More cache
    ... )
    
    See Also
    --------
    get_reg_r_pval : get p-value using spatial nulls
    fit_regression_model : High-level regression interface
    """
    X, y = _validate_regression_data(X, y)

    # Auto-determine scaling
    if scale_data is None:
        scale_data = (model_type == 'svr')
    
    # Scale data if requested
    if scale_data:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X = scaler_X.fit_transform(X)
        y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    # Get model
    model = _get_model(model_type, **model_kwargs)
    
    # Fit and predict
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Calculate R²
    r_squared = _r_squared_from_predictions(y, y_pred)
    
    # Adjust for number of predictors if requested
    if adjust:
        return _adjust_r_squared(r_squared, len(y), X.shape[1])

    return r_squared


def _get_reg_r_sq_single_permutation(X, y, spin_idx, adjust, model_type, scale_data, model_kwargs):
    """
    Helper function to get R² for a single permutation.
    
    This is separated out so it can be parallelized.
    """
    y_permuted = y[spin_idx]
    return get_reg_r_sq(
        X, y_permuted,
        adjust=adjust,
        model_type=model_type,
        scale_data=scale_data,
        **model_kwargs
    )


def get_reg_r_pval(
    X: np.ndarray,
    y: np.ndarray,
    spins: np.ndarray,
    n_spins: Optional[int] = None,
    model_type: str = 'linear',
    adjust: bool = True,
    scale_data: bool = None,
    n_jobs: int = 1,
    verbose: bool = True,
    return_null: bool = False,
    **model_kwargs
) -> float:
    """
    get p-value for regression model using spatial null distribution.
    
    Uses spin test permutations to generate a null distribution of R²
    values that preserve spatial autocorrelation structure.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input features (predictors)
    y : np.ndarray, shape (n_samples,)
        Target variable
    spins : np.ndarray, shape (n_samples, n_permutations)
        Spin test permutation indices
    n_spins : int, optional
        Number of permutations to use. If None, uses all columns in spins.
    model_type : str, default='linear'
        Type of regression model ('linear', 'random_forest', 'svr')
    adjust : bool, default=True
        Whether to use adjusted R²
    scale_data : bool, optional
        Whether to scale data. Auto-enabled for SVR.
    n_jobs : int, default=1
        Number of parallel jobs:
        - 1: Sequential (default, safest)
        - -1: Use all CPU cores
        - n > 1: Use n cores
        For SVR, parallel processing provides major speedup!
    verbose : bool, default=True
        Whether to show progress bar
    return_null : bool, default=False
        Whether to return the null distribution of R² values along with p-value
    **model_kwargs
        Additional model parameters
        
    Returns
    -------
    p_value : float
        Spin test p-value (one-tailed: fraction of the null with R^2 >= observed).
        R^2 is non-negative and only large values are evidence against the null,
        so a one-tailed test is the correct choice here.

    Notes
    -----
    **Performance Optimization:**
    
    For SVR models, this function can be VERY slow without parallelization:
    - Sequential: ~10 seconds per permutation × 1000 = 3 hours
    - Parallel (8 cores): ~1.5 seconds per permutation × 1000 = 25 minutes
    
    **Recommended Settings:**
    
    ```python
    # For SVR: Use parallel processing
    p_val = get_reg_r_pval(
        X, y, spins,
        model_type='svr',
        n_jobs=-1,           # Use all cores
        verbose=True,        # Show progress
        scale_data=True,     # Essential for SVR
        cache_size=2000      # Increase cache
    )
    ```
    
    Examples
    --------
    Linear regression (fast):
    
    >>> from HomoloMap.stats import get_reg_r_pval, spin_data
    >>> 
    >>> spins = spin_data(celltype_data, atlas='FGC', n_spins=1000)
    >>> p_val = get_reg_r_pval(
    ...     X, y, spins,
    ...     model_type='linear',
    ...     n_jobs=1  # Sequential is fine for linear
    ... )
    
    SVR with parallel processing (RECOMMENDED):
    
    >>> p_val = get_reg_r_pval(
    ...     X, y, spins,
    ...     model_type='svr',
    ...     n_jobs=-1,        # Parallel across permutations
    ...     verbose=True,     # Show progress
    ...     scale_data=True   # Critical for SVR
    ... )
    
    Reduce permutations for testing:
    
    >>> # Quick test with fewer permutations
    >>> p_val = get_reg_r_pval(
    ...     X, y, spins,
    ...     n_spins=100,      # Instead of 1000
    ...     model_type='svr',
    ...     n_jobs=-1
    ... )
    
    See Also
    --------
    get_reg_r_sq : get R² without p-value
    spin_data : Generate spatial null permutations
    """
    X, y = _validate_regression_data(X, y)
    spins = np.asarray(spins)
    if spins.ndim != 2 or spins.shape[0] != X.shape[0]:
        raise ValueError(
            "spins must have shape (n_samples, n_permutations); "
            f"got {spins.shape} for {X.shape[0]} samples"
        )
    if not np.issubdtype(spins.dtype, np.integer):
        raise ValueError("spins must contain integer sample indices")
    if spins.size and (spins.min() < 0 or spins.max() >= X.shape[0]):
        raise ValueError("spins contains sample indices outside the valid range")

    if n_spins is None:
        n_spins = spins.shape[1]
    if not isinstance(n_spins, (int, np.integer)) or isinstance(n_spins, bool):
        raise TypeError("n_spins must be an integer or None")
    if n_spins <= 0 or n_spins > spins.shape[1]:
        raise ValueError(
            f"n_spins must be between 1 and {spins.shape[1]}; got {n_spins}"
        )
    
    # Auto-determine scaling
    if scale_data is None:
        scale_data = (model_type == 'svr')
    
    # Pre-scale data once if needed
    if scale_data:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X = scaler_X.fit_transform(X)
        y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    # get observed R²
    if verbose:
        print("Computing observed R²...")
    
    r_squared_obs = get_reg_r_sq(
        X, y,
        adjust=adjust,
        model_type=model_type,
        scale_data=False,  # Already scaled
        **model_kwargs
    )
    
    if verbose:
        print(f"Observed R² = {r_squared_obs:.4f}")
        print(f"Computing null distribution with {n_spins} permutations...")
        if model_type == 'svr' and n_jobs == 1:
            warnings.warn(
                "SVR with sequential processing is very slow. "
                "Consider using n_jobs=-1 for parallel processing.",
                UserWarning
            )
    
    # get null distribution in parallel
    if n_jobs == 1:
        # Sequential with progress bar
        r_squared_null = np.zeros(n_spins)
        iterator = tqdm(range(n_spins), desc="Permutations") if verbose else range(n_spins)
        
        for i in iterator:
            r_squared_null[i] = _get_reg_r_sq_single_permutation(
                X, y, spins[:, i],
                adjust, model_type, False, model_kwargs
            )
    else:
        # Parallel processing
        if verbose:
            print(f"Using {n_jobs if n_jobs > 0 else 'all'} parallel jobs")
        
        results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(_get_reg_r_sq_single_permutation)(
                X, y, spins[:, i],
                adjust, model_type, False, model_kwargs
            )
            for i in range(n_spins)
        )
        r_squared_null = np.array(results)
    
    # Calculate p-value (two-tailed)
    p_value = (1 + np.sum(r_squared_null >= r_squared_obs)) / (n_spins + 1)
    
    if verbose:
        print(f"P-value = {p_value:.4f}")
        print(f"Null R² range: [{r_squared_null.min():.4f}, {r_squared_null.max():.4f}]")
    if return_null:
        return p_value, r_squared_null
    return p_value


def fit_regression_model(
    X: np.ndarray,
    y: np.ndarray,
    spins: Optional[np.ndarray] = None,
    model_type: str = 'linear',
    scale_data: bool = None,
    n_jobs: int = 1,
    verbose: bool = True,
    return_model: bool = False,
    **model_kwargs
) -> Dict:
    """
    Fit regression model with comprehensive output.
    
    High-level interface that fits a model and optionally gets
    p-value using spatial nulls.
    
    Parameters
    ----------
    X : np.ndarray
        Input features
    y : np.ndarray
        Target variable
    spins : np.ndarray, optional
        Spin permutation indices for p-value computation
    model_type : str, default='linear'
        Model type ('linear', 'random_forest', 'svr')
    scale_data : bool, optional
        Whether to scale data. Auto-enabled for SVR.
    n_jobs : int, default=1
        Number of parallel jobs for p-value computation
    verbose : bool, default=True
        Show progress messages
    return_model : bool, default=False
        Whether to return fitted model object
    **model_kwargs
        Additional model parameters
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'r_squared': R² value
        - 'r_squared_adjusted': Adjusted R²
        - 'p_value': Spin test p-value (if spins provided)
        - 'model': Fitted model (if return_model=True)
        - 'predictions': Model predictions
        - 'residuals': Prediction residuals
        - 'null_distribution': Array of null R² values (if spins provided)
        
    Examples
    --------
    Quick linear regression:
    
    >>> from HomoloMap.stats import fit_regression_model
    >>> 
    >>> results = fit_regression_model(
    ...     X, y,
    ...     model_type='linear'
    ... )
    
    SVR with significance testing:
    
    >>> results = fit_regression_model(
    ...     X, y,
    ...     spins=spin_indices,
    ...     model_type='svr',
    ...     scale_data=True,
    ...     n_jobs=-1,           # Parallel processing
    ...     verbose=True,
    ...     kernel='rbf',
    ...     cache_size=2000
    ... )
    >>> 
    >>> print(f"R² = {results['r_squared']:.3f}")
    >>> print(f"p = {results['p_value']:.4f}")
    """
    X, y = _validate_regression_data(X, y)

    # Auto-determine scaling
    if scale_data is None:
        scale_data = (model_type == 'svr')
    
    # Pre-scale data if needed
    X_scaled, y_scaled = X, y
    if scale_data:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    # Fit model
    if verbose:
        print(f"Fitting {model_type} regression model...")
    
    model = _get_model(model_type, **model_kwargs)
    model.fit(X_scaled, y_scaled)
    y_pred = model.predict(X_scaled)
    
    # Reuse this fit for metrics; refitting is expensive and can produce
    # inconsistent values for stochastic estimators such as random forests.
    r_squared = _r_squared_from_predictions(y_scaled, y_pred)
    r_squared_adjusted = _adjust_r_squared(
        r_squared, len(y_scaled), X_scaled.shape[1]
    )

    # get metrics
    results = {
        'r_squared': r_squared,
        'r_squared_adjusted': r_squared_adjusted,
        'predictions': y_pred,
        'residuals': y_scaled - y_pred,
    }
    
    # Add p-value if spins provided
    if spins is not None:
        if verbose:
            print("Computing p-value with spin test...")
        
        # Need to get null distribution too
        n_spins = spins.shape[1]
        
        # get observed
        r_squared_obs = results['r_squared_adjusted']
        
        # get null in parallel
        if n_jobs == 1:
            r_squared_null = np.zeros(n_spins)
            iterator = tqdm(range(n_spins), desc="Permutations") if verbose else range(n_spins)
            
            for i in iterator:
                r_squared_null[i] = _get_reg_r_sq_single_permutation(
                    X_scaled, y_scaled, spins[:, i],
                    True, model_type, False, model_kwargs
                )
        else:
            results_parallel = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
                delayed(_get_reg_r_sq_single_permutation)(
                    X_scaled, y_scaled, spins[:, i],
                    True, model_type, False, model_kwargs
                )
                for i in range(n_spins)
            )
            r_squared_null = np.array(results_parallel)
        
        p_value = (1 + np.sum(r_squared_null >= r_squared_obs)) / (n_spins + 1)
        
        results['p_value'] = p_value
        results['null_distribution'] = r_squared_null
        
        if verbose:
            print(f"P-value = {p_value:.4f}")
    
    # Add model if requested
    if return_model:
        results['model'] = model
    
    if verbose:
        print("Done!")
    
    return results


# =============================================================================
# Feature Importance and Explainability
# =============================================================================

def get_dominance_stats(
    X: np.ndarray,
    y: np.ndarray,
    use_adjusted_r_sq: bool = True,
    method: str = 'auto',
    max_features: int = 15,
    n_samples: Optional[int] = 10000,
    verbose: bool = False,
    n_jobs: int = -1
) -> Tuple[Dict, Dict]:
    """
    Perform dominance analysis with automatic method selection.
    
    Automatically chooses the best method based on number of features:
    - p <= max_features: Full exhaustive analysis
    - p > max_features: Approximate via sampling
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input features
    y : np.ndarray, shape (n_samples,)
        Target variable
    use_adjusted_r_sq : bool, default=True
        Whether to use adjusted R²
    method : {'auto', 'full', 'approximate', 'incremental'}, default='auto'
        Dominance computation method:
        - 'auto': Automatically choose based on n_features
        - 'full': Exhaustive (exact but slow for p>15)
        - 'approximate': Sample-based approximation (fast)
        - 'incremental': Forward selection approximation (fastest)
    max_features : int, default=15
        Maximum features for full analysis (when method='auto')
    n_samples : int, optional
        Number of subset samples for approximate method
    verbose : bool, default=False
        Print progress messages
    n_jobs : int, default=-1
        Number of parallel jobs
        
    Returns
    -------
    model_metrics : dict
        Dominance metrics:
        - 'individual_dominance': shape (1, n_features)
        - 'total_dominance': shape (n_features,)
        - 'full_r_sq': float
        - 'method_used': str (which method was actually used)
    model_r_sq : dict
        R² for getd subsets (may be incomplete for approximate methods)
        
    Raises
    ------
    MemoryError
        If full analysis is attempted with too many features
    ValueError
        If invalid method specified
        
    Warnings
    --------
    Warns if attempting full analysis with > max_features features.
    
    Notes
    -----
    **Computational Complexity:**
    
    Full dominance analysis requires 2^p - 1 model fits:
    - p=10: 1,023 models (~1 second)
    - p=15: 32,767 models (~30 seconds)
    - p=20: 1,048,575 models (~15 minutes)
    - p=22: 4,194,303 models (~1 hour, may crash)
    - p=25: 33,554,431 models (impractical)
    
    **Method Selection:**
    
    1. **Full** (p <= 15): Exact dominance analysis
       - gets all 2^p - 1 subset models
       - Guarantees accuracy
       - Use for final analysis with small feature sets
    
    2. **Approximate** (15 < p <= 30): Monte Carlo sampling
       - Samples subset models randomly
       - Good approximation with enough samples (n_samples ~ 10,000)
       - 100-1000x faster than full
    
    3. **Incremental** (p > 30): Forward selection
       - Sequential feature addition
       - Very fast approximation
       - May miss interaction effects
    
    Examples
    --------
    Small feature set (automatic full analysis):
    
    >>> from HomoloMap.stats import get_dominance_stats
    >>> 
    >>> # 10 cell types - uses full method automatically
    >>> X_small = celltype_data[celltype_data.columns[:10]].values
    >>> metrics, models = get_dominance_stats(
    ...     X_small, y,
    ...     verbose=True
    ... )
    [Dominance] Using full analysis (10 features, 1023 models)
    
    Large feature set (automatic approximation):
    
    >>> # 24 cell types - uses approximate method automatically
    >>> X_large = celltype_data.values  # 24 features
    >>> metrics, models = get_dominance_stats(
    ...     X_large, y,
    ...     verbose=True
    ... )
    [Dominance] Too many features for full analysis (24 > 15)
    [Dominance] Using approximate method with 10000 samples
    
    Force approximate method:
    
    >>> metrics, models = get_dominance_stats(
    ...     X, y,
    ...     method='approximate',
    ...     n_samples=5000,
    ...     verbose=True
    ... )
    
    Fast incremental approximation:
    
    >>> metrics, models = get_dominance_stats(
    ...     X, y,
    ...     method='incremental',
    ...     verbose=True
    ... )
    
    See Also
    --------
    get_dominance_full : Exhaustive dominance analysis
    get_dominance_approximate : Sample-based approximation
    get_dominance_incremental : Forward selection approximation
    """
    n_predictors = X.shape[1]
    
    # Auto-select method
    if method == 'auto':
        if n_predictors <= max_features:
            method = 'full'
            if verbose:
                n_models = 2**n_predictors - 1
                print(f"[Dominance] Using full analysis ({n_predictors} features, {n_models:,} models)")
        elif n_predictors <= 30:
            method = 'approximate'
            if verbose:
                print(f"[Dominance] Too many features for full analysis ({n_predictors} > {max_features})")
                print(f"[Dominance] Using approximate method with {n_samples:,} samples")
        else:
            method = 'incremental'
            if verbose:
                print(f"[Dominance] Large feature set ({n_predictors} features)")
                print(f"[Dominance] Using fast incremental approximation")
    
    # Dispatch to appropriate method
    if method == 'full':
        # Check if feasible
        n_models = 2**n_predictors - 1
        if n_predictors > 20:
            warnings.warn(
                f"Full dominance analysis with {n_predictors} features requires "
                f"{n_models:,} model fits and may exhaust memory or take hours.\n"
                f"Consider using method='approximate' or method='incremental' instead.",
                ResourceWarning
            )
        
        return get_dominance_full(
            X, y, 
            use_adjusted_r_sq=use_adjusted_r_sq,
            verbose=verbose,
            n_jobs=n_jobs
        )
    
    elif method == 'approximate':
        return get_dominance_approximate(
            X, y,
            use_adjusted_r_sq=use_adjusted_r_sq,
            n_samples=n_samples,
            verbose=verbose,
            n_jobs=n_jobs
        )
    
    elif method == 'incremental':
        return get_dominance_incremental(
            X, y,
            use_adjusted_r_sq=use_adjusted_r_sq,
            verbose=verbose
        )
    
    else:
        raise ValueError(
            f"Unknown method: '{method}'. "
            f"Must be 'auto', 'full', 'approximate', or 'incremental'"
        )


def get_dominance_full(
    X: np.ndarray,
    y: np.ndarray,
    use_adjusted_r_sq: bool = True,
    verbose: bool = False,
    n_jobs: int = -1,
    batch_size: int = 1000
) -> Tuple[Dict, Dict]:
    """
    Exhaustive dominance analysis.
    
    gets all 2^p - 1 subset models for exact dominance metrics.
    Only recommended for p <= 15 features.
    
    Parameters
    ----------
    X : np.ndarray
        Input features
    y : np.ndarray
        Target variable
    use_adjusted_r_sq : bool, default=True
        Use adjusted R²
    verbose : bool, default=False
        Print progress
    n_jobs : int, default=-1
        Parallel jobs
    batch_size : int, default=1000
        Process models in batches to save memory
        
    Returns
    -------
    model_metrics : dict
        Dominance metrics
    model_r_sq : dict
        R² for all subsets
    """
    def remove_element(tpl, elem):
        """Remove element from tuple."""
        lst = list(tpl)
        lst.remove(elem)
        return tuple(lst)
    
    def get_r_sq_for_subset(idx_tuple):
        """get R² for a feature subset."""
        from HomoloMap.stats.analysis import get_reg_r_sq
        return idx_tuple, get_reg_r_sq(
            X[:, idx_tuple],
            y,
            adjust=use_adjusted_r_sq,
            model_type='linear'
        )
    
    n_predictors = X.shape[1]
    
    # Generate all combinations
    predictor_combs = [
        list(combinations(range(n_predictors), i))
        for i in range(1, n_predictors + 1)
    ]
    
    if verbose:
        n_combinations = sum(len(group) for group in predictor_combs)
        print(f"[Full dominance] Computing {n_combinations:,} subset models")
    
    # get R² for all subsets in batches
    model_r_sq = {}
    
    for len_group in tqdm(predictor_combs, desc='Subset size', disable=not verbose):
        # Process in batches to manage memory
        for i in range(0, len(len_group), batch_size):
            batch = len_group[i:i+batch_size]
            
            results = Parallel(n_jobs=n_jobs)(
                delayed(get_r_sq_for_subset)(idx_tuple)
                for idx_tuple in batch
            )
            
            for idx_tuple, r_sq in results:
                model_r_sq[idx_tuple] = r_sq
    
    if verbose:
        print(f"[Full dominance] getd {len(model_r_sq):,} R² values")
    
    # get dominance metrics
    model_metrics = _get_dominance_metrics(
        model_r_sq, n_predictors, verbose
    )
    model_metrics['method_used'] = 'full'
    
    return model_metrics, model_r_sq


def get_dominance_approximate(
    X: np.ndarray,
    y: np.ndarray,
    use_adjusted_r_sq: bool = True,
    n_samples: int = 10000,
    verbose: bool = False,
    n_jobs: int = -1,
    seed: int = 42
) -> Tuple[Dict, Dict]:
    """
    Approximate dominance via Monte Carlo sampling.
    
    Randomly samples subset models instead of computing all possible subsets.
    Provides good approximation with much less computation.
    
    Parameters
    ----------
    X : np.ndarray
        Input features
    y : np.ndarray
        Target variable
    use_adjusted_r_sq : bool, default=True
        Use adjusted R²
    n_samples : int, default=10000
        Number of random subsets to sample
    verbose : bool, default=False
        Print progress
    n_jobs : int, default=-1
        Parallel jobs
    seed : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    model_metrics : dict
        Approximate dominance metrics
    model_r_sq : dict
        R² for sampled subsets
        
    Notes
    -----
    Sampling strategy:
    1. Always include individual features (n_predictors samples)
    2. Always include full model (1 sample)
    3. Sample remaining (n_samples - n_predictors - 1) uniformly
    
    With n_samples=10,000, approximation error is typically < 5%.
    """
    rng = np.random.default_rng(seed)  # local RNG, don't pollute global state
    n_predictors = X.shape[1]

    total_subsets = 2 ** n_predictors - 1
    # Can never draw more distinct subsets than exist -> cap the target,
    # otherwise the sampling loop below never terminates.
    target = min(n_samples, total_subsets)

    if verbose:
        print(f"[Approximate dominance] Sampling {target:,} of 2^{n_predictors}-1 = {total_subsets:,} possible subsets")

    # Generate samples
    sampled_subsets = set()

    # 1. Always include individual features
    for i in range(n_predictors):
        sampled_subsets.add((i,))

    # 2. Always include full model
    sampled_subsets.add(tuple(range(n_predictors)))

    # 3. Sample uniformly from other subsets
    while len(sampled_subsets) < target:
        # Random subset size (1 to n_predictors)
        size = int(rng.integers(1, n_predictors + 1))

        # Random features
        features = tuple(sorted(rng.choice(
            n_predictors, size=size, replace=False
        )))

        sampled_subsets.add(features)

    sampled_subsets = list(sampled_subsets)
    
    if verbose:
        print(f"[Approximate dominance] Computing R² for {len(sampled_subsets):,} subsets")
    
    # get R² in parallel
    def get_r_sq_for_subset(idx_tuple):
        from HomoloMap.stats.analysis import get_reg_r_sq
        return idx_tuple, get_reg_r_sq(
            X[:, idx_tuple],
            y,
            adjust=use_adjusted_r_sq,
            model_type='linear'
        )
    
    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(get_r_sq_for_subset)(idx_tuple)
        for idx_tuple in sampled_subsets
    )
    
    model_r_sq = dict(results)
    
    # Estimate dominance from samples
    model_metrics = _estimate_dominance_from_samples(
        model_r_sq, n_predictors, verbose
    )
    model_metrics['method_used'] = 'approximate'
    model_metrics['n_samples'] = len(sampled_subsets)
    
    return model_metrics, model_r_sq


def get_dominance_incremental(
    X: np.ndarray,
    y: np.ndarray,
    use_adjusted_r_sq: bool = True,
    verbose: bool = False
) -> Tuple[Dict, Dict]:
    """
    Fast dominance approximation via forward selection.
    
    Uses forward selection to estimate feature importance.
    Very fast but may miss interaction effects.
    
    Parameters
    ----------
    X : np.ndarray
        Input features
    y : np.ndarray
        Target variable
    use_adjusted_r_sq : bool, default=True
        Use adjusted R²
    verbose : bool, default=False
        Print progress
        
    Returns
    -------
    model_metrics : dict
        Approximate dominance metrics
    model_r_sq : dict
        R² for forward selection path
        
    Notes
    -----
    This method:
    1. Starts with null model (R² = 0)
    2. At each step, adds feature with largest R² increase
    3. Continues until all features added
    
    Total models: n_predictors * (n_predictors + 1) / 2
    
    Much faster than full analysis but less accurate.
    """
    from HomoloMap.stats.analysis import get_reg_r_sq
    
    n_predictors = X.shape[1]
    
    if verbose:
        print(f"[Incremental dominance] Forward selection with {n_predictors} features")
    
    # Track selected features and R² path
    selected = []
    available = list(range(n_predictors))
    model_r_sq = {}
    
    # Individual R² for all features
    individual_r_sq = np.zeros(n_predictors)
    for i in range(n_predictors):
        r_sq = get_reg_r_sq(
            X[:, [i]], y,
            adjust=use_adjusted_r_sq,
            model_type='linear'
        )
        individual_r_sq[i] = r_sq
        model_r_sq[(i,)] = r_sq
    
    # Forward selection
    while available:
        if verbose:
            print(f"  Step {len(selected)+1}/{n_predictors}", end='\r')
        
        best_r_sq = -np.inf
        best_feature = None
        
        # Try adding each remaining feature
        for feature in available:
            current_features = selected + [feature]
            r_sq = get_reg_r_sq(
                X[:, current_features], y,
                adjust=use_adjusted_r_sq,
                model_type='linear'
            )
            
            if r_sq > best_r_sq:
                best_r_sq = r_sq
                best_feature = feature
        
        # Add best feature
        selected.append(best_feature)
        available.remove(best_feature)
        model_r_sq[tuple(sorted(selected))] = best_r_sq
    
    if verbose:
        print()  # New line
    
    # get incremental importance
    r_sq_path = [model_r_sq[tuple(sorted(selected[:i+1]))] 
                  for i in range(n_predictors)]
    
    # Incremental R² = change when feature added
    incremental_r_sq = [r_sq_path[0]]
    for i in range(1, n_predictors):
        incremental_r_sq.append(r_sq_path[i] - r_sq_path[i-1])
    
    # Map back to original feature order
    dominance = np.zeros(n_predictors)
    for i, feature_idx in enumerate(selected):
        dominance[feature_idx] = incremental_r_sq[i]
    
    # Create metrics
    model_metrics = {
        'individual_dominance': individual_r_sq.reshape(1, -1),
        'total_dominance': dominance,
        'full_r_sq': r_sq_path[-1],
        'method_used': 'incremental',
        'selection_order': selected,
        'r_sq_path': r_sq_path
    }
    
    return model_metrics, model_r_sq


def _get_dominance_metrics(model_r_sq, n_predictors, verbose=False):
    """get exact dominance metrics from complete R² dictionary."""
    
    def remove_element(tpl, elem):
        lst = list(tpl)
        lst.remove(elem)
        return tuple(lst)
    
    # Individual dominance
    individual_dominance = np.array([
        model_r_sq[(i,)] for i in range(n_predictors)
    ]).reshape(1, -1)
    
    # Partial dominance
    partial_dominance = [[] for _ in range(n_predictors - 1)]
    
    for subset_size in range(n_predictors - 1):
        size_combs = list(combinations(range(n_predictors), subset_size + 2))
        
        for predictor_idx in range(n_predictors):
            with_predictor = [c for c in size_combs if predictor_idx in c]
            without_predictor = [
                remove_element(comb, predictor_idx) 
                for comb in with_predictor
            ]
            
            incremental_r_sq = [
                model_r_sq[with_predictor[i]] - model_r_sq[without_predictor[i]]
                for i in range(len(without_predictor))
            ]
            
            partial_dominance[subset_size].append(np.mean(incremental_r_sq))
    
    partial_dominance = np.array(partial_dominance)
    
    # Total dominance
    total_dominance = np.mean(
        np.vstack([individual_dominance, partial_dominance]),
        axis=0
    )
    
    full_r_sq = model_r_sq[tuple(range(n_predictors))]
    
    if not np.allclose(total_dominance.sum(), full_r_sq):
        warnings.warn(
            f"Total dominance sum ({total_dominance.sum():.6f}) != "
            f"full R² ({full_r_sq:.6f})",
            RuntimeWarning
        )
    
    return {
        'individual_dominance': individual_dominance,
        'partial_dominance': partial_dominance,
        'total_dominance': total_dominance,
        'full_r_sq': full_r_sq
    }


def _estimate_dominance_from_samples(model_r_sq, n_predictors, verbose=False):
    """Estimate dominance metrics from sampled subsets."""
    
    # Individual dominance (exact)
    individual_dominance = np.array([
        model_r_sq.get((i,), 0.0) for i in range(n_predictors)
    ]).reshape(1, -1)
    
    # Estimate incremental contributions
    contributions = {i: [] for i in range(n_predictors)}
    
    for subset, r_sq in model_r_sq.items():
        if len(subset) == 1:
            continue
            
        # Try removing each feature
        for feature in subset:
            reduced_subset = tuple(f for f in subset if f != feature)
            if reduced_subset in model_r_sq:
                contribution = r_sq - model_r_sq[reduced_subset]
                contributions[feature].append(contribution)
    
    # Average contributions
    total_dominance = np.zeros(n_predictors)
    for i in range(n_predictors):
        if contributions[i]:
            total_dominance[i] = np.mean(contributions[i])
        else:
            # Fall back to individual dominance
            total_dominance[i] = individual_dominance[0, i]
    
    # Normalize to sum to full R²
    full_r_sq = model_r_sq.get(tuple(range(n_predictors)), 
                                total_dominance.sum())
    
    if total_dominance.sum() > 0:
        total_dominance = total_dominance * (full_r_sq / total_dominance.sum())
    
    return {
        'individual_dominance': individual_dominance,
        'total_dominance': total_dominance,
        'full_r_sq': full_r_sq
    }


# [Rest of the file with get_shap_stats, get_feature_importance, etc. stays the same]
# I'll include just the key functions here for brevity

def get_shap_stats(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = 'linear',
    feature_names: Optional[List[str]] = None,
    n_samples: int = 100,
    scale_data: Optional[bool] = None,
    random_state: int = 42,
    model_kwargs: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Compute SHAP values for a given model and dataset.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    model_type : str, default='linear'
        Type of model to use ('linear', 'random_forest', 'svr').
    feature_names : list of str, optional
        Names of the features.
    n_samples : int, default=100
        Number of background samples for KernelExplainer (only used for SVR).
        Smaller = faster but less accurate. Ignored for linear/tree models.


    Returns
    -------
    summary : pd.DataFrame
        Summary of SHAP values.
    shap_values : np.ndarray
        SHAP values for each feature.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Install with: pip install shap")

    from sklearn.linear_model import Ridge

    X, y = _validate_regression_data(X, y)
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]
    elif len(feature_names) != X.shape[1]:
        raise ValueError(
            f"feature_names must contain {X.shape[1]} names; got {len(feature_names)}"
        )
    if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")

    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    if scale_data is None:
        scale_data = model_type in {'svr', 'ridge'}
    X_model, y_model = X, y
    if scale_data:
        X_model = StandardScaler().fit_transform(X)
        y_model = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()

    # Train the model
    if model_type == 'linear':
        model_kwargs.pop('n_jobs', None)
        model = LinearRegression(**model_kwargs).fit(X_model, y_model)
    elif model_type == 'ridge':
        model = Ridge(**model_kwargs).fit(X_model, y_model)
    elif model_type == 'random_forest':
        model_kwargs.setdefault('random_state', random_state)
        model = RandomForestRegressor(**model_kwargs).fit(X_model, y_model)
    elif model_type == 'svr':
        model = SVR(**model_kwargs).fit(X_model, y_model)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Pick the matching explainer and compute SHAP values exactly once.
    if model_type in ['linear', 'ridge']:
        explainer = shap.LinearExplainer(model, X_model)
        shap_values = explainer.shap_values(X_model)

    elif model_type == 'random_forest':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_model)

    else:  # svr -- KernelExplainer, optionally on a background subsample
        print(f"  Warning: KernelExplainer is slow. Using {n_samples} background samples.")
        print(f"  Consider using model_type='linear' or 'ridge' for faster results.")

        if X_model.shape[0] > n_samples:
            rng = np.random.default_rng(random_state)
            background_idx = rng.choice(X_model.shape[0], n_samples, replace=False)
            background = X_model[background_idx]
        else:
            background = X_model

        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_model)

    if hasattr(shap_values, 'values'):
        shap_values = shap_values.values
    if isinstance(shap_values, list):
        if len(shap_values) != 1:
            raise ValueError("Expected one SHAP output for a regression model")
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values, dtype=float)
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values[..., 0]
    if shap_values.shape != X_model.shape:
        raise ValueError(
            f"Unexpected SHAP value shape: expected {X_model.shape}, got {shap_values.shape}"
        )

    # Create summary DataFrame
    mean_abs = np.abs(shap_values).mean(axis=0)
    total_mean_abs = float(mean_abs.sum())
    relative = np.divide(
        mean_abs, total_mean_abs, out=np.zeros_like(mean_abs), where=total_mean_abs > 0
    )
    summary = pd.DataFrame({
        "mean(|SHAP value|)": mean_abs,
        "total(|SHAP value|)": np.abs(shap_values).sum(axis=0),
        "relative_contribution": relative,
        "mean_signed_contribution": shap_values.mean(axis=0),
    }, index=feature_names).sort_values(
        by="mean(|SHAP value|)", ascending=False
    )
    summary.attrs['total_mean_absolute_shap'] = total_mean_abs
    summary.attrs['total_contribution'] = total_mean_abs  # compatibility alias
    summary.attrs['model_type'] = model_type
    summary.attrs['scale_data'] = bool(scale_data)

    return summary, shap_values
