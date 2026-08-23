"""Compositional transformations for cell-type proportion maps."""

import numpy as np
import pandas as pd
from scipy.linalg import helmert


def close_composition(data):
    """Close non-negative rows to unit sum."""
    frame = _validate_composition(data)
    totals = frame.sum(axis=1)
    if (totals <= 0).any():
        raise ValueError("Every composition must have a positive row sum")
    return frame.div(totals, axis=0)


def multiplicative_zero_replacement(data, fraction=0.65):
    """Replace row-wise zeros while preserving ratios among non-zero parts."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")
    frame = close_composition(data)
    values = frame.to_numpy(copy=True)
    for row in range(values.shape[0]):
        zero = values[row] == 0
        n_zero = int(zero.sum())
        if not n_zero:
            continue
        positive = values[row, ~zero]
        if positive.size == 0:
            raise ValueError("A composition cannot contain only zeros")
        delta = fraction * positive.min()
        replacement_mass = n_zero * delta
        if replacement_mass >= 1:
            raise ValueError("Zero replacement mass is too large for this composition")
        values[row, zero] = delta
        values[row, ~zero] *= (1 - replacement_mass) / positive.sum()
    return pd.DataFrame(values, index=frame.index, columns=frame.columns)


def clr_transform(data, zero_method='multiplicative', zero_fraction=0.65):
    """Centered log-ratio transform, retaining one column per cell type."""
    frame = _prepare_positive(data, zero_method, zero_fraction)
    logged = np.log(frame.to_numpy())
    values = logged - logged.mean(axis=1, keepdims=True)
    return pd.DataFrame(values, index=frame.index, columns=frame.columns)


def ilr_transform(data, zero_method='multiplicative', zero_fraction=0.65,
                  basis=None):
    """Isometric log-ratio transform using an orthonormal Helmert basis."""
    frame = _prepare_positive(data, zero_method, zero_fraction)
    n_parts = frame.shape[1]
    if n_parts < 2:
        raise ValueError("ILR requires at least two compositional parts")
    if basis is None:
        basis = helmert(n_parts, full=False)
    basis = np.asarray(basis, dtype=float)
    if basis.shape != (n_parts - 1, n_parts):
        raise ValueError(
            f"basis must have shape {(n_parts - 1, n_parts)}; got {basis.shape}"
        )
    values = np.log(frame.to_numpy()) @ basis.T
    columns = [f'ilr_balance_{i + 1}' for i in range(n_parts - 1)]
    result = pd.DataFrame(values, index=frame.index, columns=columns)
    result.attrs['ilr_basis'] = pd.DataFrame(basis, columns=frame.columns)
    return result


def transform_composition(data, method='none', zero_method='multiplicative',
                          zero_fraction=0.65, min_mapping_coverage=None,
                          mapping_coverage=None):
    """Apply none/CLR/ILR transformation and return data plus provenance."""
    method = method.lower()
    if method not in {'none', 'clr', 'ilr'}:
        raise ValueError("method must be 'none', 'clr', or 'ilr'")
    frame = _validate_composition(data)
    if min_mapping_coverage is not None:
        if not 0 <= min_mapping_coverage <= 1:
            raise ValueError("min_mapping_coverage must be between 0 and 1")
        if mapping_coverage is None:
            raise ValueError("mapping_coverage is required when a threshold is set")
        coverage = pd.Series(mapping_coverage).reindex(frame.index)
        failed = coverage.isna() | (coverage < min_mapping_coverage)
        if failed.any():
            raise ValueError(
                f"{int(failed.sum())} rows fall below min_mapping_coverage="
                f"{min_mapping_coverage}; minimum observed coverage is "
                f"{coverage.min():.4f}"
            )
    if method == 'none':
        transformed = frame.copy()
    elif method == 'clr':
        transformed = clr_transform(frame, zero_method, zero_fraction)
    else:
        transformed = ilr_transform(frame, zero_method, zero_fraction)
    metadata = {
        'method': method,
        'zero_method': zero_method if method != 'none' else None,
        'zero_fraction': zero_fraction if method != 'none' else None,
        'closed': method != 'none',
        'min_mapping_coverage': min_mapping_coverage,
        'original_features': frame.columns.tolist(),
        'transformed_features': transformed.columns.tolist(),
    }
    if method == 'ilr':
        metadata['ilr_basis'] = transformed.attrs['ilr_basis']
    transformed.attrs['composition_transform'] = metadata
    return transformed, metadata


def _prepare_positive(data, zero_method, zero_fraction):
    frame = close_composition(data)
    if (frame == 0).any(axis=None):
        if zero_method == 'raise':
            raise ValueError("Log-ratio transformation requires strictly positive values")
        if zero_method != 'multiplicative':
            raise ValueError("zero_method must be 'raise' or 'multiplicative'")
        frame = multiplicative_zero_replacement(frame, zero_fraction)
    return frame


def _validate_composition(data):
    frame = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if frame.empty or frame.shape[1] == 0:
        raise ValueError("Composition must be a non-empty two-dimensional table")
    frame = frame.astype(float)
    values = frame.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Composition must contain only finite values")
    if (values < 0).any():
        raise ValueError("Composition cannot contain negative values")
    return frame
