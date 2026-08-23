# HomoloMap/parcellation.py
"""
Parcellation transformation and data aggregation utilities.

This module provides functions for:
- Converting between surface vertex and parcellation data
- Transforming data between different atlases
- Aggregating vertex-level data to parcels
- Loading and processing neuroimaging data across multiple formats

Key functions:
- parc2vertex: Convert parcel data to vertex representation
- vertices_to_parcels: Aggregate vertex data to parcels
- surf_relabel: Transform surface data between atlases
- vol_relabel: Transform volumetric data between atlases
- load_data: Unified data loading interface (NIfTI, GIFTI, CSV)
- load_data_list: Batch loading for multiple files

Volumetric atlas handling uses small functional helpers from ``atlas``.
"""

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union, Tuple, Literal, List
from scipy.spatial import cKDTree, distance_matrix

try:  # scipy >= 1.8.0
    from scipy.ndimage._measurements import _stats, labeled_comprehension
except ImportError:  # scipy < 1.8.0
    from scipy.ndimage.measurements import _stats, labeled_comprehension

import nibabel as nib
from nibabel.filebasedimages import ImageFileError

try:
    from nilearn import image as nli
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False
    warnings.warn("nilearn not available. Volumetric resampling disabled.")

try:
    from neuromaps import transforms as nm_transforms
    NEUROMAPS_AVAILABLE = True
except ImportError:
    NEUROMAPS_AVAILABLE = False
    warnings.warn("neuromaps not available. Cross-space transforms disabled.")

from HomoloMap.datasets import fetch_parc, fetch_annot, fetch_fslr
from .atlas import assign_closest_centroids, load_volume_atlas

# Try to import parc_smooth - handle both old and new locations
try:
    from HomoloMap.transforms.smoothing import parc_smooth
except ImportError:
    try:
        from HomoloMap.stats.spins import parc_smooth
    except ImportError:
        warnings.warn("parc_smooth not available. Smoothing disabled.")
        parc_smooth = None


# =============================================================================
# Constants
# =============================================================================

def _to_density(data: pd.DataFrame, roi_labels: np.ndarray) -> pd.DataFrame:
    """
    Divide each ROI's value by its voxel count → per-voxel density.

    Required before expanding ROI data to voxel level when ``method='sum'``:
    without normalisation, each voxel gets the full ROI value C_j, so
    summing m_jk overlapping voxels yields C_j * m_jk instead of the
    correct proportional contribution C_j * m_jk / n_j.
    """
    unique, counts = np.unique(roi_labels, return_counts=True)
    size = pd.Series(counts, index=unique.astype(int)).reindex(data.index)
    if size.isna().any():
        warnings.warn(
            "Some ROI labels in data have no voxels in the source atlas; "
            "their per-voxel densities will be NaN."
        )
    return data.div(size, axis=0)

def ndimage_adjust(
    vertex_data: np.ndarray,
    labels: np.ndarray,
    method: Literal['mean', 'sum', 'median'] = 'mean'
) -> np.ndarray:
    """
    Aggregate vertex-level data to parcel-level using specified method.
    
    This is a low-level function for efficient aggregation using scipy.ndimage.
    For most use cases, use vertices_to_parcels() instead.
    
    Parameters
    ----------
    vertex_data : np.ndarray
        Vertex-level data, shape (n_vertices,) or (n_vertices, n_features)
    labels : np.ndarray
        Parcel label for each vertex, shape (n_vertices,)
    method : {'mean', 'sum', 'median'}, default='mean'
        Aggregation method:
        - 'mean': Average values within each parcel
        - 'sum': Sum values within each parcel
        - 'median': Median value within each parcel
        
    Returns
    -------
    reduced : np.ndarray
        Parcel-level data, shape (n_parcels,) or (n_parcels, n_features)
        
    Notes
    -----
    Handles NaN values appropriately:
    - 'mean': Computes mean of non-NaN values
    - 'sum': Sums non-NaN values
    - 'median': Takes median of non-NaN values
    
    Examples
    --------
    >>> vertex_data = np.random.randn(32492)
    >>> labels = np.random.randint(0, 378, 32492)
    >>> parcel_data = ndimage_adjust(vertex_data, labels, method='mean')
    >>> print(parcel_data.shape)  # (378,)
    
    See Also
    --------
    vertices_to_parcels : High-level interface for aggregation
    """
    # Get number of parcels and features
    n_parc = np.unique(labels).size
    n_features = vertex_data.shape[-1]
    
    # Initialize numerator and denominator
    numerator = np.zeros((n_parc, n_features), dtype=vertex_data.dtype)
    denominator = np.zeros((n_parc, n_features), dtype=vertex_data.dtype)
    indices = np.unique(labels)
    
    # Iterate over each feature dimension
    for idx in range(n_features):
        currdata = np.squeeze(vertex_data[:, idx]).astype(float)
        
        # Compute sums and counts within each parcel
        counts, sums = _stats(np.nan_to_num(currdata), labels, indices)
        _, nacounts = _stats(np.isnan(currdata), labels, indices)
        
        # Update numerator and denominator
        counts = (np.asanyarray(counts, dtype=float) - 
                  np.asanyarray(nacounts, dtype=float))
        numerator[:, idx] += sums
        denominator[:, idx] += counts
    
    # Compute statistics based on method
    if method == 'mean':
        with np.errstate(divide='ignore', invalid='ignore'):
            reduced = np.squeeze(numerator / denominator)
    
    elif method == 'sum':
        reduced = numerator.astype(float, copy=True)
        reduced[denominator == 0] = np.nan
    
    elif method == 'median':
        reduced = np.zeros((n_parc, n_features), dtype=vertex_data.dtype)
        for i, label in enumerate(indices):
            mask = labels == label
            for j in range(n_features):
                reduced[i, j] = np.nanmedian(vertex_data[mask, j])
    
    else:
        raise ValueError(
            f"Unsupported method: {method}. "
            "Choose from 'mean', 'sum', 'median'."
        )
    
    return reduced


def vertices_to_parcels(
    vertex_data: Union[np.ndarray, list],
    parc: str = 'FGC',
    background: Optional[float] = None,
    method: Literal['mean', 'sum', 'median'] = 'mean'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate vertex-level data to parcel-level.
    
    Parameters
    ----------
    vertex_data : np.ndarray or list
        Vertex-level data, shape (n_vertices,) or (n_vertices, n_features)
        If list, will be stacked
    parc : str, default='FGC'
        Parcellation key or atlas name
    background : float, optional
        Value to treat as background/missing (will be set to NaN)
    method : {'mean', 'sum', 'median'}, default='mean'
        Aggregation method
        
    Returns
    -------
    parcel_data : np.ndarray
        Parcel-level aggregated data, shape (n_parcels,) or (n_parcels, n_features)
    labels : np.ndarray
        Parcel labels
        
    Examples
    --------
    Aggregate single feature:
    
    >>> from HomoloMap.parcellation import vertices_to_parcels
    >>> 
    >>> # Vertex data (32,492 vertices for fsLR-32k)
    >>> vertex_data = np.random.randn(32492)
    >>> 
    >>> # Aggregate to parcels
    >>> parcel_data, labels = vertices_to_parcels(
    ...     vertex_data,
    ...     parc='FGC',
    ...     method='mean'
    ... )
    >>> 
    >>> print(f"Parcel data: {parcel_data.shape}")  # (378,)
    >>> print(f"Labels: {labels.shape}")  # (378,)
    
    See Also
    --------
    parc2vertex : Convert parcel data to vertex representation
    """
    # Stack if list
    vertex_data = np.vstack(vertex_data) if isinstance(vertex_data, list) else vertex_data
    vertex_data = vertex_data.astype(float)
    
    # Handle background
    if background is not None:
        vertex_data[vertex_data == background] = np.nan
    
    # Load parcellation
    vertices = np.hstack([fetch_parc(key=parc).agg_data()])
    n_parc = int(np.max(vertices)) + 1
    
    # Validate size
    expected = vertices.shape[0]
    if expected != len(vertex_data):
        raise ValueError(
            f'Number of vertices in provided annotation files '
            f'differs from size of vertex-level data array.\n'
            f'    EXPECTED: {expected} vertices\n'
            f'    RECEIVED: {len(vertex_data)} vertices'
        )
    
    # Aggregate
    labels_surf = fetch_parc(key=parc).agg_data().astype('int')
    reduced = ndimage_adjust(vertex_data, vertices, method=method)
    label = np.unique(vertices)
    
    
    # Handle single vs multiple features
    if len(reduced.shape) > 1:
        return reduced[label>0, :], label[label>0]
    else:
        return reduced[label>0], label[label>0]


def parc2vertex(
    parc_data: Union[pd.DataFrame, pd.Series, np.ndarray],
    atlas: str = 'FGC',
    data_labels: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert parcellation data to vertex-level representation for visualization.
    
    Maps regional values to all vertices within each region, creating
    a vertex-wise array suitable for surface plotting.
    
    Parameters
    ----------
    parc_data :  pd.DataFrame, pd.Series, or np.ndarray
        Parcellation data with regions as index.
        - If `pd.DataFrame`, shape: (n_regions, n_features)
        - If `pd.Series`, shape: (n_regions,)
        - If `np.ndarray`, shape: (n_regions,) or (n_regions, n_features)
    atlas : str, default='FGC'
        Atlas name for parcellation
        
    Returns
    -------
    vertex_data : np.ndarray
        Vertex-level data, shape (n_vertices,) or (n_vertices, n_features)
        
    Examples
    --------
    Convert cell type data for visualization:
    
    >>> from HomoloMap.parcellation import parc2vertex
    >>> from HomoloMap.datasets import fetch_ctype_ratio
    >>> 
    >>> # Load regional data
    >>> celltype_data = fetch_ctype_ratio(level='subclass')
    >>> 
    >>> # Convert to vertex representation
    >>> vertex_data = parc2vertex(celltype_data[['L2/3_IT']], atlas='FGC')
    >>> print(vertex_data.shape)  # (32492, 1)
    
    See Also
    --------
    vertices_to_parcels : Aggregate vertex data to parcels
    """
    # Load parcellation
    try:
        vertex_labels = fetch_parc(key=atlas).agg_data()
    except FileExistsError:
        vertex_labels = nib.load(atlas).agg_data()
    
    # Handle D99 atlas (special case)
    if atlas == 'D99':
        vertex_labels = np.where(vertex_labels == 0, 0, vertex_labels - 298)
    elif atlas == 'BN':
        vertex_labels = np.where(vertex_labels == -1, 0, vertex_labels)
    vertex_labels = vertex_labels.astype(int)
    # Get unique labels
    labels = np.unique(vertex_labels)

    # Handle pd.Series
    if isinstance(parc_data, pd.Series):
        parc_data = parc_data.to_frame(name='feature')
    
    # Handle np.ndarray
    if isinstance(parc_data, np.ndarray):
        if data_labels is None:
            positive_labels = labels[labels > 0]
            if not np.array_equal(
                positive_labels, np.arange(1, len(positive_labels) + 1)
            ):
                raise ValueError(
                    "data_labels is required for NumPy input when atlas labels "
                    "are non-consecutive"
                )
            data_labels = positive_labels
        data_labels = np.asarray(data_labels)
        if data_labels.ndim != 1 or len(data_labels) != len(parc_data):
            raise ValueError("data_labels must contain one label per data row")
        if len(np.unique(data_labels)) != len(data_labels):
            raise ValueError("data_labels must be unique")
        if parc_data.ndim == 1:
            parc_data = parc_data[:, np.newaxis]
            parc_data = pd.DataFrame(parc_data,index=data_labels, columns=['feature'])
        else:
            parc_data = pd.DataFrame(parc_data,index=data_labels, columns=[f'feature_{i}' for i in range(parc_data.shape[1])])

    # Handle MultiIndex columns
    if isinstance(parc_data.columns, pd.MultiIndex):
        parc_data = parc_data.copy()
        parc_data.columns = parc_data.columns.get_level_values(-1)
    expected_labels = range(labels.max() + 1)
    if not set(parc_data.index).issubset(expected_labels):
        raise ValueError("Indices of 'parc_data' do not match the expected labels in the atlas.")

    # Create lookup table
    parc_data = (
        pd.DataFrame(range(labels.max() + 1))
        .merge(parc_data, left_on=0, right_index=True, how='left')
        .iloc[:, 1:]
        .values
    )
    
    # Map to vertices
    vertex_data = parc_data[vertex_labels]
    
    return vertex_data


# =============================================================================
# Atlas Transformation
# =============================================================================

def _validate_relabel_data(
    data: pd.DataFrame,
    source_labels: np.ndarray,
    missing: str
) -> pd.DataFrame:
    """Validate ROI-indexed data and align it to real source labels."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame indexed by ROI label")
    if not data.index.is_unique:
        raise ValueError("data ROI labels must be unique")
    if missing not in {'omit', 'propagate', 'raise'}:
        raise ValueError("missing must be 'omit', 'propagate', or 'raise'")

    source_labels = np.asarray(source_labels, dtype=int)
    try:
        data = data.copy()
        data.index = data.index.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError("data index must contain integer ROI labels") from exc

    unknown = data.index.difference(source_labels)
    if len(unknown):
        raise ValueError(f"data contains labels absent from source atlas: {unknown.tolist()}")

    aligned = data.reindex(source_labels).astype(float)
    if missing == 'raise' and aligned.isna().any().any():
        raise ValueError("source data contains missing ROI values")
    return aligned


def _aggregate_relabel_values(
    sample_data: np.ndarray,
    target_labels: np.ndarray,
    columns: pd.Index,
    method: str,
    missing: str
) -> pd.DataFrame:
    """Aggregate sample-level values using real target labels."""
    target_labels = np.asarray(target_labels, dtype=int)
    labels = np.unique(target_labels)
    labels = labels[labels > 0]
    reduced = ndimage_adjust(sample_data, target_labels, method=method)
    reduced = np.asarray(reduced)
    if reduced.ndim == 1:
        reduced = reduced[:, np.newaxis]
    all_labels = np.unique(target_labels)
    reduced = reduced[all_labels > 0]

    if missing == 'propagate':
        missing_values = np.isnan(sample_data)
        if missing_values.ndim == 1:
            missing_values = missing_values[:, np.newaxis]
        for row, label in enumerate(labels):
            affected = missing_values[target_labels == label].any(axis=0)
            reduced[row, affected] = np.nan

    return pd.DataFrame(reduced, index=labels, columns=columns)


def relabel(
    data: pd.DataFrame,
    src: str,
    trg: str,
    space: Literal['surface', 'volume'],
    method: Literal['mean', 'sum', 'median'] = 'mean',
    missing: Literal['omit', 'propagate', 'raise'] = 'omit',
    cross_species: bool = False
) -> pd.DataFrame:
    """Relabel left-hemisphere ROI data through a shared surface/volume API."""
    if space not in {'surface', 'volume'}:
        raise ValueError("space must be 'surface' or 'volume'")

    if cross_species:
        intermediate = relabel(
            data, src, 'MacBN', space, method, missing, cross_species=False
        )
        return relabel(
            intermediate, 'MacBN_human', trg, space, method, missing,
            cross_species=False
        )

    if space == 'surface':
        source_vertices = np.hstack([fetch_parc(key=src).agg_data()]).astype(int)
        target_vertices = np.hstack([fetch_parc(key=trg).agg_data()]).astype(int)
        if source_vertices.shape != target_vertices.shape:
            raise ValueError(
                "source and target surface atlases must use the same mesh, "
                "hemisphere, and vertex density"
            )
        source_vertices[source_vertices <= 0] = 0
        target_vertices[target_vertices <= 0] = 0
        source_labels = np.unique(source_vertices)
        source_labels = source_labels[source_labels > 0]
        aligned = _validate_relabel_data(data, source_labels, missing)
        if method == 'sum':
            aligned = _to_density(aligned, source_vertices)
        sample_data = aligned.reindex(source_vertices).to_numpy(dtype=float)
        return _aggregate_relabel_values(
            sample_data, target_vertices, data.columns, method, missing
        )

    try:
        source_path, source_info = fetch_annot(atlas=src, annot=True)
        source_atlas = load_volume_atlas(source_path, source_info, hemisphere='left')
    except FileExistsError:
        source_atlas = load_volume_atlas(src, hemisphere='left')
    target_path, target_info = fetch_annot(atlas=trg, annot=True)
    target_atlas = load_volume_atlas(target_path, target_info, hemisphere='left')

    aligned = _validate_relabel_data(data, source_atlas['roi_labels'], missing)
    if method == 'sum':
        aligned = _to_density(aligned, source_atlas['voxel_labels'])
    sample_data = aligned.reindex(source_atlas['voxel_labels']).to_numpy(dtype=float)
    closest = assign_closest_centroids(
        source_atlas['coords'], target_atlas['centroids']
    )
    target_labels = target_atlas['roi_labels'][closest]
    return _aggregate_relabel_values(
        sample_data, target_labels, data.columns, method, missing
    )

def vol_relabel(
    src: str,
    trg: str,
    data: pd.DataFrame,
    cross_species: bool = False,
    method: Literal['mean', 'sum', 'median'] = 'mean',
    missing: Literal['omit', 'propagate', 'raise'] = 'omit'
) -> pd.DataFrame:
    """
    Transform volumetric parcellation data between atlases.
    
    Relabels volumetric data by mapping centroids from source to target atlas.
    
    Parameters
    ----------
    src : str
        Source atlas name or path
    trg : str
        Target atlas name or path
    data : pd.DataFrame
        Data in source atlas space
    cross_species : bool, default=False
        If True, perform cross-species transformation via macaque intermediate
    method : {'mean', 'sum', 'median'}, default='mean'
        Aggregation method for voxels
        
    Returns
    -------
    data_transformed : pd.DataFrame
        Data in target atlas space
        
    Examples
    --------
    >>> from HomoloMap.parcellation import vol_relabel
    >>> 
    >>> # Transform between human atlases
    >>> data_fgc = vol_relabel(
    ...     src='DK',
    ...     trg='FGC',
    ...     data=data_dk
    ... )
    
    See Also
    --------
    surf_relabel : Transform surface data between atlases
    """
    return relabel(
        data=data, src=src, trg=trg, space='volume', method=method,
        missing=missing, cross_species=cross_species
    )


def surf_relabel(
    data: pd.DataFrame,
    src: str = 'DK',
    trg: str = 'FGC',
    cross_species: bool = False,
    method: Literal['mean', 'sum', 'median'] = 'mean',
    missing: Literal['omit', 'propagate', 'raise'] = 'omit'
) -> pd.DataFrame:
    """
    Transform surface parcellation data between atlases.
    
    Relabels data from one surface atlas to another by:
    1. Converting source parcel data to vertices
    2. Aggregating vertices to target parcels
    
    Parameters
    ----------
    data : pd.DataFrame
        Data in source atlas space
    src : str, default='DK'
        Source atlas name
    trg : str, default='FGC'
        Target atlas name
    cross_species : bool, default=False
        If True, transform via macaque→human mapping
    method : {'mean', 'sum', 'median'}, default='mean'
        Aggregation method
        
    Returns
    -------
    data_transformed : pd.DataFrame
        Data in target atlas space
        
    Examples
    --------
    Transform from Desikan-Killiany to FGC:
    
    >>> from HomoloMap.parcellation import surf_relabel
    >>> 
    >>> # Data in DK atlas (34 regions)
    >>> data_dk = pd.DataFrame(...)  # (34, n_features)
    >>> 
    >>> # Transform to FGC (378 regions)
    >>> data_fgc = surf_relabel(
    ...     data_dk,
    ...     src='DK',
    ...     trg='FGC',
    ...     method='mean'
    ... )
    
    See Also
    --------
    vol_relabel : Transform volumetric data between atlases
    """
    return relabel(
        data=data, src=src, trg=trg, space='surface', method=method,
        missing=missing, cross_species=cross_species
    )


# =============================================================================
# Unified Data Loading
# =============================================================================

def load_data(
    data: Union[str, Path, pd.Series, pd.DataFrame, nib.spatialimages.SpatialImage],
    space: Literal['mni152', 'fslr', 'fsaverage', 'civet'] = 'fslr',
    atlas: Optional[str] = None,
    path: Optional[Union[str, Path]] = None,
    trg: str = 'FGC',
    keep_left: bool = True,
    smooth: bool = True,
    cross_species: bool = False,
    transform: bool = False,
    transform_method: str = 'linear',
    smooth_param: Optional[dict] = None
) -> pd.DataFrame:
    """
    Unified interface for loading and processing brain data.
    
    Handles multiple input types and automatically performs necessary
    transformations to get data into the requested atlas.
    
    Parameters
    ----------
    data : str, Path, pd.DataFrame, or nibabel image
        Input data:
        - File path to NIfTI or GIFTI image
        - Pandas DataFrame with ROI labels as rows
        - nibabel image object
    space : {'mni152', 'fslr', 'fsaverage', 'civet'}, default='mni152'
        Space of input data
    atlas : str, optional
        Source atlas (if None, assumes trg)
    path : str or Path, optional
        Path to atlas file (if atlas not preloaded)
    trg : str, default='FGC'
        Target atlas for output
    keep_left : bool, default=True
        Keep only left hemisphere data
    smooth : bool, default=True
        Apply spatial smoothing
    cross_species : bool, default=False
        Perform cross-species transformation
    transform : bool, default=False
        Transform to different space using neuromaps
    transform_method : str, default='linear'
        Interpolation method ('linear', 'nearest')
    smooth_param : dict, optional
        Parameters for smoothing (radius, sigma, etc.)
        
    Returns
    -------
    data_processed : pd.DataFrame
        Processed data in target atlas
        
    Examples
    --------
    Load CSV file:
    
    >>> data = load_data(
    ...     'cell_types.csv',
    ...     space='fslr',
    ...     trg='FGC'
    ... )
    
    Load NIfTI with transformation:
    
    >>> data = load_data(
    ...     'zstat.nii.gz',
    ...     space='mni152',
    ...     atlas='DK',
    ...     trg='FGC',
    ...     smooth=True
    ... )
    
    Load GIFTI surface data:
    
    >>> data = load_data(
    ...     'thickness.gii',
    ...     space='fslr',
    ...     trg='FGC'
    ... )
    
    See Also
    --------
    load_data_list : Batch loading for multiple files
    surf_relabel : Transform between surface atlases
    vol_relabel : Transform between volumetric atlases
    """
    valid_spaces = {'mni152', 'fslr', 'fsaverage', 'civet'}
    if space.lower() not in valid_spaces:
        raise ValueError(f"space must be one of {sorted(valid_spaces)}; got {space!r}")
    space = space.lower()
    if transform_method not in {'linear', 'nearest'}:
        raise ValueError("transform_method must be 'linear' or 'nearest'")
    if keep_left is not True:
        raise ValueError("HomoloMap analyses are left-hemisphere only; keep_left must be True")
    if not isinstance(smooth, bool):
        raise TypeError("smooth and keep_left must be boolean")
    if atlas is None:
        atlas = trg
    
    if smooth_param is None:
        smooth_param = {}
    
    # Load data
    if isinstance(data, (str, os.PathLike)):
        input_path = Path(data)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")
        try:
            data = nib.load(str(input_path))
        except ImageFileError:
            try:
                data = pd.read_csv(input_path, index_col=0)
            except Exception as e:
                raise ValueError(f"Cannot load {input_path} as an image or table: {e}") from e

    if isinstance(data, pd.Series):
        data = data.to_frame(name=data.name or 'feature')
    
    # Process DataFrame
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        try:
            df.index = pd.Index(pd.to_numeric(df.index), dtype='int64')
        except (TypeError, ValueError) as exc:
            raise ValueError("DataFrame index must contain integer ROI labels") from exc
        if not df.index.is_unique:
            raise ValueError("DataFrame ROI labels must be unique")
        try:
            df = df.apply(pd.to_numeric, errors='raise')
        except (TypeError, ValueError) as exc:
            raise ValueError("DataFrame values must be numeric") from exc
        if df.isna().all(axis=None):
            raise ValueError("Input DataFrame contains no numeric values")
    
    # Process GIFTI
    elif isinstance(data, nib.GiftiImage):
        # Transform space if needed
        if transform and not NEUROMAPS_AVAILABLE:
            raise ImportError("neuromaps is required when transform=True")
        if transform:
            if space == 'fsaverage':
                data = nm_transforms.fsaverage_to_fslr(
                    data, '32k', hemi='L', method=transform_method
                )[0]
            elif space == 'fslr':
                data = nm_transforms.fslr_to_fslr(
                    data, '32k', hemi='L', method=transform_method
                )[0]
            elif space == 'civet':
                data = nm_transforms.civet_to_fslr(
                    data, '32k', hemi='L', method=transform_method
                )[0]
        
        # Get parcellation
        try:
            parc = fetch_parc(key=atlas)
        except (FileExistsError, FileNotFoundError):
            if path is None:
                raise
            parc = nib.load(str(path))
            atlas = str(path)
        
        parc_data = parc.agg_data()
        labels = np.trim_zeros(np.unique(parc_data)).astype(int)
        
        # Extract data
        surf_data = np.asarray(data.agg_data(), dtype=float)
        surf_data = np.squeeze(surf_data)
        if surf_data.ndim != 1:
            raise ValueError("GIFTI input must contain one scalar value per vertex")
        
        if parc_data.shape != surf_data.shape:
            raise ValueError('Provided data do not match shape of label')
        
        # Aggregate to parcels
        df = pd.DataFrame(index=labels, columns=['feature'])
        for label in labels:
            mask = parc_data == label
            df.loc[label, 'feature'] = np.nanmean(surf_data[mask])
    
    # Process NIfTI
    elif isinstance(data, nib.Nifti1Image):
        if not NILEARN_AVAILABLE:
            raise ImportError("nilearn required for volumetric data")
        
        # Transform space if needed
        if transform and not NEUROMAPS_AVAILABLE:
            raise ImportError("neuromaps is required when transform=True")
        if transform:
            if space == 'mni152':
                data = nm_transforms.mni152_to_mni152(
                    data, target='1mm', method=transform_method
                )
        
        # Load atlas
        try:
            atlas_path = fetch_annot(atlas=atlas, annot=False)
            parc = nib.load(atlas_path)
        except (FileExistsError, FileNotFoundError):
            if path is None:
                raise
            parc = nib.load(str(path))
            atlas = str(path)

        data_resampled = nli.resample_to_img(
            data, parc, 
            interpolation='continuous'
        )
        
        # Get data arrays
        parc_data = np.squeeze(parc.get_fdata()).astype(int)
        vol_data = np.squeeze(data_resampled.get_fdata().astype(float))
        
        # Verify shape match
        if parc_data.shape != vol_data.shape:
            raise ValueError(
                f'Shape mismatch: data {vol_data.shape} vs atlas {parc_data.shape}'
            )
        
        # Get unique labels (excluding 0/background)
        labels = np.unique(parc_data)
        labels = labels[labels > 0]  # Remove background (0)

        df = pd.DataFrame(index=labels, columns=['feature'])
        
        for label in labels:
            # Get mask for this region
            mask = parc_data == label
            region_values = vol_data[mask]
            
            region_mean = np.nanmean(region_values) if region_values.size else np.nan
            
            df.loc[label, 'feature'] = region_mean        
    
    else:
        raise ValueError(
            f"Unsupported data type: {type(data)}. "
            "Supported types: DataFrame, NIfTI, GIFTI, or file path"
        )
    
    # Keep left hemisphere
    if keep_left:
        if space in ['fslr', 'fsaverage', 'civet'] and not Path(str(atlas)).exists():
            labels_dict = fetch_parc(key=atlas).labeltable.get_labels_as_dict().keys()
            labels = np.array(list(labels_dict))
            labels = labels[labels > 0]
            df = df.loc[df.index.isin(labels)]
        
        elif space == 'mni152' and not Path(str(atlas)).exists():
            if atlas in ['BN', 'FGC', 'MacBN_human']:
                atlas_path, annot = fetch_annot(atlas=atlas, annot=True)
                labels = load_volume_atlas(
                    atlas_path, annot, hemisphere='left'
                )['roi_labels']
            else:
                labels = load_volume_atlas(
                    atlas, hemisphere='left'
                )['roi_labels']
            df = df.loc[df.index.isin(labels)]
    
    # Transform atlas if needed
    if str(atlas) != str(trg):
        if space == 'mni152':
            df = vol_relabel(src=atlas, trg=trg, data=df, cross_species=cross_species)
        elif space in ['fslr', 'fsaverage', 'civet']:
            df = surf_relabel(df, src=atlas, trg=trg, cross_species=cross_species)
    
    # Smooth if requested
    if smooth and parc_smooth is not None:
        df = parc_smooth(
            df,
            mesh=fetch_fslr(surf='inflated', return_path=True),
            parc=fetch_parc(key=trg),
            **smooth_param
        )
    
    df = df.astype(float).sort_index()
    df.index.name = df.index.name or 'roi'
    return df


def load_data_list(
    data: Union[List, Tuple],
    space: str = 'mni152',
    atlas: Optional[str] = None,
    path: Optional[Union[str, Path]] = None,
    trg: str = 'FGC',
    keep_left: bool = True,
    smooth: bool = True,
    cross_species: bool = False,
    transform: bool = False,
    transform_method: str = 'linear',
    smooth_param: Optional[dict] = None
) -> pd.DataFrame:
    """
    Load and process multiple brain data files.
    
    Parameters
    ----------
    data : list or tuple
        List of image objects or file paths
    space : str, default='mni152'
        Space of input data
    atlas : str, optional
        Source atlas name
    path : str or Path, optional
        Path to atlas file
    trg : str, default='FGC'
        Target atlas name
    keep_left : bool, default=True
        Keep only left hemisphere
    smooth : bool, default=True
        Apply smoothing
    cross_species : bool, default=False
        Cross-species transformation
    transform : bool, default=False
        Transform to standard space
    transform_method : str, default='linear'
        Transformation method
    smooth_param : dict, optional
        Smoothing parameters
        
    Returns
    -------
    processed_data : pd.DataFrame
        Concatenated processed data from all files
        
    Examples
    --------
    >>> files = ['subject1.nii.gz', 'subject2.nii.gz', 'subject3.nii.gz']
    >>> data = load_data_list(files, space='mni152', trg='FGC')
    >>> print(data.shape)  # (n_regions, 3)
    
    See Also
    --------
    load_data : Load single file
    """
    if not isinstance(data, (list, tuple)):
        raise ValueError("Input data must be a list or tuple of images or paths.")
    
    if not data:
        raise ValueError("Input data list must not be empty")
    processed_data = []
    
    for position, d in enumerate(data):
        processed = load_data(
            data=d,
            space=space,
            atlas=atlas,
            path=path,
            trg=trg,
            keep_left=keep_left,
            smooth=smooth,
            cross_species=cross_species,
            transform=transform,
            transform_method=transform_method,
            smooth_param=smooth_param
        )
        if processed.shape[1] == 1:
            if isinstance(d, (str, os.PathLike)):
                name = Path(d).name
                for suffix in ('.nii.gz', '.func.gii', '.shape.gii', '.label.gii', '.csv'):
                    if name.endswith(suffix):
                        name = name[:-len(suffix)]
                        break
            else:
                name = f'feature_{position}'
            processed = processed.rename(columns={processed.columns[0]: name})
        processed_data.append(processed)
    
    # Concatenate
    processed_data = pd.concat(processed_data, axis=1, join='outer')
    if not processed_data.columns.is_unique:
        counts = {}
        columns = []
        for column in processed_data.columns:
            counts[column] = counts.get(column, 0) + 1
            columns.append(column if counts[column] == 1 else f'{column}_{counts[column]}')
        processed_data.columns = columns
    
    return processed_data
