# HomoloMap/transforms/smoothing.py
"""
Spatial smoothing operations for brain data.

This module provides functions for smoothing both surface (vertex-level) and
parcellated brain data using geodesic distance-based kernels.
"""

import warnings
import os
import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, Literal
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm

from scipy.spatial import distance_matrix


def parc_smooth(
    parc_data: pd.DataFrame,
    roi_disc: Optional[pd.DataFrame] = None,
    radius: float = 6.0,
    method: Literal['mean', 'gaussian', 'median'] = 'gaussian',
    sigma: float = 5.0,
    mesh: Optional[Union[str, Path]] = None,
    parc: Optional = None,
    n_jobs: int = -1,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Apply spatial smoothing to parcellated brain data.
    
    Smooths each parcel's values by incorporating information from neighboring
    parcels, weighted by their geodesic distance on the cortical surface.
    
    Parameters
    ----------
    parc_data : pd.DataFrame
        Parcellated brain data with regions as rows and features as columns.
        Shape: (n_regions, n_features)
    roi_disc : pd.DataFrame, optional
        Precomputed geodesic distance matrix between ROIs.
        If None, will be computed from mesh and parc.
        Shape: (n_regions, n_regions)
    radius : float, default=6.0
        Smoothing radius in millimeters. Only parcels within this distance
        contribute to smoothing.
    method : {'mean', 'gaussian', 'median'}, default='gaussian'
        Smoothing method:
        - 'mean': Unweighted average of neighbors within radius
        - 'gaussian': Gaussian-weighted average (recommended)
        - 'median': Median of neighbors within radius
    sigma : float, default=5.0
        Standard deviation for Gaussian kernel (mm).
        Only used when method='gaussian'.
    mesh : str or Path, optional
        Path to surface mesh file. Required if roi_disc is None.
    parc : nibabel.GiftiImage, optional
        Parcellation image. Required if roi_disc is None.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all available cores.
    verbose : bool, default=False
        Show progress bar
        
    Returns
    -------
    smoothed_data : pd.DataFrame
        Smoothed parcellated data with same shape as input.
        Index and columns preserved from input.
        
    Raises
    ------
    ValueError
        If roi_disc is None and mesh or parc not provided
        If method is not valid
    TypeError
        If parc_data is not a DataFrame
        
    Notes
    -----
    **Smoothing Methods:**
    
    1. **Mean smoothing:**
       Equal weights for all neighbors within radius
       
       .. math:: \\bar{x}_i = \\frac{1}{|N_i|} \\sum_{j \\in N_i} x_j
       
    2. **Gaussian smoothing (recommended):**
       Weights decay with distance
       
       .. math:: w(d) = \\exp\\left(-\\frac{d^2}{2\\sigma^2}\\right)
       
       .. math:: \\bar{x}_i = \\frac{\\sum_{j \\in N_i} w(d_{ij}) x_j}{\\sum_{j \\in N_i} w(d_{ij})}
       
    3. **Median smoothing:**
       Robust to outliers but slower
       
       .. math:: \\bar{x}_i = \\text{median}(\\{x_j : j \\in N_i\\})
    
    **Performance Tips:**
    
    - Precompute roi_disc once if smoothing multiple datasets
    - Use n_jobs=-1 for parallel processing
    - Gaussian smoothing is fastest with best results
    - Larger radius = more smoothing but slower computation
    
    Examples
    --------
    Basic smoothing with automatic distance computation:
    
    >>> from HomoloMap.transforms import parc_smooth
    >>> from HomoloMap.datasets import fetch_fslr, fetch_parc
    >>> 
    >>> # Load data
    >>> mesh = fetch_fslr(surf='inflated', return_path=True)
    >>> parc = fetch_parc(key='FGC')
    >>> 
    >>> # Smooth
    >>> smoothed = parc_smooth(
    ...     celltype_data,
    ...     mesh=mesh,
    ...     parc=parc,
    ...     radius=10,
    ...     method='gaussian',
    ...     sigma=5
    ... )
    
    Efficient smoothing with precomputed distances:
    
    >>> # Compute distance matrix once
    >>> roi_disc = get_parcel_geodist(mesh, parc, n_jobs=-1)
    >>> 
    >>> # Smooth multiple datasets using same distances
    >>> for dataset in [data1, data2, data3]:
    ...     smoothed = parc_smooth(
    ...         dataset,
    ...         roi_disc=roi_disc,
    ...         radius=10,
    ...         n_jobs=-1
    ...     )
    
    Compare smoothing methods:
    
    >>> methods = ['mean', 'gaussian', 'median']
    >>> results = {}
    >>> 
    >>> for method in methods:
    ...     results[method] = parc_smooth(
    ...         data,
    ...         roi_disc=roi_disc,
    ...         radius=10,
    ...         method=method
    ...     )
    
    See Also
    --------
    get_parcel_geodist : Compute geodesic distance matrix
    smooth_single_roi : Smooth a single ROI
    disc_smooth : Smooth vertex-level data
    """
    # Validate input
    if not isinstance(parc_data, pd.DataFrame):
        raise TypeError(
            f"parc_data must be pd.DataFrame, got {type(parc_data).__name__}"
        )
    
    valid_methods = ['mean', 'gaussian', 'median']
    if method not in valid_methods:
        raise ValueError(
            f"method must be one of {valid_methods}, got '{method}'"
        )
    if parc in ['FGC','BN']:
        root_path = Path(__file__).parent
        path = root_path.parent / 'datasets' / 'surfaces' / 'parcellations'
        roi_disc = pd.read_csv(os.path.join(path,f'{parc}_infl_geodisc.csv'), index_col=0)
    # Compute distance matrix if not provided
    if roi_disc is None:
        if mesh is None or parc is None:
            raise ValueError(
                'mesh and parc must be provided when roi_disc is None'
            )
        
        if verbose:
            print("Computing geodesic distance matrix...")
        
        roi_disc = get_parcel_geodist(mesh, parc, n_jobs=n_jobs)
    
    # Ensure distance matrix matches data
    roi_disc = roi_disc.loc[parc_data.index, parc_data.index]
    
    # Smooth each ROI in parallel
    if verbose:
        print(f"Smoothing {len(parc_data)} regions with {method} kernel...")
    
    iterator = tqdm(parc_data.index, desc="Smoothing") if verbose else parc_data.index
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(smooth_single_roi)(
            roi, parc_data, roi_disc, radius, method, sigma
        )
        for roi in iterator
    )
    
    # Reconstruct DataFrame
    smoothed_data = np.zeros_like(parc_data.values)
    for i, data in enumerate(results):
        smoothed_data[i, :] = data
    
    return pd.DataFrame(
        smoothed_data,
        index=parc_data.index,
        columns=parc_data.columns
    )


def smooth_single_roi(
    roi: int,
    parc_data: pd.DataFrame,
    roi_disc: pd.DataFrame,
    radius: float,
    method: str = 'gaussian',
    sigma: float = 5.0
) -> np.ndarray:
    """
    Smooth a single ROI using neighboring parcels.
    
    This is a helper function called by parc_smooth for each ROI.
    
    Parameters
    ----------
    roi : int
        ROI label to smooth
    parc_data : pd.DataFrame
        Full parcellation data
    roi_disc : pd.DataFrame
        Geodesic distance matrix
    radius : float
        Smoothing radius (mm)
    method : str
        Smoothing method ('mean', 'gaussian', 'median')
    sigma : float
        Gaussian kernel width (mm)
        
    Returns
    -------
    smoothed : np.ndarray
        Smoothed values for this ROI, shape (n_features,)
    """
    # Get distances to all other ROIs
    disc = roi_disc.loc[roi, :].values
    labels = roi_disc.index.values
    
    # Find neighbors within radius (excluding self)
    mask = (disc < radius) & (disc > 0)
    neighbor_labels = labels[mask]
    neighbor_disc = disc[mask]
    
    # Get current data
    current_data = parc_data.loc[roi, :].values
    
    # If no neighbors, return current data
    if len(neighbor_labels) == 0:
        return current_data
    
    # Apply smoothing based on method
    if method == 'mean':
        # Unweighted average
        neighbor_data = parc_data.loc[neighbor_labels, :].values
        smoothed = (np.nansum(neighbor_data, axis=0) + current_data) / (len(neighbor_labels) + 1)
        
    elif method == 'gaussian':
        # Gaussian-weighted average
        # Include current ROI with distance 0
        all_labels = np.append(neighbor_labels, roi)
        all_distances = np.append(neighbor_disc, 0.0)
        
        # Compute Gaussian weights
        weights = np.exp(-(all_distances ** 2) / (2 * sigma ** 2))
        weights = weights / np.nansum(weights)  # Normalize
        
        # Weighted average
        all_data = parc_data.loc[all_labels, :].values
        smoothed = np.nansum(all_data * weights[:, np.newaxis], axis=0)
        
    elif method == 'median':
        # Median (robust to outliers)
        neighbor_data = parc_data.loc[neighbor_labels, :].values
        all_data = np.vstack([neighbor_data, current_data])
        smoothed = np.nanmedian(all_data, axis=0)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return smoothed


def get_parcel_geodist(
    mesh: Union[str, Path],
    parc,
    n_jobs: int = -1,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Compute geodesic distance matrix between parcel centroids.
    
    Calculates the geodesic (surface) distance between the centroid
    of each parcel pair on the cortical surface.
    
    Parameters
    ----------
    mesh : str or Path
        Path to cortical mesh file (.gii)
    parc : nibabel.GiftiImage
        Parcellation image
    n_jobs : int, default=-1
        Number of parallel jobs
    verbose : bool, default=False
        Show progress messages
        
    Returns
    -------
    geodist_matrix : pd.DataFrame
        Geodesic distance matrix, shape (n_parcels, n_parcels)
        Index and columns are parcel labels
        
    Notes
    -----
    This function:
    1. Computes the centroid of each parcel
    2. For each centroid, computes geodesic distance to all vertices
    3. Extracts distances between centroids
    
    The result is symmetric with zeros on the diagonal.
    
    Distances are in millimeters (assuming mesh coordinates are in mm).
    
    Examples
    --------
    Compute and cache distance matrix:
    
    >>> from HomoloMap.transforms import get_parcel_geodist
    >>> from HomoloMap.datasets import fetch_fslr, fetch_parc
    >>> 
    >>> mesh = fetch_fslr(surf='inflated', return_path=True)
    >>> parc = fetch_parc(key='FGC')
    >>> 
    >>> # Compute once
    >>> roi_disc = get_parcel_geodist(mesh, parc, n_jobs=-1)
    >>> 
    >>> # Save for reuse
    >>> roi_disc.to_csv('FGC_geodesic_distances.csv')
    >>> 
    >>> # Load later
    >>> roi_disc = pd.read_csv('FGC_geodesic_distances.csv', index_col=0)
    
    Inspect distances:
    
    >>> print(f"Distance matrix shape: {roi_disc.shape}")
    >>> print(f"Min distance (excluding diagonal): {roi_disc[roi_disc>0].min().min():.2f} mm")
    >>> print(f"Max distance: {roi_disc.max().max():.2f} mm")
    >>> print(f"Mean distance: {roi_disc[roi_disc>0].mean().mean():.2f} mm")
    
    See Also
    --------
    parc_smooth : Use distance matrix for smoothing
    get_gd_parc_centroids : Compute single parcel centroid
    """
    from .geometry import get_gd_parc_centroids, get_gd_disc
    
    # Extract unique parcel labels (excluding background)
    labels = np.trim_zeros(np.unique(parc.agg_data()))
    
    if verbose:
        print(f"Computing distances for {len(labels)} parcels...")
    
    # Compute centroids for each parcel
    if verbose:
        print("  Computing parcel centroids...")
    
    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(get_gd_parc_centroids)(mesh, parc, label)
        for label in labels
    )
    centroids = [result[0] for result in results]
    
    # Compute geodesic distances from each centroid
    if verbose:
        print("  Computing geodesic distances...")
    
    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(get_gd_disc)(mesh, centroid, radius=None, return_dist=True)
        for centroid in centroids
    )
    
    # Build distance matrix
    # results[i] is (distances, mask) for centroid i
    # Extract distances at all centroid locations
    geodist_matrix = np.zeros((len(labels), len(labels)))
    for i in range(len(results)):
        distances, _ = results[i]
        geodist_matrix[i, :] = distances[centroids]
    
    # Create DataFrame with labels
    geodist_matrix = pd.DataFrame(
        geodist_matrix,
        index=labels,
        columns=labels
    )
    
    if verbose:
        print(f"  Distance range: [{geodist_matrix[geodist_matrix>0].min().min():.1f}, "
              f"{geodist_matrix.max().max():.1f}] mm")
    
    return geodist_matrix


def disc_smooth(
    surface_data: np.ndarray,
    smooth_disc_radius: float,
    approach: Literal['euclidean', 'geodesic'] = 'euclidean',
    mesh: Optional[Union[str, Path]] = None,
    n_jobs: int = 1,
    verbose: bool = False
) -> Tuple[np.ndarray, Union[np.ndarray, dict]]:
    """
    Smooth vertex-level surface data using distance-based discs.
    
    Smooths data at each vertex by averaging values within a disc
    (sphere in 3D space or geodesic neighborhood on surface).
    
    Parameters
    ----------
    surface_data : np.ndarray
        Vertex-level data, shape (n_vertices,) or (n_vertices, n_features)
        For fsLR 32k, n_vertices should be 32492
    smooth_disc_radius : float
        Radius of smoothing disc in millimeters
    approach : {'euclidean', 'geodesic'}, default='euclidean'
        Distance metric:
        - 'euclidean': Straight-line distance in 3D space (faster)
        - 'geodesic': Distance along cortical surface (more accurate but slower)
    mesh : str or Path, optional
        Path to surface mesh. Required for geodesic smoothing.
        For euclidean, uses fsLR 32k inflated surface if not provided.
    n_jobs : int, default=1
        Number of parallel jobs (for geodesic approach)
    verbose : bool, default=False
        Show progress messages
        
    Returns
    -------
    smoothed_data : np.ndarray
        Smoothed data, same shape as input
    discs : np.ndarray or dict
        For euclidean: boolean array (n_vertices, n_vertices) indicating neighbors
        For geodesic: dict mapping vertex index to neighbor indices
        
    Notes
    -----
    **Euclidean vs Geodesic:**
    
    - **Euclidean** is much faster but treats cortex as embedded in 3D space
    - **Geodesic** respects cortical folding but requires many distance computations
    
    For most applications, euclidean smoothing with appropriate radius gives
    good results and is 100x faster.
    
    **Performance:**
    
    For fsLR 32k (32,492 vertices):
    - Euclidean: ~5 seconds
    - Geodesic: ~5-10 minutes
    
    Examples
    --------
    Basic euclidean smoothing:
    
    >>> from HomoloMap.transforms import disc_smooth
    >>> 
    >>> # Load vertex data (e.g., from GIFTI)
    >>> vertex_data = ...  # shape (32492,)
    >>> 
    >>> # Smooth with 5mm radius
    >>> smoothed, discs = disc_smooth(
    ...     vertex_data,
    ...     smooth_disc_radius=5.0,
    ...     approach='euclidean'
    ... )
    
    Geodesic smoothing (more accurate):
    
    >>> from HomoloMap.datasets import fetch_fslr
    >>> 
    >>> mesh = fetch_fslr(surf='inflated', return_path=True)
    >>> smoothed, discs = disc_smooth(
    ...     vertex_data,
    ...     smooth_disc_radius=5.0,
    ...     approach='geodesic',
    ...     mesh=mesh,
    ...     n_jobs=-1
    ... )
    
    Smooth multiple features:
    
    >>> # Data with multiple features
    >>> vertex_data = np.random.randn(32492, 5)
    >>> smoothed, discs = disc_smooth(vertex_data, smooth_disc_radius=5.0)
    >>> print(smoothed.shape)  # (32492, 5)
    
    See Also
    --------
    parc_smooth : Smooth parcellated data
    get_gd_disc : Get geodesic distance disc for single vertex
    """
    # Validate input
    if surface_data.shape[0] != 32492:
        warnings.warn(
            f"Surface data has {surface_data.shape[0]} vertices. "
            f"Expected 32492 for fsLR 32k surface.",
            UserWarning
        )
    
    n_vertices = surface_data.shape[0]
    
    # Ensure 2D array
    if surface_data.ndim == 1:
        surface_data = surface_data[:, np.newaxis]
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Initialize output
    smoothed_data = np.zeros_like(surface_data)
    
    if approach == 'euclidean':
        # Use Euclidean distance in 3D space
        if mesh is None:
            from ..datasets import fetch_fslr
            if verbose:
                print("Loading fsLR inflated mesh...")
            mesh_obj = fetch_fslr(surf='inflated')
        else:
            import nibabel as nib
            mesh_obj = nib.load(mesh)
        
        vertices, _ = mesh_obj.agg_data()
        
        if verbose:
            print("Computing Euclidean distance matrix...")
        
        # Compute pairwise distances
        ed_matrix = distance_matrix(vertices, vertices)
        
        # Create disc mask
        discs = ed_matrix < smooth_disc_radius
        
        if verbose:
            print("Smoothing vertices...")
        
        # Smooth each vertex
        for vertex in range(n_vertices):
            if np.isnan(surface_data[vertex]).all():
                # Skip vertices with all NaN
                smoothed_data[vertex] = np.nan
            else:
                disc = discs[vertex, :]
                smoothed_data[vertex] = np.nanmean(
                    surface_data[disc],
                    axis=0
                )
    
    elif approach == 'geodesic':
        # Use geodesic distance along surface
        if mesh is None:
            raise ValueError("mesh must be provided for geodesic approach")
        
        from .geometry import get_gd_disc
        
        if verbose:
            print("Computing geodesic discs...")
        
        discs = {}
        
        # Process vertices
        iterator = tqdm(range(n_vertices), desc="Vertices") if verbose else range(n_vertices)
        
        for vertex in iterator:
            if np.isnan(surface_data[vertex]).all():
                smoothed_data[vertex] = np.nan
                discs[vertex] = []
            else:
                # Get geodesic disc
                disc = get_gd_disc(mesh, vertex, smooth_disc_radius)
                discs[vertex] = disc
                
                # Smooth
                smoothed_data[vertex] = np.nanmean(
                    surface_data[disc],
                    axis=0
                )
    
    else:
        raise ValueError(
            f"approach must be 'euclidean' or 'geodesic', got '{approach}'"
        )
    
    # Squeeze if input was 1D
    if squeeze_output:
        smoothed_data = smoothed_data.squeeze()
    
    return smoothed_data, discs


def adaptive_smooth(
    parc_data: pd.DataFrame,
    roi_disc: pd.DataFrame,
    target_smoothness: float = 0.8,
    max_radius: float = 15.0,
    method: str = 'gaussian',
    sigma: float = 5.0,
    n_jobs: int = -1
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Adaptive smoothing with variable radius per region.
    
    Adjusts smoothing radius for each region to achieve target smoothness level.
    Useful for data with variable spatial resolution or SNR.
    
    Parameters
    ----------
    parc_data : pd.DataFrame
        Input parcellation data
    roi_disc : pd.DataFrame
        Geodesic distance matrix
    target_smoothness : float, default=0.8
        Target smoothness level (0-1). Higher = more smoothing.
    max_radius : float, default=15.0
        Maximum smoothing radius (mm)
    method : str, default='gaussian'
        Smoothing method
    sigma : float, default=5.0
        Gaussian kernel width
    n_jobs : int, default=-1
        Parallel jobs
        
    Returns
    -------
    smoothed_data : pd.DataFrame
        Adaptively smoothed data
    radii_used : pd.Series
        Smoothing radius used for each region
        
    Examples
    --------
    >>> from HomoloMap.transforms import adaptive_smooth
    >>> 
    >>> smoothed, radii = adaptive_smooth(
    ...     data,
    ...     roi_disc,
    ...     target_smoothness=0.8,
    ...     max_radius=15.0
    ... )
    >>> 
    >>> print(f"Radius range: [{radii.min():.1f}, {radii.max():.1f}] mm")
    """
    # Start with small radius
    test_radii = np.linspace(2, max_radius, 10)
    
    def compute_smoothness_for_radius(roi, radius):
        """Compute correlation between original and smoothed data."""
        smoothed = smooth_single_roi(
            roi, parc_data, roi_disc, radius, method, sigma
        )
        original = parc_data.loc[roi].values
        
        # Correlation as smoothness metric
        if np.std(original) > 0 and np.std(smoothed) > 0:
            corr = np.corrcoef(original, smoothed)[0, 1]
            return corr
        return 1.0
    
    # Determine optimal radius for each ROI
    radii_used = {}
    
    for roi in parc_data.index:
        # Try different radii
        smoothness_levels = [
            compute_smoothness_for_radius(roi, r)
            for r in test_radii
        ]
        
        # Find radius closest to target
        idx = np.argmin(np.abs(np.array(smoothness_levels) - target_smoothness))
        radii_used[roi] = test_radii[idx]
    
    # Apply smoothing with chosen radii
    results = Parallel(n_jobs=n_jobs)(
        delayed(smooth_single_roi)(
            roi, parc_data, roi_disc, radii_used[roi], method, sigma
        )
        for roi in parc_data.index
    )
    
    smoothed_data = pd.DataFrame(
        np.array(results),
        index=parc_data.index,
        columns=parc_data.columns
    )
    
    radii_series = pd.Series(radii_used)
    
    return smoothed_data, radii_series


def iterative_smooth(
    parc_data: pd.DataFrame,
    roi_disc: pd.DataFrame,
    n_iterations: int = 3,
    radius: float = 6.0,
    method: str = 'gaussian',
    sigma: float = 5.0,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Apply smoothing iteratively for stronger effect.
    
    Repeatedly applies smoothing operation. Equivalent to larger kernel
    but may preserve features better.
    
    Parameters
    ----------
    parc_data : pd.DataFrame
        Input data
    roi_disc : pd.DataFrame
        Distance matrix
    n_iterations : int, default=3
        Number of smoothing iterations
    radius : float, default=6.0
        Smoothing radius per iteration
    method : str, default='gaussian'
        Smoothing method
    sigma : float, default=5.0
        Kernel width
    n_jobs : int, default=-1
        Parallel jobs
        
    Returns
    -------
    smoothed_data : pd.DataFrame
        Iteratively smoothed data
        
    Examples
    --------
    >>> from HomoloMap.transforms import iterative_smooth
    >>> 
    >>> # 3 iterations with small radius
    >>> smoothed = iterative_smooth(
    ...     data,
    ...     roi_disc,
    ...     n_iterations=3,
    ...     radius=4.0
    ... )
    """
    smoothed = parc_data.copy()
    
    for i in range(n_iterations):
        smoothed = parc_smooth(
            smoothed,
            roi_disc=roi_disc,
            radius=radius,
            method=method,
            sigma=sigma,
            n_jobs=n_jobs
        )
    
    return smoothed