# HomoloMap/stats/nulls.py
"""
Spatial null models for brain map comparisons.

This module implements spatial permutation tests (spin tests) that
preserve spatial autocorrelation structure when generating null
distributions for statistical testing.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
from sklearn.utils.validation import check_random_state
from scipy.spatial import distance_matrix, cKDTree
from scipy import optimize


def gen_spinsamples(
    coords: np.ndarray,
    n_rotate: int = 1000,
    check_duplicates: bool = True,
    method: str = 'Alexander-Bloch',
    seed: Optional[int] = None,
    verbose: bool = False,
    return_cost: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate spin resampling indices for spatial permutation testing.
    
    This function implements the spatial permutation framework from
    Alexander-Bloch et al. (2018) for creating null distributions that
    preserve spatial autocorrelation structure.
    
    Parameters
    ----------
    coords : np.ndarray, shape (n_parcels, 3)
        Coordinates of parcels on sphere (must be on unit sphere)
    n_rotate : int, default=1000
        Number of rotations to generate
    check_duplicates : bool, default=True
        Whether to check for and avoid duplicate rotations.
        May increase runtime but ensures unique permutations.
    method : {'Alexander-Bloch', 'vasa', 'hungarian'}, default='Alexander-Bloch'
        Rotation matching method:
        - 'Alexander-Bloch': Nearest neighbor matching (fastest)
        - 'vasa': Min-max matching from Vasa et al. (2018)
        - 'hungarian': Optimal assignment (slowest, most accurate)
    seed : int, optional
        Random seed for reproducibility
    verbose : bool, default=False
        Whether to print progress messages
    return_cost : bool, default=False
        Whether to return assignment costs (Euclidean distances)
        
    Returns
    -------
    spinsamples : np.ndarray, shape (n_parcels, n_rotate)
        Resampling indices for each rotation
    cost : np.ndarray, shape (n_parcels, n_rotate), optional
        Assignment costs if return_cost=True
        
    Raises
    ------
    ValueError
        If coords is not 2D with 3 columns
        If method is not valid
        
    Warnings
    --------
    If unable to generate n_rotate unique rotations after 500 attempts,
    a warning is raised and duplicate rotations may be included.
        
    Notes
    -----
    The spin test preserves spatial autocorrelation by rotating parcels
    on a sphere and reassigning values based on the rotated positions.
    This is critical for valid statistical inference on spatially
    autocorrelated brain maps.
    
    References
    ----------
    .. [1] Alexander-Bloch, A. F., et al. (2018). "On testing for spatial
           correspondence between maps of human brain structure and function."
           NeuroImage, 178, 540-551.
    .. [2] Vasa, F., et al. (2018). "Adolescent tuning of association cortex
           in human structural brain networks." Cerebral Cortex, 28(1), 281-294.
           
    Examples
    --------
    Generate spin samples for correlation testing:
    
    >>> from HomoloMap.stats import gen_spinsamples
    >>> from HomoloMap.transforms import get_parcel_centroids
    >>> from HomoloMap.datasets import fetch_fslr, fetch_parc
    >>> 
    >>> # Get parcel coordinates on sphere
    >>> surf = fetch_fslr(surf='sphere', return_path=True)
    >>> parc = fetch_parc(key='FGC')
    >>> coords = get_parcel_centroids(surf, parc, method='surface')
    >>> 
    >>> # Generate 1000 rotations
    >>> spins = gen_spinsamples(coords, n_rotate=1000, seed=42)
    >>> 
    >>> # Use for permutation testing
    >>> original_corr = np.corrcoef(data1, data2)[0, 1]
    >>> null_corrs = [
    ...     np.corrcoef(data1[spins[:, i]], data2)[0, 1]
    ...     for i in range(1000)
    ... ]
    >>> p_value = (1 + np.sum(np.abs(null_corrs) >= np.abs(original_corr))) / 1001
    
    Compare different matching methods:
    
    >>> # Fast but approximate
    >>> spins_original = gen_spinsamples(coords, method='Alexander-Bloch')
    >>> 
    >>> # Optimal but slow
    >>> spins_hungarian = gen_spinsamples(coords, method='hungarian')
    
    See Also
    --------
    spin_data : High-level interface for generating spin samples
    _gen_rotation : Generate random rotation matrix
    """
    # Validate method
    valid_methods = ['Alexander-Bloch', 'vasa', 'hungarian']
    if method not in valid_methods:
        raise ValueError(
            f"Method '{method}' invalid. Must be one of {valid_methods}"
        )
    
    # Initialize random state
    seed = check_random_state(seed)
    coords = np.asanyarray(coords)
    
    # Validate coordinate shape
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"Coords must be 2D array with shape (n_parcels, 3), "
            f"got {coords.shape}"
        )
    
    n_parcels = len(coords)
    
    # Initialize output arrays
    spinsamples = np.zeros((n_parcels, n_rotate), dtype=np.int32)
    cost = np.zeros((n_parcels, n_rotate), dtype=np.float32)
    inds = np.arange(n_parcels, dtype=np.int32)
    
    # Generate rotations
    warned = False
    
    for n in range(n_rotate):
        count = 0
        duplicated = True
        
        if verbose:
            msg = f'Generating spin {n+1:>5} of {n_rotate:>5}'
            print(msg, end='\r', flush=True)
        
        # Try up to 500 times to get a unique rotation
        while duplicated and count < 500:
            count += 1
            duplicated = False
            
            # Generate random rotation
            rot = _gen_rotation(seed=seed)
            rotated_coords = coords @ rot
            
            # Match rotated to original coordinates
            if method == 'vasa':
                col = _match_vasa(coords, rotated_coords, cost[:, n])
                
            elif method == 'hungarian':
                dist = distance_matrix(coords, rotated_coords)
                row, col = optimize.linear_sum_assignment(dist)
                cost[:, n] = dist[row, col]
                
            elif method == 'Alexander-Bloch':
                dist, col = cKDTree(rotated_coords).query(coords, k=1)
                cost[:, n] = dist
            
            resampled = inds[col]
            
            # Check for duplicates
            if check_duplicates:
                # Check against previous rotations
                if np.any(np.all(resampled[:, None] == spinsamples[:, :n], axis=0)):
                    duplicated = True
                # Check if rotation is identity
                elif np.all(resampled == inds):
                    duplicated = True
        
        # Warn if couldn't get unique rotation
        if count >= 500 and not warned:
            warnings.warn(
                f'Could not generate {n_rotate} unique rotations. '
                f'Duplicate rotations may be present. '
                f'Consider reducing n_rotate or disabling check_duplicates.',
                stacklevel=2
            )
            warned = True
        
        spinsamples[:, n] = resampled
    
    if verbose:
        print()  # New line after progress
    
    if return_cost:
        return spinsamples, cost
    
    return spinsamples


def _gen_rotation(seed: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    Generate random 3D rotation matrix.
    
    Uses QR decomposition of random normal matrix to generate
    uniformly distributed rotation matrices.
    
    Parameters
    ----------
    seed : np.random.RandomState, optional
        Random state for reproducibility
        
    Returns
    -------
    rotation : np.ndarray, shape (3, 3)
        Random rotation matrix with det(R) = 1
        
    Notes
    -----
    This ensures proper rotations (no reflections) by checking
    the determinant and adjusting if necessary.
    """
    rs = check_random_state(seed)
    
    # QR decomposition of random matrix
    rotate, temp = np.linalg.qr(rs.normal(size=(3, 3)))
    rotate = rotate @ np.diag(np.sign(np.diag(temp)))
    
    # Ensure proper rotation (det = 1, not -1)
    if np.linalg.det(rotate) < 0:
        rotate[:, 0] = -rotate[:, 0]
    
    return rotate


def _match_vasa(
    coords: np.ndarray,
    rotated_coords: np.ndarray,
    cost: np.ndarray
) -> np.ndarray:
    """
    Match coordinates using Vasa et al. (2018) min-max method.
    
    Parameters
    ----------
    coords : np.ndarray
        Original coordinates
    rotated_coords : np.ndarray
        Rotated coordinates
    cost : np.ndarray
        Array to store costs (modified in place)
        
    Returns
    -------
    col : np.ndarray
        Column indices of matches
    """
    dist = distance_matrix(coords, rotated_coords)
    col = np.zeros(len(coords), dtype=np.int32)
    
    for _ in range(len(dist)):
        # Find parcel whose closest neighbor is farthest
        row = dist.min(axis=1).argmax()
        col[row] = dist[row].argmin()
        cost[row] = dist[row, col[row]]
        
        # Mark as assigned
        dist[row] = -np.inf
        dist[:, col[row]] = np.inf
    
    return col


def spin_data(
    data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    atlas: str = 'FGC',
    n_spins: int = 1000,
    seed: int = 1234,
    method: str = 'Alexander-Bloch',
    return_ind: bool = False,
    data_labels: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generate spatial permutations for brain data.
    
    High-level interface that handles atlas loading and coordinate
    extraction automatically.
    
    Parameters
    ----------
    data : np.ndarray or pd.DataFrame, optional
        Brain data to generate permutations for.
        If None, returns spin indices without applying to data.
    atlas : str, default='FGC'
        Atlas name or path for fetching parcellation
    n_spins : int, default=1000
        Number of permutations to generate
    seed : int, default=1234
        Random seed for reproducibility
    method : {'Alexander-Bloch', 'vasa', 'hungarian'}, default='Alexander-Bloch'
        Matching method for rotations
    return_ind : bool, default=False
        If True, return indices instead of permuted data
        
    Returns
    -------
    spins : np.ndarray
        If data is None or return_ind=True: spin indices, 
            shape (n_parcels, n_spins)
        If data provided and return_ind=False: permuted data,
            shape (n_parcels, n_spins, n_features)
            
    Examples
    --------
    Generate spin indices for an atlas:
    
    >>> from HomoloMap.stats import spin_data
    >>> spins = spin_data(atlas='FGC', n_spins=1000, seed=42)
    >>> print(f"Generated {spins.shape[1]} permutations")
    
    Generate permuted data directly:
    
    >>> import pandas as pd
    >>> data = pd.DataFrame(...)  # Your brain data
    >>> permuted = spin_data(data, atlas='FGC', n_spins=1000)
    >>> 
    >>> # Original correlation
    >>> r_orig = np.corrcoef(data['feature1'], data['feature2'])[0, 1]
    >>> 
    >>> # Null distribution
    >>> r_null = [
    ...     np.corrcoef(permuted[:, i, 0], data['feature2'])[0, 1]
    ...     for i in range(1000)
    ... ]
    
    See Also
    --------
    gen_spinsamples : Lower-level spin generation
    get_parcel_centroids : Extract coordinates for spin generation
    """
    from ..datasets import fetch_parc, fetch_fslr
    from ..transforms import get_parcel_centroids
    
    # Load atlas and surface
    parc = fetch_parc(key=atlas)
    surf = fetch_fslr(surf='sphere', return_path=True)
    
    # Get parcel coordinates on sphere
    coords, labels = get_parcel_centroids(
        surf, parc=parc, method='surface', return_labels=True
    )
    
    # If no data, just return spin indices
    if data is None:
        spins = gen_spinsamples(
            coords,
            n_rotate=n_spins,
            method=method,
            seed=seed
        )
        return spins
    
    # Handle DataFrame input
    if isinstance(data, pd.DataFrame):
        # shared_labels: label values present in both data and atlas
        shared_labels = labels
        # cortex: positional indices of shared_labels within the sorted labels
        # array — used to index coords, which is (n_atlas_labels, 3).
        # The old code did `data.index.intersection(labels) - 1` which assumed
        # consecutive 1-indexed labels; searchsorted works for any label set.
        cortex = np.arange(len(labels))
        # data_values rows align with cortex/coords[cortex]
        data_values = data.reindex(shared_labels).to_numpy(dtype=float)
    else:
        # NumPy array
        data_values = np.asarray(data)
        if data_values.ndim not in (1, 2):
            raise ValueError("data must be a 1D or 2D array")
        if data_labels is not None:
            data_labels = np.asarray(data_labels)
            if data_labels.ndim != 1 or len(data_labels) != len(data_values):
                raise ValueError("data_labels must contain one label per data row")
            if len(np.unique(data_labels)) != len(data_labels):
                raise ValueError("data_labels must be unique")
            data_values = pd.DataFrame(
                data_values, index=data_labels
            ).reindex(labels).to_numpy(dtype=float)
        elif len(data_values) != len(labels):
            raise ValueError(
                "NumPy data must contain one row per atlas parcel in atlas "
                "order; pass data_labels for sparse or partial ROI data"
            )
        cortex = np.arange(len(labels))

    # Generate spins for valid regions only
    spins = gen_spinsamples(
        coords[cortex, :],
        n_rotate=n_spins,
        method=method,
        seed=seed
    )

    if return_ind:
        return spins
    else:
        # Apply permutations to data
        return np.squeeze(data_values[spins])


class SpinTest:
    """
    Object-oriented interface for spin testing.
    
    This class encapsulates spin test generation and application,
    making it easier to reuse spin samples across multiple tests.
    
    Parameters
    ----------
    atlas : str, default='FGC'
        Atlas name for coordinate extraction
    n_spins : int, default=1000
        Number of rotations to generate
    method : str, default='Alexander-Bloch'
        Rotation matching method
    seed : int, optional
        Random seed for reproducibility
        
    Attributes
    ----------
    spins : np.ndarray
        Generated spin indices
    coords : np.ndarray
        Parcel coordinates used for generation
        
    Examples
    --------
    >>> from HomoloMap.stats import SpinTest
    >>> 
    >>> # Create spin test object
    >>> spinner = SpinTest(atlas='FGC', n_spins=1000, seed=42)
    >>> 
    >>> # Apply to multiple datasets
    >>> null_data1 = spinner.permute(data1)
    >>> null_data2 = spinner.permute(data2)
    >>> 
    >>> # Compute p-values
    >>> p_val = spinner.test_correlation(data1, data2)
    """
    
    def __init__(
        self,
        atlas: str = 'FGC',
        n_spins: int = 1000,
        method: str = 'Alexander-Bloch',
        seed: Optional[int] = None
    ):
        self.atlas = atlas
        self.n_spins = n_spins
        self.method = method
        self.seed = seed
        
        # Generate spins
        self.spins, self.coords, self.labels = self._generate_spins()
    
    def _generate_spins(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate spin indices and store coordinates."""
        from ..datasets import fetch_parc, fetch_fslr
        from ..transforms import get_parcel_centroids
        
        parc = fetch_parc(key=self.atlas)
        surf = fetch_fslr(surf='sphere', return_path=True)
        coords, labels = get_parcel_centroids(
            surf, parc=parc, method='surface', return_labels=True
        )
        
        spins = gen_spinsamples(
            coords,
            n_rotate=self.n_spins,
            method=self.method,
            seed=self.seed
        )
        
        return spins, coords, labels
    
    def permute(self, data: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply spin permutations to a full-length parcel vector.

        Parameters
        ----------
        data : np.ndarray, shape (n_parcels,)
            Data to permute. Its length must equal the number of parcels used
            to build the spins (``self.spins.shape[0]``).
        mask : np.ndarray of bool, optional
            If given, positions where ``mask`` is False are set to NaN in the
            output (values are still rotated from the full vector first).

        Returns
        -------
        permuted : np.ndarray, shape (n_parcels, n_spins)
            Spun copies of ``data``; NaNs propagate through the rotation.
        """
        if isinstance(data, pd.Series):
            data = data.reindex(self.labels).to_numpy(dtype=float)
        else:
            data = np.asarray(data, dtype=float)
        if data.shape[0] != self.spins.shape[0]:
            raise ValueError(
                f"data length ({data.shape[0]}) must match number of parcels "
                f"used to build spins ({self.spins.shape[0]})"
            )
        # self.spins holds indices into the full parcel vector -> index directly.
        permuted = data[self.spins]
        if mask is not None:
            out = np.full((data.shape[0], self.n_spins), np.nan)
            out[mask, :] = permuted[mask, :]
            return out
        return permuted
    
    def correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        metric: str = 'pearson',
        min_samples: int = 3,
    ) -> Tuple[float, float]:
        """
        Test correlation with spin-based null model.
        
        Parameters
        ----------
        x, y : np.ndarray
            Data arrays to correlate
        metric : str, default='pearson'
            Correlation metric ('pearson' or 'spearman')
            
        Returns
        -------
        r : float
            Observed correlation
        p : float
            Spin test p-value
        """
        from scipy.stats import pearsonr, spearmanr

        if isinstance(x, pd.Series):
            x = x.reindex(self.labels).to_numpy(dtype=float)
        else:
            x = np.asarray(x, dtype=float)
        if isinstance(y, pd.Series):
            y = y.reindex(self.labels).to_numpy(dtype=float)
        else:
            y = np.asarray(y, dtype=float)
        if x.ndim != 1 or y.ndim != 1 or len(x) != len(self.labels) or len(y) != len(self.labels):
            raise ValueError(
                "x and y must each contain one value per atlas parcel; use "
                "pandas Series indexed by atlas labels for partial ROI data"
            )

        if metric in ('pearson', 'pearsonr'):
            corr = pearsonr
        elif metric in ('spearman', 'spearmanr'):
            corr = spearmanr
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Observed correlation over positions valid in both maps
        base_mask = ~np.isnan(x) & ~np.isnan(y)
        if base_mask.sum() < min_samples:
            raise ValueError(
                f"At least {min_samples} paired finite parcels are required; "
                f"found {base_mask.sum()}"
            )
        r_obs, _ = corr(x[base_mask], y[base_mask])

        # Null distribution: rotate the full y map, then re-mask per spin because
        # a rotation can move a NaN into a previously-valid position (and vice versa).
        y_permuted = self.permute(y)
        r_null = np.full(self.n_spins, np.nan)

        for i in range(self.n_spins):
            yi = y_permuted[:, i]
            m = ~np.isnan(x) & ~np.isnan(yi)
            if m.sum() >= min_samples:
                r_null[i], _ = corr(x[m], yi[m])

        # Two-tailed spin p-value
        valid_null = r_null[np.isfinite(r_null)]
        if valid_null.size == 0:
            raise ValueError("No valid spin permutations remain after masking")
        p_value = (1 + np.sum(np.abs(valid_null) >= np.abs(r_obs))) / (valid_null.size + 1)
        
        return r_obs, p_value
