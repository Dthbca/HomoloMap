"""Functional helpers for left-hemisphere volumetric atlases."""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


AtlasSource = Union[str, os.PathLike, nib.spatialimages.SpatialImage]
AtlasData = Dict[str, object]


def compute_centroids(
    coords: np.ndarray,
    voxel_labels: np.ndarray,
    roi_labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return real ROI labels and their centroids in matching row order."""
    coords = np.asarray(coords, dtype=float)
    voxel_labels = np.asarray(voxel_labels, dtype=int)
    if coords.ndim != 2 or coords.shape[0] != voxel_labels.shape[0]:
        raise ValueError("coords and voxel_labels must have matching rows")
    if roi_labels is None:
        roi_labels = np.unique(voxel_labels)
    roi_labels = np.asarray(roi_labels, dtype=int)
    centroids = np.row_stack([
        coords[voxel_labels == label].mean(axis=0) for label in roi_labels
    ])
    return roi_labels, centroids


def load_volume_atlas(
    atlas: AtlasSource,
    atlas_info: Optional[pd.DataFrame] = None,
    hemisphere: Optional[str] = 'left',
) -> AtlasData:
    """Load an atlas and return filtered voxel labels, coordinates and centroids."""
    if isinstance(atlas, (str, os.PathLike)) and os.path.exists(atlas):
        image = nib.load(atlas)
    elif isinstance(atlas, nib.spatialimages.SpatialImage):
        image = atlas
    else:
        raise TypeError("atlas must be an existing path or a nibabel image")

    image = nib.funcs.squeeze_image(image)
    atlas_data = np.asarray(image.dataobj, dtype=int)
    nonzero = atlas_data.nonzero()
    voxel_labels = atlas_data[nonzero]
    coords = nib.affines.apply_affine(image.affine, np.atleast_2d(np.c_[nonzero]))

    if hemisphere == 'left':
        mask = coords[:, 0] < 0
    elif hemisphere == 'right':
        mask = coords[:, 0] >= 0
    elif hemisphere is None:
        mask = np.ones(len(coords), dtype=bool)
    else:
        raise ValueError("hemisphere must be 'left', 'right', or None")
    if atlas_info is not None:
        mask &= np.isin(voxel_labels, atlas_info.index.to_numpy(dtype=int))

    voxel_labels = voxel_labels[mask]
    coords = coords[mask]
    if not len(voxel_labels):
        raise ValueError("atlas has no voxels after hemisphere/annotation filtering")
    roi_labels, centroids = compute_centroids(coords, voxel_labels)
    return {
        'voxel_labels': voxel_labels,
        'coords': coords,
        'roi_labels': roi_labels,
        'centroids': centroids,
        'atlas_info': atlas_info,
    }


def assign_closest_centroids(
    coords: np.ndarray,
    centroids: np.ndarray,
    return_dist: bool = False,
):
    """Assign each coordinate to the nearest centroid using a KD-tree."""
    coords = np.asarray(coords, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    if coords.ndim != 2 or centroids.ndim != 2 or coords.shape[1] != centroids.shape[1]:
        raise ValueError("coords and centroids must be 2D arrays with matching dimensions")
    distances, positions = cKDTree(centroids).query(coords, k=1)
    return (positions, distances) if return_dist else positions


def validate_atlas_compatibility(
    atlas1: AtlasData,
    atlas2: AtlasData,
    tolerance: float = 1.0,
) -> Tuple[bool, str]:
    """Check label overlap and matching-label centroid distances."""
    labels1 = np.asarray(atlas1['roi_labels'])
    labels2 = np.asarray(atlas2['roi_labels'])
    common = np.intersect1d(labels1, labels2)
    if len(labels1) != len(labels2):
        return False, f"Different number of regions: {len(labels1)} vs {len(labels2)}"
    if len(common) < len(labels1) * 0.8:
        return False, f"Insufficient label overlap: {len(common)}/{len(labels1)} labels"

    pos1 = {label: i for i, label in enumerate(labels1)}
    pos2 = {label: i for i, label in enumerate(labels2)}
    centroids1 = np.asarray(atlas1['centroids'])
    centroids2 = np.asarray(atlas2['centroids'])
    max_dist = max(
        np.linalg.norm(centroids1[pos1[label]] - centroids2[pos2[label]])
        for label in common
    )
    if max_dist > tolerance:
        return False, f"Centroids differ by up to {max_dist:.1f}mm (tolerance: {tolerance}mm)"
    return True, "Atlases are compatible"
