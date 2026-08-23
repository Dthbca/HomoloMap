# HomoloMap/transforms/geometry.py
"""
Surface geometry and geodesic distance computations.

This module provides functions for computing centroids, geodesic distances,
and other geometric properties on cortical surface meshes.
"""

import os
import uuid
import subprocess
import warnings
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Union, Optional, Tuple, List
from scipy.spatial import cKDTree

# Labels to ignore during processing
PARCIGNORE = [
    'unknown', 'corpuscallosum', 'Background+FreeSurfer_Defined_Medial_Wall',
    '???', 'Unknown', 'Medial_wall', 'Medial wall', 'medial_wall'
]


def get_parcel_centroids(
    surf: Union[str, Path],
    parc: Optional[nib.GiftiImage] = None,
    method: str = 'surface',
    drop: Optional[List[str]] = None,
    return_labels: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute centroids for parcellated surface regions.
    
    Parameters
    ----------
    surf : str or Path
        Path to surface mesh file (.gii)
    parc : nibabel.GiftiImage, optional
        Loaded parcellation. If None, returns all vertex coordinates
    method : {'average', 'surface', 'geodesic'}, default='surface'
        Method for centroid calculation:
        - 'average': Arithmetic mean of vertex coordinates
        - 'surface': Average coordinate projected to nearest surface vertex
        - 'geodesic': Vertex that minimizes geodesic distance to all others
    drop : list of str, optional
        Label names to exclude from processing. 
        If None, uses default PARCIGNORE list
        
    Returns
    -------
    centroids : np.ndarray
        Array of centroid coordinates, shape (n_parcels, 3)
        
    Raises
    ------
    ValueError
        If method is not one of the valid options
        
    Notes
    -----
    The 'geodesic' method is most accurate but computationally expensive
    as it uses Connectome Workbench to compute geodesic distances on the
    surface. It requires `wb_command` to be available in PATH.
    
    The 'surface' method provides a good balance of accuracy and speed,
    as it projects the average coordinate to the nearest vertex on the
    actual surface.
    
    Examples
    --------
    Compute centroids using surface projection:
    
    >>> from HomoloMap.transforms import get_parcel_centroids
    >>> from HomoloMap.datasets import fetch_fslr, fetch_parc
    >>> 
    >>> surf = fetch_fslr(surf='sphere', return_path=True)
    >>> parc = fetch_parc(key='FGC')
    >>> centroids = get_parcel_centroids(surf, parc, method='surface')
    >>> print(f"Computed {len(centroids)} centroids")
    
    For spin tests (requires sphere surface):
    
    >>> surf_sphere = fetch_fslr(surf='sphere', return_path=True)
    >>> centroids = get_parcel_centroids(surf_sphere, parc, method='surface')
    
    See Also
    --------
    get_gd_parc_centroids : Compute geodesic centroid for single parcel
    get_gd_disc : Get geodesic distance disc around a vertex
    """
    valid_methods = ['average', 'surface', 'geodesic']
    if method not in valid_methods:
        raise ValueError(
            f"Method '{method}' invalid. Must be one of {valid_methods}"
        )
    
    if drop is None:
        drop = PARCIGNORE
    
    centroids = []
    centroid_labels = []
    
    # Load surface mesh
    vertices, faces = nib.load(surf).agg_data()
    
    if parc is not None:
        labels = parc.agg_data()
        labeltable = parc.labeltable.get_labels_as_dict()
        surface_tree = cKDTree(vertices) if method == 'surface' else None
        
        for lab in np.unique(labels):
            # Skip ignored labels
            if (labeltable.get(lab) in drop) | (lab <= 0):
                continue
            
            mask = labels == lab
            
            if method in ('average', 'surface'):
                # Compute average coordinate
                roi = np.atleast_2d(vertices[mask].mean(axis=0))
                
                if method == 'surface':
                    # Project to nearest surface vertex
                    _, idx = surface_tree.query(roi[0])
                    roi = vertices[idx]
                    
            elif method == 'geodesic':
                # Use geodesic distance to find true centroid
                _, roi = get_gd_parc_centroids(surf, parc, lab)
            
            centroids.append(roi)
            centroid_labels.append(int(lab))
    else:
        # No parcellation - return all vertices
        centroids.append(vertices)
    
    if not centroids:
        raise ValueError(
            "Parcellation contains no positive labels after applying exclusions"
        )

    centroid_array = np.row_stack(centroids)
    if return_labels:
        return centroid_array, np.asarray(centroid_labels, dtype=int)
    return centroid_array


def get_gd_parc_centroids(
    mesh: Union[str, Path],
    parc: nib.GiftiImage,
    label: int
) -> Tuple[int, np.ndarray]:
    """
    Compute geodesic centroid for a single parcel.
    
    The geodesic centroid is the vertex that minimizes the sum of
    geodesic distances to all other vertices in the parcel.
    
    Parameters
    ----------
    mesh : str or Path
        Path to surface mesh file
    parc : nibabel.GiftiImage
        Loaded parcellation
    label : int
        Parcel label to compute centroid for
        
    Returns
    -------
    centroid_idx : int
        Index of the centroid vertex
    centroid_coord : np.ndarray
        3D coordinates of the centroid vertex
        
    Notes
    -----
    This function uses Connectome Workbench's `wb_command` to compute
    geodesic distances. It requires `wb_command` to be available in PATH.
    
    The function creates temporary files during computation which are
    automatically cleaned up.
    
    Examples
    --------
    >>> centroid_idx, centroid_coord = get_gd_parc_centroids(
    ...     'path/to/mesh.gii',
    ...     parc,
    ...     label=42
    ... )
    >>> print(f"Centroid at vertex {centroid_idx}: {centroid_coord}")
    
    See Also
    --------
    get_parcel_centroids : Compute centroids for all parcels
    get_gd_disc : Get geodesic distance disc
    """
    parc_data = parc.agg_data()
    label_vertices = np.where(parc_data == label)[0]
    
    # Create metric file for the ROI
    metric_data = np.zeros(shape=parc_data.shape)
    metric_data[label_vertices] = 1
    
    data_array = nib.gifti.GiftiDataArray(
        data=metric_data,
        datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32'],
        intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE']
    )
    metric_img = nib.gifti.GiftiImage()
    metric_img.add_gifti_data_array(data_array)
    
    # Save to temporary file
    output_file = f'/tmp/roi_metric_{uuid.uuid4().hex[:8]}.func.gii'
    nib.save(metric_img, output_file)
    
    # Compute geodesic distances
    tmp_fname = f'/tmp/surface_disc_{uuid.uuid4().hex[:8]}.dconn.nii'
    
    cmd = [
        'wb_command', '-surface-geodesic-distance-all-to-all',
        str(mesh), tmp_fname, '-roi', output_file
    ]
    
    try:
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"wb_command failed: {e.stderr}\n"
                "Make sure Connectome Workbench is installed and in PATH"
            ) from e

        # Load distances and find centroid
        disc = np.asarray(nib.load(tmp_fname).dataobj)

        # Centroid is vertex with minimum sum of distances
        centroid_idx = label_vertices[disc.sum(axis=0).argmin()]

        # Get coordinates
        vertices, _ = nib.load(mesh).agg_data()
        centroid_coord = vertices[centroid_idx]
    finally:
        for filename in (output_file, tmp_fname):
            if os.path.exists(filename):
                os.remove(filename)
    
    return centroid_idx, centroid_coord


def get_gd_disc(
    mesh: Union[str, Path],
    vertex_number: int,
    radius: Optional[float] = None,
    return_dist: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Get vertices within geodesic distance from a seed vertex.
    
    Parameters
    ----------
    mesh : str or Path
        Path to surface mesh file
    vertex_number : int
        Seed vertex index
    radius : float, optional
        Maximum geodesic distance in mm. If None, returns all distances
    return_dist : bool, default=False
        Whether to return the actual distances in addition to the mask
        
    Returns
    -------
    gd_mask : np.ndarray
        Boolean mask of vertices within radius (or all vertices if radius=None)
    gd : np.ndarray, optional
        Actual geodesic distances (if return_dist=True)
        
    Notes
    -----
    Uses Connectome Workbench's `wb_command` for geodesic distance computation.
    Requires `wb_command` in PATH.
    
    Examples
    --------
    Get all vertices within 10mm of vertex 1000:
    
    >>> from HomoloMap.transforms import get_gd_disc
    >>> mask = get_gd_disc('mesh.gii', vertex_number=1000, radius=10)
    >>> neighbor_vertices = np.where(mask)[0]
    
    Get distances for smoothing kernel:
    
    >>> distances, mask = get_gd_disc(
    ...     'mesh.gii',
    ...     vertex_number=1000,
    ...     radius=15,
    ...     return_dist=True
    ... )
    >>> weights = np.exp(-(distances**2) / (2 * 5**2))  # Gaussian kernel
    
    See Also
    --------
    get_parcel_geodist : Compute full geodesic distance matrix between parcels
    """
    # Generate temporary output file
    tmp_fname = f'/tmp/gd_{vertex_number}_{uuid.uuid4().hex[:8]}.shape.gii'
    
    # Build command
    cmd = [
        'wb_command', '-surface-geodesic-distance',
        str(mesh), str(vertex_number), tmp_fname
    ]
    
    if radius is not None:
        cmd.extend(['-limit', str(radius)])
    
    # Execute command
    try:
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"wb_command failed: {e.stderr}\n"
                "Make sure Connectome Workbench is installed and in PATH"
            ) from e

        # Load result
        gd = nib.load(tmp_fname).agg_data()
        gd_mask = gd >= 0  # Positive values are valid distances
    finally:
        if os.path.exists(tmp_fname):
            os.remove(tmp_fname)
    
    if return_dist:
        return gd, gd_mask
    else:
        return gd_mask


def get_euclidean_disc(
    vertices: np.ndarray,
    vertex_idx: int,
    radius: float
) -> np.ndarray:
    """
    Compute Euclidean distance disc (fast alternative to geodesic).
    
    Parameters
    ----------
    vertices : np.ndarray
        Vertex coordinates, shape (n_vertices, 3)
    vertex_idx : int
        Seed vertex index
    radius : float
        Maximum Euclidean distance
        
    Returns
    -------
    mask : np.ndarray
        Boolean mask of vertices within radius
        
    Notes
    -----
    This is much faster than geodesic distance but less accurate for
    cortical surfaces where geodesic distance better represents the
    actual distance along the folded surface.
    
    Examples
    --------
    >>> vertices, _ = nib.load('mesh.gii').agg_data()
    >>> mask = compute_euclidean_disc(vertices, vertex_idx=1000, radius=10)
    """
    seed_coord = vertices[vertex_idx]
    distances = np.linalg.norm(vertices - seed_coord, axis=1)
    return distances < radius


def project_to_surface(
    coords: np.ndarray,
    surface_vertices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D coordinates to nearest vertices on a surface.
    
    Parameters
    ----------
    coords : np.ndarray
        Coordinates to project, shape (n_points, 3)
    surface_vertices : np.ndarray
        Surface vertex coordinates, shape (n_vertices, 3)
        
    Returns
    -------
    projected_coords : np.ndarray
        Coordinates of nearest surface vertices
    vertex_indices : np.ndarray
        Indices of the nearest vertices
        
    Examples
    --------
    >>> vertices, _ = nib.load('mesh.gii').agg_data()
    >>> volume_coords = np.array([[10, 20, 30], [15, 25, 35]])
    >>> proj_coords, indices = project_to_surface(volume_coords, vertices)
    """
    from scipy.spatial import cKDTree
    
    tree = cKDTree(surface_vertices)
    distances, indices = tree.query(coords)
    projected_coords = surface_vertices[indices]
    
    return projected_coords, indices
