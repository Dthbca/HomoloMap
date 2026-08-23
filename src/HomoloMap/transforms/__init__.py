# HomoloMap/transforms/__init__.py
"""
Brain data transformations and preprocessing.

This subpackage provides tools for:
- Atlas management and transformations
- Parcellation operations
- Surface geometry computations
- Spatial smoothing
"""

from .atlas import (
    assign_closest_centroids,
    compute_centroids,
    load_volume_atlas,
    validate_atlas_compatibility,
)
from .parcellation import (
    relabel,
    vertices_to_parcels,
    parc2vertex,
    vol_relabel,
    surf_relabel,
    load_data,
    load_data_list,
)
from .geometry import (
    get_parcel_centroids,
    get_gd_disc,
)
from .smoothing import (
    parc_smooth,
    get_parcel_geodist,
)
from .composition import (
    close_composition,
    multiplicative_zero_replacement,
    clr_transform,
    ilr_transform,
    transform_composition,
)
from .layers import make_layer_subcompositions

__all__ = [
    # Atlas
    'load_volume_atlas',
    'compute_centroids',
    'assign_closest_centroids',
    'validate_atlas_compatibility',
    # Parcellation
    'relabel',
    'vertices_to_parcels',
    'parc2vertex',
    'vol_relabel',
    'surf_relabel',
    'load_data',
    'load_data_list',
    # Geometry
    'get_parcel_centroids',
    'get_gd_disc',
    # Smoothing
    'parc_smooth',
    'get_parcel_geodist',
    # Composition
    'close_composition',
    'multiplicative_zero_replacement',
    'clr_transform',
    'ilr_transform',
    'transform_composition',
    'make_layer_subcompositions',
]
