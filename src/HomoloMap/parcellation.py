"""Compatibility imports for parcellation and relabeling functions.

The maintained implementation lives in :mod:`HomoloMap.transforms.parcellation`.
This module preserves the historical import path without duplicating behavior.
"""

from .transforms.parcellation import (
    load_data,
    load_data_list,
    ndimage_adjust,
    parc2vertex,
    relabel,
    surf_relabel,
    vertices_to_parcels,
    vol_relabel,
)

__all__ = [
    'load_data',
    'load_data_list',
    'ndimage_adjust',
    'parc2vertex',
    'relabel',
    'surf_relabel',
    'vertices_to_parcels',
    'vol_relabel',
]
