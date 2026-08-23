
# Default rendering
OFF_SCREEN = False

"""
HomoloMap: Mining Correlations Between Cell Type Maps and Brain Phenotypes
===========================================================================

A comprehensive toolkit for analyzing spatial correlations between brain 
cell type distributions and imaging-derived phenotypes (IDPs).

Main Features
-------------
- Load and transform brain data across multiple atlas spaces
- Compute spatial correlations with proper null models (spin tests)
- Perform feature importance analysis (SHAP, dominance)
- Visualize brain maps and statistical results
- Support for both surface and volumetric data

Quick Start
-----------
>>> from HomoloMap import datasets, stats, plotting
>>> 
>>> # Load cell type ratios and phenotype data
>>> celltype_data = datasets.fetch_ctype_ratio(level='subclass', smooth=True)
>>> phenotype_data = datasets.fetch_enigma(atlas='FGC')
>>> 
>>> # Generate spatial null model
>>> spins = stats.gen_spinsamples(celltype_data, atlas='FGC', n_rotate=1000)
>>> 
>>> # Compute correlations
>>> spinner = stats.SpinTest(atlas='FGC', n_spins=1000)
>>> r, p = spinner.correlation(celltype_data['L2/3_IT'], phenotype_data['asd'])
>>> print(f"r = {r:.3f}, p = {p:.3f}")

Modules
-------
datasets
    Functions for fetching cell type maps, atlases, and phenotype data
transforms  
    Tools for transforming data between atlas spaces and parcellations
stats
    Statistical analysis including correlations, spin tests, and regression
plotting
    Visualization functions for brain maps and statistical results
utils
    Utility functions and configuration management

See Also
--------
Tutorial notebooks in the examples/ directory
Repository: https://github.com/Dthbca/HomoloMap
"""

from . import datasets
from . import transforms
from . import stats
from . import utils
from . import parcellation

from .__version__ import (
    __version__,
    __author__,
    __license__,
   # __url__
)

__all__ = [
    # Modules
    'datasets',
    'transforms',
    'stats',
    'utils',
    'parcellation',
    # Version info
    '__version__',
    '__author__',
    '__license__',
    #'__url__',
]


# Set up a library logger without calling logging.basicConfig(): as a library
# we must not reconfigure the importing application's root logger. Attach a
# NullHandler and let the application decide on handlers/levels.
import logging
logger = logging.getLogger('HomoloMap')
logger.addHandler(logging.NullHandler())
