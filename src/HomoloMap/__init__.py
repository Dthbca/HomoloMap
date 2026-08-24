
# Default rendering
OFF_SCREEN = False

"""
HomoloMap: Linking Human Brain Organization to Cellular Architecture
=====================================================================

A cross-species toolkit that combines cortical areal alignment with
transcriptomic cell-type homology to construct human cortical cell-type maps
and relate them to imaging-derived phenotypes (IDPs).

Main Features
-------------
- Construct and audit human-aligned subclass and cluster maps
- Load and transform brain data across macaque and human atlas spaces
- Compute spatial correlations with proper null models (spin tests)
- Perform feature importance analysis (SHAP, dominance)
- Visualize brain maps and statistical results
- Support for both surface and volumetric data

Quick Start
-----------
>>> import pandas as pd
>>> from HomoloMap.utils import run_analysis
>>> 
>>> # The external input is a region-by-IDP table indexed by atlas labels.
>>> idps = pd.read_csv('brain_idps_bn.csv', index_col=0)
>>> idps.index = idps.index.astype(int)
>>> result = run_analysis(
...     data=idps, atlas='BN', feature_type='ratio',
...     ctype_level='subclass', n_spins=1000,
...     cumulative=True, explanations='shap', random_state=42,
... )

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
Tutorial notebooks in the tutorials/ directory
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
