# HomoloMap/stats/__init__.py
"""
Statistical analysis tools for brain data.

This subpackage provides:
- Spatial null models (spin tests)
- Regression models with spatial nulls
- Feature importance analysis (SHAP, dominance)
"""

from .nulls import (
    gen_spinsamples,
    spin_data,
    SpinTest,
)
from .analysis import (
    get_reg_r_sq, 
    get_reg_r_pval,
    get_dominance_stats,
    get_shap_stats,
    fit_regression_model,
)
from .layers import (filter_layer_inputs, layer_spin_correlation,
                     layer_match_permutation)



__all__ = [
    # Nulls
    'gen_spinsamples',
    'spin_data',
    'SpinTest',
    # Regression
    'get_reg_r_sq', 
    'get_reg_r_pval', 
    # Explainability
    'get_dominance_stats',
    'get_shap_stats',
    'fit_regression_model',
    'layer_spin_correlation',
    'layer_match_permutation',
    'filter_layer_inputs',
]
