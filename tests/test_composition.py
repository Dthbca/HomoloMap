import numpy as np
import pandas as pd
import pytest

from HomoloMap.transforms.composition import transform_composition


def test_clr_is_finite_and_centered():
    data = pd.DataFrame([[0.5, 0.3, 0.2], [0.2, 0.4, 0.4]])
    transformed, audit = transform_composition(data, method="clr")
    assert np.isfinite(transformed.to_numpy()).all()
    assert np.allclose(transformed.sum(axis=1), 0.0)
    assert audit["method"] == "clr"


def test_ilr_reduces_dimension_by_one():
    data = pd.DataFrame([[0.5, 0.3, 0.2], [0.2, 0.4, 0.4]])
    transformed, audit = transform_composition(data, method="ilr")
    assert transformed.shape == (2, 2)
    assert audit["method"] == "ilr"


def test_negative_components_are_rejected():
    data = pd.DataFrame([[0.8, 0.3, -0.1]])
    with pytest.raises(ValueError):
        transform_composition(data, method="clr")
