"""Laminar subcomposition preparation."""

import numpy as np
import pandas as pd
from .composition import transform_composition


def make_layer_subcompositions(layer_data, present_mask=None, transform='clr',
                               zero_method='multiplicative', zero_fraction=0.65,
                               invalid_rows='drop'):
    """Transform each layer over only cell types structurally present there."""
    if invalid_rows not in {'drop', 'raise'}:
        raise ValueError("invalid_rows must be 'drop' or 'raise'")
    output, metadata = {}, {}
    for layer, frame in layer_data.items():
        label = _layer_label(layer, present_mask)
        selected = frame
        if present_mask is not None:
            keep = present_mask.loc[label].reindex(frame.columns, fill_value=False)
            selected = frame.loc[:, keep]
        finite = np.isfinite(selected.to_numpy()).all(axis=1)
        positive = selected.fillna(0).sum(axis=1).to_numpy() > 0
        valid = finite & positive
        invalid_index = selected.index[~valid]
        if len(invalid_index) and invalid_rows == 'raise':
            raise ValueError(
                f"Layer {layer!r} contains {len(invalid_index)} non-finite or "
                "zero-sum compositions")
        selected = selected.loc[valid]
        if selected.empty:
            raise ValueError(f"Layer {layer!r} has no valid compositions")
        if selected.shape[1] < 2 and transform in {'clr', 'ilr'}:
            raise ValueError(f"Layer {layer!r} has fewer than two present cell types")
        transformed, audit = transform_composition(
            selected, method=transform, zero_method=zero_method,
            zero_fraction=zero_fraction)
        output[layer] = transformed
        audit['invalid_rows_policy'] = invalid_rows
        audit['dropped_rows'] = invalid_index.tolist()
        metadata[layer] = audit
    return output, metadata


def _layer_label(layer, mask):
    if mask is None or layer in mask.index:
        return layer
    aliases = {'l1': 'Layer I', 'l2': 'Layer II', 'l3': 'Layer III',
               'l4': 'Layer IV', 'l5': 'Layer V', 'l6': 'Layer VI'}
    return aliases.get(layer, layer)
