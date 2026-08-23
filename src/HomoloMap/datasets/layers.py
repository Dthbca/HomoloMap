"""Load laminar cell-type maps and BigBrain cortical thickness."""

from pathlib import Path
import os
import numpy as np
import pandas as pd

from .atlas import ctype_ratio_agg

LAYER_KEYS = ('l1', 'l2', 'l3', 'l4', 'l5', 'l6')
LAYER_LABELS = ('Layer I', 'Layer II', 'Layer III',
                'Layer IV', 'Layer V', 'Layer VI')


def _data_root(data_dir=None):
    if data_dir is not None:
        return Path(data_dir)
    configured = os.environ.get('BIGBRAIN_DATASET_ROOT')
    if configured:
        return Path(configured)
    return Path(__file__).parent / 'features'


def load_layer_counts(data_dir=None, source_atlas='D99', mapping_column='subclass',
                      unmapped='drop', map_df=None, return_mapping=False):
    """Load raw macaque counts as one DataFrame per cortical layer."""
    root = _data_root(data_dir)
    payload_path = root / 'Spatial' / 'raw_counts_d99.npy'
    if not payload_path.exists():
        raise FileNotFoundError(payload_path)
    payload = np.load(payload_path, allow_pickle=True).item()
    counts = np.asarray(payload['counts'])
    ctypes = list(payload['ctypes'])
    regions = list(payload['regions'])
    if counts.shape != (len(regions), len(LAYER_KEYS), len(ctypes)):
        raise ValueError("raw_counts_d99.npy has inconsistent dimensions")
    if map_df is None:
        local_map = root / 'Spatial' / 'cluster_mapping_dict.csv'
        if local_map.exists():
            map_df = pd.read_csv(local_map, index_col=0)
    output, audits = {}, []
    for index, layer in enumerate(LAYER_KEYS):
        frame = pd.DataFrame(counts[:, index, :], index=regions, columns=ctypes)
        frame, audit = ctype_ratio_agg(
            frame, map_df=map_df, key=mapping_column, unmapped=unmapped,
            return_mapping=True)
        frame.attrs = {}
        output[layer] = frame
        audits.append(audit)
    audit = dict(audits[0])
    audit.update({'source_atlas': source_atlas, 'layers': list(LAYER_KEYS)})
    return (output, audit) if return_mapping else output


def relabel_layer_counts(layer_counts, source_atlas='D99', target_atlas='BN',
                         method='sum', cross_species=True,
                         unknown_labels='drop', return_audit=False):
    """Relabel every layer count table into a target atlas."""
    from HomoloMap.transforms.parcellation import vol_relabel
    if unknown_labels not in {'raise', 'drop'}:
        raise ValueError("unknown_labels must be 'raise' or 'drop'")
    from .atlas import fetch_annot
    from HomoloMap.transforms.atlas import load_volume_atlas
    source_path, source_info = fetch_annot(atlas=source_atlas, annot=True)
    valid_labels = set(load_volume_atlas(
        source_path, source_info, hemisphere='left')['roi_labels'])
    result, dropped = {}, {}
    for layer, frame in layer_counts.items():
        unknown = sorted(set(frame.index) - valid_labels)
        if unknown and unknown_labels == 'raise':
            raise ValueError(f"Layer {layer!r} has unknown {source_atlas} labels: {unknown}")
        dropped[layer] = unknown
        frame = frame.drop(index=unknown, errors='ignore')
        if target_atlas == source_atlas:
            result[layer] = frame.copy()
        else:
            result[layer] = vol_relabel(
                source_atlas, target_atlas, frame, method=method,
                cross_species=cross_species)
    audit = {'source_atlas': source_atlas, 'target_atlas': target_atlas,
             'method': method, 'unknown_labels_policy': unknown_labels,
             'dropped_labels': dropped}
    return (result, audit) if return_audit else result


def normalize_layer_composition(layer_counts, mode='within_layer',
                                zero_policy='nan'):
    """Normalize layer counts using an explicitly named denominator.

    ``within_region`` is retained for backwards compatibility and closes each
    cell type independently across layers.  It is *not* a joint composition.
    ``joint_all_layers_all_celltypes`` closes every layer-by-cell-type part
    together within each ROI.
    """
    modes = {'within_layer', 'within_region',
             'joint_all_layers_all_celltypes'}
    if mode not in modes:
        raise ValueError(f"mode must be one of {sorted(modes)}")
    if zero_policy not in {'nan', 'zero', 'raise'}:
        raise ValueError("zero_policy must be 'nan', 'zero', or 'raise'")
    layers = list(layer_counts)
    index = layer_counts[layers[0]].index
    columns = layer_counts[layers[0]].columns
    aligned = {k: v.reindex(index=index, columns=columns) for k, v in layer_counts.items()}
    if mode == 'within_region':
        denominators = sum(aligned.values())
    elif mode == 'joint_all_layers_all_celltypes':
        denominators = sum(frame.sum(axis=1) for frame in aligned.values())
    output = {}
    for layer, frame in aligned.items():
        denominator = frame.sum(axis=1) if mode == 'within_layer' else denominators
        if mode in {'within_layer', 'joint_all_layers_all_celltypes'}:
            transformed = frame.div(denominator.replace(0, np.nan), axis=0)
        else:
            transformed = frame.div(denominator.replace(0, np.nan))
        if zero_policy == 'raise' and transformed.isna().any(axis=None):
            raise ValueError("Zero denominator encountered during layer normalization")
        output[layer] = transformed.fillna(0) if zero_policy == 'zero' else transformed
    return output


def mask_and_close_joint_composition(layer_counts, present_mask,
                                     zero_policy='zero', atol=1e-10):
    """Apply a structural layer mask, then close all retained parts per ROI.

    This is the intended post-relabel feature definition for joint laminar
    composition: the denominator spans every retained layer and cell type.
    """
    masked = {layer: frame.copy().astype(float)
              for layer, frame in layer_counts.items()}
    aliases = dict(zip(LAYER_KEYS, LAYER_LABELS))
    for layer, frame in masked.items():
        row = layer if layer in present_mask.index else aliases[layer]
        keep = present_mask.loc[row].reindex(frame.columns, fill_value=False)
        frame.loc[:, ~keep] = 0.0
    closed = normalize_layer_composition(
        masked, mode='joint_all_layers_all_celltypes', zero_policy=zero_policy)
    row_sums = sum(frame.sum(axis=1) for frame in closed.values())
    positive = row_sums > 0
    errors = (row_sums.loc[positive] - 1.0).abs()
    max_error = float(errors.max()) if len(errors) else 0.0
    if max_error > atol:
        raise AssertionError(
            f"Joint composition closure error {max_error:.3g} exceeds {atol}")
    audit = {
        'mode': 'joint_all_layers_all_celltypes',
        'denominator': 'all mask-retained layer_by_celltype parts within ROI',
        'n_rows': int(len(row_sums)),
        'n_positive_rows': int(positive.sum()),
        'n_zero_rows': int((~positive).sum()),
        'min_positive_row_sum': float(row_sums.loc[positive].min()) if positive.any() else None,
        'max_positive_row_sum': float(row_sums.loc[positive].max()) if positive.any() else None,
        'max_abs_closure_error': max_error,
    }
    return closed, audit


def fetch_bigbrain_layer_thickness(atlas='FGC', data_dir=None, relative=False,
                                   relabel_method='mean', regions=None):
    """Load absolute or relative BigBrain thickness for the six cortical layers."""
    root = _data_root(data_dir)
    path = root / 'BigBrain' / 'layer_thickness_parced.csv'
    if not path.exists():
        raise FileNotFoundError(path)
    thickness = pd.read_csv(path, index_col=0)
    if atlas != 'FGC':
        from HomoloMap.transforms.parcellation import vol_relabel
        thickness = vol_relabel('FGC', atlas, thickness, method=relabel_method)
    if regions is not None:
        thickness = thickness.reindex(regions)
    if relative:
        thickness = thickness.div(thickness.sum(axis=1).replace(0, np.nan), axis=0)
    thickness.columns = list(LAYER_LABELS[:thickness.shape[1]])
    return thickness


def fetch_laminar_mask(kind='external', cell_types=None, layers=LAYER_KEYS,
                       threshold=0.05, counts=None, data_dir=None, strict=True):
    """Return a present-mask (True means a cell type is present in a layer)."""
    labels = list(LAYER_LABELS[:len(layers)])
    if kind == 'external':
        path = _data_root(data_dir) / 'Spatial' / 'mask_by_nc2025.csv'
        if not path.exists():
            raise FileNotFoundError(path)
        mask = pd.read_csv(path, index_col=0).iloc[:, :len(layers)].T
        mask.index = labels
    elif kind == 'enrichment':
        if counts is None:
            raise ValueError("counts is required for enrichment mask")
        totals = pd.DataFrame({k: v.sum(axis=0) for k, v in counts.items()}).T
        shares = totals.div(totals.sum(axis=0).replace(0, np.nan), axis=1)
        mask = shares > threshold
        mask.index = labels
    else:
        raise ValueError("kind must be 'external' or 'enrichment'")
    if cell_types is not None:
        missing = pd.Index(cell_types).difference(mask.columns)
        if strict and len(missing):
            raise KeyError(f"Mask lacks {len(missing)} cell types: {missing[:10].tolist()}")
        mask = mask.reindex(columns=cell_types, fill_value=False)
    audit = {'kind': kind, 'threshold': threshold if kind == 'enrichment' else None,
             'semantics': 'True=present'}
    return mask.astype(bool), audit
