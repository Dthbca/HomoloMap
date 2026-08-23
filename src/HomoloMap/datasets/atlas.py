import pandas as pd
import numpy as np
from pathlib import Path
import nibabel as nib
import os

ATLAS = dict(
    DK='Desikan', Yeo17='Yeo_JNeurophysiol11_17Networks', Yeo7='Yeo_JNeurophysiol11_7Networks',
    FGC='Zhang_fine-grained'
)

def ctype_ratio_agg(df, map_df=None, key='celltype', unmapped='drop',
                    renormalize=False, return_mapping=False):
    """Map macaque cell types to one explicit human homology annotation."""
    root_path = Path(__file__).parent
    if map_df is None:
        map_df = pd.read_csv(os.path.join(root_path,'features/SpatialTranscriptomics/ctype_map.csv'),index_col=0)
    if key not in map_df.columns:
        raise KeyError(f"Mapping column {key!r} is not present in ctype_map.csv")
    if unmapped not in {'raise', 'drop', 'keep'}:
        raise ValueError("unmapped must be 'raise', 'drop', or 'keep'")
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("df must be a non-empty DataFrame")
    if not df.columns.is_unique:
        raise ValueError("Macaque cell-type columns must be unique")

    agg_df = df.copy()
    mapping = map_df[key].reindex(agg_df.columns)
    unresolved_mask = mapping.isna() | mapping.astype('string').str.strip().eq('')
    unresolved = mapping.index[unresolved_mask].tolist()
    if unresolved and unmapped == 'raise':
        raise KeyError(
            f"{len(unresolved)} macaque cell types lack a human transcriptomic-"
            f"homology mapping in {key!r}: {unresolved[:10]}"
        )
    total_mass = agg_df.sum(axis=1, min_count=1)
    resolved_columns = mapping.index[~unresolved_mask]
    resolved_mass = agg_df.loc[:, resolved_columns].sum(axis=1, min_count=1)
    if unresolved and unmapped == 'drop':
        agg_df = agg_df.drop(columns=unresolved)
        mapping = mapping.drop(index=unresolved)
    elif unresolved:
        mapping.loc[unresolved] = [f"macaque_unresolved::{x}" for x in unresolved]
    agg_df.columns = mapping.astype(str).to_numpy()
    agg_df = agg_df.T.groupby(level=0).sum().T
    mapped_mass = agg_df.sum(axis=1, min_count=1)
    coverage = resolved_mass.div(total_mass.replace(0, np.nan))
    if renormalize:
        agg_df = agg_df.div(mapped_mass.replace(0, np.nan), axis=0)
    audit = {
        'source_species': 'macaque', 'target_species': 'human',
        'mapping_basis': 'transcriptomic_homology', 'mapping_column': key,
        'unmapped_policy': unmapped, 'renormalized': bool(renormalize),
        'n_original_types': int(df.shape[1]),
        'n_mapped_types': int(df.shape[1] - len(unresolved)),
        'n_unresolved_types': int(len(unresolved)),
        'unresolved_types': unresolved, 'mapping_coverage': coverage,
        'mapping_table': pd.DataFrame({
            'macaque_celltype': df.columns,
            'human_homologue': map_df[key].reindex(df.columns).to_numpy(),
            'resolved': ~unresolved_mask.to_numpy(),
        }),
    }
    agg_df.attrs['celltype_mapping'] = audit
    return (agg_df, audit) if return_mapping else agg_df


def fetch_ctype_ratio(level='subclass', smooth=True, mapping_column=None,
                      unmapped='drop', renormalize=False, return_mapping=False):
    root_path = Path(__file__).parent
    if smooth:
        ctype_ratio = pd.read_csv(os.path.join(root_path,'features/SpatialTranscriptomics/ctype_ratio_plot_FGC_smoothed.csv'),index_col=0)
    else:
        ctype_ratio = pd.read_csv(os.path.join(root_path,'features/SpatialTranscriptomics/ctype_ratio_plot_FGC.csv'),index_col=0)
    
    if level in ['subclass','cluster']:
        ctype_ratio = ctype_ratio_agg(
            ctype_ratio, key=mapping_column or level, unmapped=unmapped,
            renormalize=renormalize, return_mapping=return_mapping)

    return ctype_ratio

def fetch_ctype_density(level='subclass', smooth=True, mapping_column=None,
                        unmapped='drop', renormalize=False, return_mapping=False):
    root_path = Path(__file__).parent
    if smooth:
        ctype_density = pd.read_csv(os.path.join(root_path,'features/SpatialTranscriptomics/ctype_density_plot_FGC_smoothed.csv'),index_col=0)
    else:
        ctype_density = pd.read_csv(os.path.join(root_path,'features/SpatialTranscriptomics/ctype_density_plot_FGC.csv'),index_col=0)
    
    if level in ['subclass','cluster']:
        ctype_density = ctype_ratio_agg(
            ctype_density, key=mapping_column or level, unmapped=unmapped,
            renormalize=renormalize, return_mapping=return_mapping)

    return ctype_density


def fetch_layer_ratio(level='subclass', donor='M1', mask=True, mapping_column=None,
                      unmapped='drop', return_mapping=False):
    root_path = Path(__file__).parent
    layers = ['l1','l2','l3','l4','l5','l6']

    if level not in ['subclass','cluster']:
        raise ValueError(
            f"level must be 'subclass' or 'cluster', got {level!r}"
        )

    layer_data_list, audits = [], []
    for layer in layers:
        path = os.path.join(root_path,'features/SpatialTranscriptomics/layer_data',f'ctype_region_counts_FGC_{donor}_{layer}.csv')
        layer_data = pd.read_csv(path,index_col=0)
        layer_data, audit = ctype_ratio_agg(
            layer_data, key=mapping_column or level, unmapped=unmapped,
            return_mapping=True)
        audits.append(audit)
        # pandas.concat compares attrs for equality; audit contains Series,
        # whose elementwise equality has no scalar truth value. The audit is
        # retained separately and restored on the concatenated result below.
        layer_data.attrs = {}
        layer_data_list.append(layer_data)
    layers_data = pd.concat(layer_data_list,axis=1,keys=layers)

    if mask:
        layer_mask =pd.read_csv(os.path.join(root_path,'features/SpatialTranscriptomics/mask_by_nc2025.csv'),index_col=0)
        for ctype in layer_mask.index:
            for layer in layers:
                if not layer_mask.loc[ctype,layer]:
                    layers_data.loc[:,(layer,ctype)] = 0
    
    layer_ctype_ratio = layers_data.div(layers_data.sum(axis=1),axis=0)
    audit = dict(audits[0])
    audit['layers'] = layers
    layer_ctype_ratio.attrs['celltype_mapping'] = audit
    return (layer_ctype_ratio, audit) if return_mapping else layer_ctype_ratio


def fetch_enigma(atlas='DK'):
    root_path = Path(__file__).parent
    if atlas=='DK':
        enigma_disease = pd.read_csv(os.path.join(root_path,'features/enigma_ct.csv'),index_col=-1)
        enigma_disease = enigma_disease[enigma_disease.hemi=='L'].iloc[:,1:-2]
    elif atlas == 'FGC':
        enigma_disease = pd.read_csv(os.path.join(root_path,'features/enigma_fgc_smoothed.csv'),index_col=0)
    else:
        raise ValueError(f"Unsupported atlas {atlas!r}; expected 'DK' or 'FGC'")

    return enigma_disease


def fetch_fslr(density='32k', hemi='L', surf='inflated', base_dir=None,return_path = False):
    if base_dir is None:
        root_path = Path(__file__).parent
        base_dir = root_path / 'surfaces' / 'fslr_32k'
    else:
        base_dir = Path(base_dir)

    fn = f'fs_LR.{density}.{hemi}.{surf}.surf.gii'
    surface_path = base_dir / fn

    if not surface_path.exists():
        raise FileNotFoundError(f"file not exist: {surface_path}")

    if return_path:
        return str(surface_path)
    else:
        gii_mesh = nib.load(str(surface_path))
        return gii_mesh


def fetch_Yerks(hemi='L', surf='inflated', base_dir=None,return_path = False):
    if base_dir is None:
        root_path = Path(__file__).parent
        base_dir = root_path / 'surfaces' / 'MacaqueYerks19'
    else:
        base_dir = Path(base_dir)

    fn = f'MacaqueYerkes19.{hemi}.{surf}.32k_fs_LR.surf.gii'
    surface_path = base_dir / fn

    if not surface_path.exists():
        raise FileNotFoundError(f"file not exist: {surface_path}")

    if return_path:
        return str(surface_path)
    else:
        gii_mesh = nib.load(str(surface_path))
        return gii_mesh


def fetch_parc(data_dir=None, hemi='L', key='FGC'):

    #如果key使到图谱的路径，则直接返回路径
    if os.path.exists(key):
        parc = nib.load(key)
        return parc

    if data_dir is None:
        root_path = Path(__file__).parent
        data_dir = root_path / 'surfaces' / 'parcellations'
    
    # Retrieve the atlas name from ATLAS; if the key is not found, use the key itself as the fallback.
    atlas = ATLAS.get(key, key)
    fn = f'{atlas}.fs_LR_32k.{hemi}.label.gii'
    parc_path = data_dir / fn

    if not parc_path.exists():
        raise FileNotFoundError(f"file not exist: {parc_path}")
    
    parc = nib.load(str(parc_path))
    return parc


def fetch_annot(data_dir=None, atlas='FGC',res='1mm', annot=True):
    # root_path is used for the annotation dir below regardless of whether the
    # caller supplied data_dir, so define it unconditionally.
    root_path = Path(__file__).parent
    if data_dir is None:
        data_dir = root_path / 'volumes'
    else:
        data_dir = Path(data_dir)

    if atlas in ['D99','MacBN']:
        fn = f'{atlas}_NMT2asym.nii.gz'
    elif atlas in ['MacBN_human','economo','FGC','BN']:
        fn = f'{atlas}_MNI152_{res}.nii.gz'
    else:
        raise ValueError(f"Unsupported atlas: {atlas}")

    parc_path = data_dir / fn
    parc = str(parc_path)

    if annot:
        annot_dir = root_path / 'annotation'
        if atlas == 'MacBN_human':
            atlas = 'MacBN'
        annot_path = annot_dir / f'{atlas}_annot.csv'
        annot = pd.read_csv(str(annot_path),index_col=0)
        return parc, annot
    else:
        return parc
