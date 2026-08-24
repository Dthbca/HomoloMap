import pandas as pd
import numpy as np
from pathlib import Path
import nibabel as nib
import os

from .resources import fetch_resource, get_data_dir

ATLAS = dict(
    DK='Desikan', Yeo17='Yeo_JNeurophysiol11_17Networks', Yeo7='Yeo_JNeurophysiol11_7Networks',
    FGC='Zhang_fine-grained'
)

def ctype_ratio_agg(df, map_df=None, key='celltype', unmapped='drop',
                    renormalize=False, return_mapping=False):
    """Map macaque cell types to one explicit human homology annotation."""
    root_path = Path(__file__).parent
    if map_df is None:
        map_df = pd.read_csv(
            root_path / 'features' / 'SpatialTranscriptomics' /
            'cluster_mapping_dict.csv'
        ).set_index('plot')
    if key not in map_df.columns:
        raise KeyError(
            f"Mapping column {key!r} is not present in "
            "cluster_mapping_dict.csv"
        )
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


def _celltype_resource(filename):
    return (Path(__file__).parent / 'features' / 'SpatialTranscriptomics' /
            filename)


def fetch_ctype_ratio(level='subclass', smooth=None, mapping_column=None,
                      unmapped='drop', renormalize=True, return_mapping=False,
                      atlas='BN'):
    """Load released mapped cell-type ratios in D99 or BN space.

    BN maps are the public, human-aligned products (23 subclasses or 71
    clusters). D99 maps retain the original macaque regional sampling and are
    aggregated with ``cluster_mapping_dict.csv``. ``smooth`` is retained only
    for backwards compatibility; released maps are not smoothed here.
    """
    level = mapping_column or level
    if level not in {'subclass', 'cluster'}:
        raise ValueError("level must be 'subclass' or 'cluster'")
    if atlas == 'BN':
        n_types = {'subclass': 23, 'cluster': 71}[level]
        data = pd.read_csv(
            _celltype_resource(f'ctype_ratio_BN_{n_types}_{level}.csv'),
            index_col=0,
        )
        data.index = data.index.astype(int)
        audit = {
            'source_space': 'macaque_D99_spatial_transcriptomics',
            'target_space': 'human_BN_left_hemisphere',
            'mapping_basis': 'joint_embedding_transcriptomic_homology',
            'mapping_column': level,
            'n_mapped_types': n_types,
            'renormalized': True,
            'resource': f'ctype_ratio_BN_{n_types}_{level}.csv',
        }
        data.attrs['celltype_mapping'] = audit
        return (data, audit) if return_mapping else data
    if atlas != 'D99':
        raise ValueError("Released ratio maps support atlas='D99' or 'BN'")
    data = pd.read_csv(_celltype_resource('ctype_ratio_plot_D99.csv'), index_col=0)
    data.index = data.index.astype(int)
    return ctype_ratio_agg(
        data, key=level, unmapped=unmapped, renormalize=renormalize,
        return_mapping=return_mapping,
    )

def fetch_ctype_density(level='subclass', smooth=None, mapping_column=None,
                        unmapped='drop', renormalize=False,
                        return_mapping=False, atlas='D99'):
    """Load D99 cell density and aggregate it with the canonical map table."""
    if atlas != 'D99':
        raise ValueError(
            "Density is released in D99 space; load atlas='D99' and use "
            "HomoloMap.transforms.vol_relabel for another atlas."
        )
    data = pd.read_csv(_celltype_resource('ctype_density_plot_D99.csv'), index_col=0)
    data.index = data.index.astype(int)
    return ctype_ratio_agg(
        data, key=mapping_column or level, unmapped=unmapped,
        renormalize=renormalize, return_mapping=return_mapping,
    )


def fetch_layer_ratio(*args, **kwargs):
    """Compatibility wrapper for :func:`HomoloMap.datasets.layers.fetch_layer_ratio`."""
    from .layers import fetch_layer_ratio as _fetch_layer_ratio
    return _fetch_layer_ratio(*args, **kwargs)


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


def fetch_fslr(density='32k', hemi='L', surf='inflated', base_dir=None,
               return_path=False, download=True, verbose=1):
    """Fetch an fsLR surface through neuromaps and cache it outside the package."""
    hemi = hemi.upper()
    if hemi not in {'L', 'R'}:
        raise ValueError("hemi must be 'L' or 'R'")
    if base_dir is None:
        packaged = Path(__file__).parent / 'surfaces' / 'fslr_32k'
        legacy = packaged / f'fs_LR.{density}.{hemi}.{surf}.surf.gii'
        surface_path = legacy if legacy.exists() else None
    else:
        base_dir = Path(base_dir)
        legacy = base_dir / f'fs_LR.{density}.{hemi}.{surf}.surf.gii'
        if legacy.exists():
            surface_path = legacy
        elif not download:
            raise FileNotFoundError(f"Surface is not cached: {legacy}")
        else:
            surface_path = None
    if surface_path is None:
        if not download:
            location = (
                Path(base_dir) if base_dir is not None
                else Path(__file__).parent / 'surfaces' / 'fslr_32k'
            )
            raise FileNotFoundError(
                f"Surface is not available offline in {location}. "
                "Set download=True once to populate the neuromaps cache, or "
                "pass base_dir containing the requested surface."
            )
        from neuromaps.datasets import fetch_fslr as _fetch_fslr
        cache = get_data_dir(base_dir) / 'neuromaps'
        atlas = _fetch_fslr(
            density=density, data_dir=str(cache),
            verbose=verbose,
        )
        key = {'very_inflated': 'veryinflated'}.get(surf, surf)
        if key not in atlas:
            raise ValueError(
                f"Surface {surf!r} is unavailable for fsLR {density}; "
                f"available: {sorted(atlas)}"
            )
        pair = atlas[key]
        surface_path = Path(getattr(pair, hemi, pair[0 if hemi == 'L' else 1]))
    return str(surface_path) if return_path else nib.load(str(surface_path))


def fetch_Yerks(hemi='L', surf='inflated', base_dir=None, return_path=False,
                url=None, sha256=None, download=True):
    """Fetch one Yerkes19 surface from a user-declared official URL."""
    data_root = get_data_dir(base_dir)
    base_dir = data_root / 'Yerkes19'

    fn = f'MacaqueYerkes19.{hemi}.{surf}.32k_fs_LR.surf.gii'
    surface_path = base_dir / fn

    if not surface_path.exists():
        if url is None or sha256 is None:
            raise FileNotFoundError(
                f"Surface is not cached: {surface_path}. Supply the provider's "
                "direct url and sha256, or set HOMOLOMAP_DATA to an existing cache."
            )
        surface_path = fetch_resource(
            'Yerkes19', url=url, sha256=sha256, filename=fn,
            data_dir=data_root, download=download,
        )

    if return_path:
        return str(surface_path)
    else:
        gii_mesh = nib.load(str(surface_path))
        return gii_mesh


def fetch_parc(data_dir=None, hemi='L', key='FGC', url=None, sha256=None,
               download=True):

    #如果key使到图谱的路径，则直接返回路径
    if os.path.exists(key):
        parc = nib.load(key)
        return parc

    if data_dir is None:
        packaged = Path(__file__).parent / 'surfaces' / 'parcellations'
        packaged_file = (
            packaged /
            f'{ATLAS.get(key, key)}.fs_LR_32k.{hemi}.label.gii'
        )
        data_dir = (
            packaged if packaged_file.exists()
            else get_data_dir() / f'parcellation-{key}'
        )
    else:
        data_dir = Path(data_dir)
    
    # Retrieve the atlas name from ATLAS; if the key is not found, use the key itself as the fallback.
    atlas = ATLAS.get(key, key)
    fn = f'{atlas}.fs_LR_32k.{hemi}.label.gii'
    parc_path = data_dir / fn

    if not parc_path.exists():
        if url is None or sha256 is None:
            raise FileNotFoundError(
                f"Parcellation is not cached: {parc_path}. Supply an official "
                "direct url and sha256 to download it."
            )
        parc_path = fetch_resource(
            f'parcellation-{key}', url=url, sha256=sha256, filename=fn,
            data_dir=get_data_dir() if data_dir is None else Path(data_dir).parent,
            download=download,
        )
    
    parc = nib.load(str(parc_path))
    return parc


def fetch_annot(data_dir=None, atlas='FGC', res='1mm', annot=True,
                volume_url=None, volume_sha256=None, annotation_url=None,
                annotation_sha256=None, download=True):
    data_root = get_data_dir(data_dir)
    volume_dir = data_root / f'atlas-{atlas}-volumes'

    if atlas in ['D99','MacBN']:
        fn = f'{atlas}_NMT2asym.nii.gz'
    elif atlas in ['MacBN_human','economo','FGC','BN']:
        fn = f'{atlas}_MNI152_{res}.nii.gz'
    else:
        raise ValueError(f"Unsupported atlas: {atlas}")

    parc_path = volume_dir / fn
    if not parc_path.exists():
        if volume_url is None or volume_sha256 is None:
            raise FileNotFoundError(
                f"Atlas volume is not cached: {parc_path}. Supply volume_url "
                "and volume_sha256 from the atlas provider."
            )
        parc_path = fetch_resource(
            f'atlas-{atlas}-volumes', url=volume_url, sha256=volume_sha256,
            filename=fn, data_dir=data_root, download=download,
        )
    parc = str(parc_path)

    if annot:
        annotation_atlas = 'MacBN' if atlas == 'MacBN_human' else atlas
        annot_dir = data_root / f'atlas-{annotation_atlas}-annotation'
        annot_path = annot_dir / f'{annotation_atlas}_annot.csv'
        if not annot_path.exists():
            if annotation_url is None or annotation_sha256 is None:
                raise FileNotFoundError(
                    f"Atlas annotation is not cached: {annot_path}. Supply "
                    "annotation_url and annotation_sha256."
                )
            annot_path = fetch_resource(
                f'atlas-{annotation_atlas}-annotation', url=annotation_url,
                sha256=annotation_sha256, filename=annot_path.name,
                data_dir=data_root, download=download,
            )
        annot = pd.read_csv(str(annot_path),index_col=0)
        return parc, annot
    else:
        return parc
