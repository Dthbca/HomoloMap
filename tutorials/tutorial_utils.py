"""Small shared helpers used by the HomoloMap tutorial notebooks."""

from pathlib import Path

import numpy as np
import pandas as pd


BN_LABELS = np.arange(1, 210, 2, dtype=int)


def find_repo_root(start=None):
    """Find the repository root from the root or ``tutorials`` directory."""
    start = Path.cwd() if start is None else Path(start)
    for root in (start, start.parent):
        if (root / "data" / "maps" / "BN").exists():
            return root
    raise FileNotFoundError("Run the notebook from the repository or tutorials directory.")


def load_celltype_map(level="subclass", root=None):
    """Load a released BN ratio map at subclass or cluster resolution."""
    root = find_repo_root() if root is None else Path(root)
    filenames = {
        "subclass": "ctype_ratio_BN_23_subclass.csv",
        "cluster": "ctype_ratio_BN_71_cluster.csv",
    }
    if level not in filenames:
        raise ValueError("level must be 'subclass' or 'cluster'")
    data = pd.read_csv(root / "data" / "maps" / "BN" / filenames[level], index_col=0)
    data.index = data.index.astype(int)
    return data


def load_idps(path=None, index=None, seed=42):
    """Load user IDPs or create deterministic toy maps for execution checks."""
    if path is not None:
        data = pd.read_csv(path, index_col=0)
        data.index = data.index.astype(int)
        return data, False
    labels = BN_LABELS if index is None else np.asarray(index, dtype=int)
    rng = np.random.default_rng(seed)
    position = np.linspace(-1, 1, len(labels))
    data = pd.DataFrame(
        {
            "toy_IDP_1": position + rng.normal(0, 0.35, len(labels)),
            "toy_IDP_2": np.sin(np.pi * position) + rng.normal(0, 0.35, len(labels)),
        },
        index=labels,
    )
    return data, True


def align_and_validate(celltypes, idps, require_complete_bn=False):
    """Align tables by ROI label and validate the released composition."""
    if not celltypes.index.is_unique or not idps.index.is_unique:
        raise ValueError("ROI labels must be unique")
    shared = celltypes.index.intersection(idps.index, sort=False)
    x = celltypes.loc[shared].astype(float)
    y = idps.loc[shared].apply(pd.to_numeric, errors="coerce")
    valid = x.notna().all(axis=1) & y.notna().all(axis=1)
    x, y = x.loc[valid], y.loc[valid]
    if not np.isfinite(x.to_numpy()).all() or (x.to_numpy() < 0).any():
        raise ValueError("Cell-type map must be finite and non-negative")
    if not np.allclose(x.sum(axis=1), 1.0):
        raise ValueError("Ratio map is not closed within ROI")
    if require_complete_bn and not np.array_equal(x.index.to_numpy(), BN_LABELS):
        raise ValueError("This analysis requires all 105 BN cortical labels")
    return x, y


def load_prepared_or_example(root=None, level="subclass", idp_path=None, seed=42):
    """Load Tutorial 1 outputs when present, otherwise build example inputs."""
    root = find_repo_root() if root is None else Path(root)
    output = root / "tutorial_outputs"
    x_path = output / "aligned_celltype_predictors.csv"
    y_path = output / "aligned_brain_idps.csv"
    if x_path.exists() and y_path.exists():
        x = pd.read_csv(x_path, index_col=0)
        y = pd.read_csv(y_path, index_col=0)
        x.index = x.index.astype(int)
        y.index = y.index.astype(int)
        return x, y
    x = load_celltype_map(level, root)
    y, _ = load_idps(idp_path, x.index, seed)
    return align_and_validate(x, y)
