"""Validate the canonical mapping and generate public derived tables."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parents[1]
DATA = ROOT / "data"
CANONICAL = DATA / "mappings" / "cluster_mapping_dict.csv"
NORMALIZED = DATA / "mappings" / "cluster_mapping_table.csv"
SUMMARY = DATA / "mappings" / "cluster_mapping_summary.csv"
AUDIT = DATA / "metadata" / "MAPPING_AUDIT.json"

EXPECTED_SHA256 = "996255b8ad827615dcbe786ef2ae0a3b7dcaae062971d1b4eeb445726f9a8377"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if sha256(CANONICAL) != EXPECTED_SHA256:
    raise ValueError("Canonical mapping does not match the verified source file.")

mapping = pd.read_csv(CANONICAL, dtype=str)
if mapping.columns.tolist() != ["plot", "cluster", "subclass"]:
    raise ValueError(f"Unexpected mapping columns: {mapping.columns.tolist()}")
if mapping.shape != (400, 3):
    raise ValueError(f"Unexpected mapping shape: {mapping.shape}")
if mapping.isna().any().any() or mapping["plot"].duplicated().any():
    raise ValueError("Canonical mapping contains missing or duplicate plot keys.")

normalized = mapping.rename(columns={"plot": "source_plot"}).copy()
normalized = normalized.sort_values(
    ["subclass", "cluster", "source_plot"], kind="stable"
).reset_index(drop=True)
normalized.to_csv(NORMALIZED, index=False)

subclass_summary = (
    mapping.groupby("subclass", sort=True)
    .agg(n_source_plots=("plot", "nunique"), n_clusters=("cluster", "nunique"))
    .reset_index()
    .rename(columns={"subclass": "label"})
)
subclass_summary.insert(0, "level", "subclass")

cluster_summary = (
    mapping.groupby(["cluster", "subclass"], sort=True)
    .agg(n_source_plots=("plot", "nunique"))
    .reset_index()
    .rename(columns={"cluster": "label", "subclass": "parent_subclass"})
)
cluster_summary.insert(0, "level", "cluster")
cluster_summary["n_clusters"] = 1

summary = pd.concat(
    [
        subclass_summary.assign(parent_subclass="")[
            ["level", "label", "parent_subclass", "n_source_plots", "n_clusters"]
        ],
        cluster_summary[
            ["level", "label", "parent_subclass", "n_source_plots", "n_clusters"]
        ],
    ],
    ignore_index=True,
)
summary.to_csv(SUMMARY, index=False)

coverage = {}
mapping_keys = set(mapping["plot"])
for representation in ["ratio", "density"]:
    path = DATA / "maps" / "D99" / f"ctype_{representation}_plot_D99.csv"
    table = pd.read_csv(path, index_col=0)
    columns = set(map(str, table.columns))
    mapped = columns & mapping_keys
    unresolved = columns - mapping_keys
    coverage[representation] = {
        "source_features": len(columns),
        "mapped_features": len(mapped),
        "unresolved_features": len(unresolved),
        "feature_coverage_fraction": len(mapped) / len(columns),
        "unresolved_feature_names": sorted(unresolved),
    }

audit = {
    "canonical_file": "data/mappings/cluster_mapping_dict.csv",
    "canonical_source_filename": "Macaque_ST/notebook/cluster_mapping_dict.csv",
    "canonical_bytes": CANONICAL.stat().st_size,
    "canonical_sha256": sha256(CANONICAL),
    "shape": list(mapping.shape),
    "unique_plot": int(mapping["plot"].nunique()),
    "unique_cluster": int(mapping["cluster"].nunique()),
    "unique_subclass": int(mapping["subclass"].nunique()),
    "missing_values": int(mapping.isna().sum().sum()),
    "duplicate_plot": int(mapping["plot"].duplicated().sum()),
    "coverage": coverage,
    "derived_files": [
        "data/mappings/cluster_mapping_table.csv",
        "data/mappings/cluster_mapping_summary.csv",
    ],
}
AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

print(
    "PASS "
    f"plots={audit['unique_plot']} clusters={audit['unique_cluster']} "
    f"subclasses={audit['unique_subclass']}"
)

