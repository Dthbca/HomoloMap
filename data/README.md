# HomoloMap example spatial maps

This directory contains original D99 cell-type maps and their HomoloMap-derived Brainnetome representations. These are data products, not statistical results from the MEG, ENIGMA, or BigBrain analyses.

## Files

### Original D99 maps

- `maps/D99/ctype_ratio_plot_D99.csv`: 132 D99 regions × 226 fine cell-type features. Values are the source ratio representation. The original rows are not all exactly closed; row sums range from approximately 1.0000 to 1.0681.
- `maps/D99/ctype_density_plot_D99.csv`: 141 D99 regions × 257 fine cell-type features. Values are continuous density estimates and must not be interpreted as a closed composition.

### HomoloMap-derived BN maps

- `maps/BN/ctype_ratio_BN_23_subclass.csv`: 105 BN cortical regions × 23 subclasses.
- `maps/BN/ctype_ratio_BN_71_cluster.csv`: 105 BN cortical regions × 71 clusters.

Both BN tables were regenerated and byte-for-byte verified against the maps
produced with the canonical `cluster_mapping_dict.csv` (SHA256
`996255b8ad827615dcbe786ef2ae0a3b7dcaae062971d1b4eeb445726f9a8377`).
Source feature names are joined to the mapping's `plot` column by exact match.

Both BN tables are mapped compositions: only source features resolved by the supplied mapping table are represented. Each row was reclosed after mapping and regional relabeling, so the 23 or 71 components sum to one within numerical precision. Values therefore mean relative composition among successfully mapped cell types, not absolute abundance among all cells.

## Mapping provenance

- Canonical mapping source: `Macaque_ST/notebook/cluster_mapping_dict.csv`.
- Verified canonical SHA256: `996255b8ad827615dcbe786ef2ae0a3b7dcaae062971d1b4eeb445726f9a8377`.
- Mapping key: source feature name matched exactly to the `plot` column in `mappings/cluster_mapping_dict.csv`.
- Mapping coverage: 191 of 226 ratio features.
- The full mapping table contains 24 subclasses and 72 clusters; the released ratio source resolves to 23 subclasses and 71 clusters because not every mapping category is represented among the 191 mapped ratio features.
- Unresolved features: 35; listed in `metadata/unmapped_features.csv`.
- Excluded source ratio mass: 0.112938 overall; median 0.113274 and maximum 0.151287 across D99 regions.
- Regional target: 105 cortical Brainnetome labels (`1, 3, ..., 209`).
- D99 labels without target contribution: 106, 118, and 194.
- Regional aggregation: mean-based HomoloMap relabeling followed by target-space reclosure.
- No CLR transformation was applied to the released BN ratio tables.

The complete machine-readable provenance is stored in:

- `metadata/BN_23_subclass_mapping_audit.json`
- `metadata/BN_71_cluster_mapping_audit.json`
- `metadata/MAP_MANIFEST.csv`
- `metadata/MAPPING_AUDIT.json`

The canonical file is preserved byte-for-byte as `mappings/cluster_mapping_dict.csv`. For easier browsing, `mappings/cluster_mapping_table.csv` contains the same 400 mappings with explicit `source_plot` naming and stable sorting. `mappings/cluster_mapping_summary.csv` reports the number of source plots per subclass and cluster.

## Loading the maps

```python
import pandas as pd

subclass = pd.read_csv(
    "data/maps/BN/ctype_ratio_BN_23_subclass.csv",
    index_col=0,
)
cluster = pd.read_csv(
    "data/maps/BN/ctype_ratio_BN_71_cluster.csv",
    index_col=0,
)

assert subclass.shape == (105, 23)
assert cluster.shape == (105, 71)
```

## Interpretation boundary

The maps are atlas-level spatial estimates. They do not provide participant-level measurements and should not be interpreted as causal effects, direct human cell counts, or evidence that homologous cell types have identical abundance or molecular state in every biological reference.
