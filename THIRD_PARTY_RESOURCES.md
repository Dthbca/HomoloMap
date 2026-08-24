# Third-party resources

HomoloMap distributes its project-derived D99 and Brainnetome cell-type maps
and the associated cell-type homology table. External atlas surfaces,
parcellations, annotations, and imaging-derived phenotypes remain governed by
their original providers' licenses and citation requirements.

## Resource boundary

- **fsLR surfaces** are obtained through
  [`neuromaps.datasets.fetch_fslr`](https://netneurolab.github.io/neuromaps/generated/neuromaps.datasets.fetch_fslr.html).
  Neuromaps manages its download metadata and cache.
- **Brainnetome atlas resources** should be obtained from the
  [official Brainnetome resource page](https://www.brainnetome.org/resource/).
- **D99 atlas resources** should be obtained from the
  [official AFNI D99 distribution](https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/nonhuman/macaque_tempatl/atlas_d99v2.html).
- **Yerkes19 and other provider files** are downloaded only from a user-supplied
  URL and require a caller-supplied SHA256 checksum.
- **ENIGMA disease maps and other brain IDPs** are user-supplied inputs.
  HomoloMap neither bundles nor downloads them.

Small left-hemisphere fsLR/Brainnetome files currently included for the offline
plotting example retain their upstream provenance and are not relicensed by
HomoloMap. Before redistributing a release, maintainers should verify the
upstream terms for every packaged third-party file. Users remain responsible
for provider-specific access, attribution, and redistribution conditions.

## Caching and integrity

Downloaded resources are stored outside the installed source tree. Set
`HOMOLOMAP_DATA` or pass `data_dir=` to select the cache. Explicit provider
downloads are written atomically and accepted only when their SHA256 matches.
Set `download=False` for strict offline use.