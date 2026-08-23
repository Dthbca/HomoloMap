"""Validate released map tables and write their checksum manifest."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parents[1]
DATA = ROOT / "data"
OUTPUT = DATA / "metadata" / "MAP_MANIFEST.csv"

EXPECTED = {
    "maps/D99/ctype_ratio_plot_D99.csv": (132, 226, False),
    "maps/D99/ctype_density_plot_D99.csv": (141, 257, False),
    "maps/BN/ctype_ratio_BN_23_subclass.csv": (105, 23, True),
    "maps/BN/ctype_ratio_BN_71_cluster.csv": (105, 71, True),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


rows = []
for relative, (n_rows, n_columns, closed) in EXPECTED.items():
    path = DATA / relative
    table = pd.read_csv(path, index_col=0)
    values = table.to_numpy(dtype=float)
    if table.shape != (n_rows, n_columns):
        raise ValueError(f"Unexpected shape for {relative}: {table.shape}")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Invalid values in {relative}")
    row_sums = values.sum(axis=1)
    if closed and not np.allclose(row_sums, 1.0):
        raise ValueError(f"Released composition is not closed: {relative}")
    rows.append(
        {
            "path": relative,
            "rows": n_rows,
            "columns": n_columns,
            "index_min": table.index.min(),
            "index_max": table.index.max(),
            "minimum": values.min(),
            "maximum": values.max(),
            "row_sum_min": row_sums.min(),
            "row_sum_max": row_sums.max(),
            "closed": closed,
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
    )

with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print(f"PASS maps={len(rows)} manifest={OUTPUT}")

