"""Regenerate the repository payload checksum manifest."""

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).parents[1]
OUTPUT = ROOT / "MANIFEST_SHA256.csv"
EXCLUDED_PARTS = {
    ".git", ".pytest_cache", "build", "HomoloMap.egg-info", "__pycache__"
}


files = sorted(
    path for path in ROOT.rglob("*")
    if path.is_file()
    and path != OUTPUT
    and not any(part in EXCLUDED_PARTS for part in path.parts)
    and path.suffix not in {".pyc", ".pyo"}
)

with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"])
    writer.writeheader()
    for path in files:
        payload = path.read_bytes()
        writer.writerow({
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })

print(f"PASS files={len(files)} manifest={OUTPUT}")
