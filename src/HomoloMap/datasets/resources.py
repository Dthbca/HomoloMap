"""Safe, checksum-verified retrieval of third-party atlas resources."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def get_data_dir(data_dir=None) -> Path:
    """Return the HomoloMap cache directory and create it when needed."""
    if data_dir is not None:
        root = Path(data_dir).expanduser()
    elif os.environ.get("HOMOLOMAP_DATA"):
        root = Path(os.environ["HOMOLOMAP_DATA"]).expanduser()
    elif os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        root = Path(os.environ["LOCALAPPDATA"]) / "HomoloMap"
    elif os.environ.get("XDG_CACHE_HOME"):
        root = Path(os.environ["XDG_CACHE_HOME"]) / "HomoloMap"
    else:
        root = Path.home() / ".cache" / "HomoloMap"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def file_sha256(path, chunk_size=1024 * 1024) -> str:
    """Calculate SHA256 without loading the whole resource into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_resource(name, *, url, sha256, filename=None, data_dir=None,
                   download=True, force=False, timeout=60):
    """Download one declared resource atomically and verify its checksum.

    Parameters
    ----------
    name : str
        Stable resource identifier, used as a cache subdirectory.
    url : str
        Direct HTTPS URL supplied by the resource provider.
    sha256 : str
        Expected 64-character SHA256 digest. Downloads without a fixed digest
        are intentionally rejected.
    filename : str, optional
        Cached filename. Defaults to the final URL path component.
    data_dir : path-like, optional
        Cache root. Defaults to ``HOMOLOMAP_DATA`` or the user cache directory.
    download : bool, default=True
        If False, only an already cached and verified file may be returned.
    force : bool, default=False
        Redownload even when a valid cached file exists.
    """
    if not name or Path(name).name != name:
        raise ValueError("name must be one safe path component")
    parsed = urlparse(url)
    if parsed.scheme not in {"https", "file"}:
        raise ValueError("url must use HTTPS (or file:// for local testing)")
    expected = str(sha256).lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("sha256 must be a fixed 64-character hexadecimal digest")
    filename = filename or Path(parsed.path).name
    if not filename or Path(filename).name != filename:
        raise ValueError("filename must be one safe path component")

    target_dir = get_data_dir(data_dir) / name
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    if target.exists() and not force:
        if file_sha256(target) == expected:
            return target
        if not download:
            raise OSError(f"Cached resource failed SHA256 verification: {target}")
    if not download:
        raise FileNotFoundError(
            f"Resource is not cached: {target}. Enable download or provide data_dir."
        )

    request = Request(url, headers={"User-Agent": "HomoloMap/0.1"})
    fd, temporary = tempfile.mkstemp(prefix=filename + ".", suffix=".part",
                                     dir=target_dir)
    os.close(fd)
    temporary = Path(temporary)
    try:
        with urlopen(request, timeout=timeout) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        observed = file_sha256(temporary)
        if observed != expected:
            raise OSError(
                f"SHA256 mismatch for {name}: expected {expected}, observed {observed}"
            )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
