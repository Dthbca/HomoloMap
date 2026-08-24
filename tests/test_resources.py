import hashlib
from pathlib import Path

import pytest

from HomoloMap.datasets.resources import fetch_resource, file_sha256, get_data_dir


def test_fetch_resource_downloads_and_reuses_verified_file(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"HomoloMap atlas fixture")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    cache = tmp_path / "cache"
    fetched = fetch_resource(
        "fixture", url=source.as_uri(), sha256=expected,
        filename="atlas.bin", data_dir=cache,
    )
    assert fetched == get_data_dir(cache) / "fixture" / "atlas.bin"
    assert file_sha256(fetched) == expected
    assert fetch_resource(
        "fixture", url=source.as_uri(), sha256=expected,
        filename="atlas.bin", data_dir=cache, download=False,
    ) == fetched


def test_fetch_resource_rejects_bad_checksum(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"not the declared resource")
    with pytest.raises(OSError, match="SHA256 mismatch"):
        fetch_resource(
            "fixture", url=source.as_uri(), sha256="0" * 64,
            filename="atlas.bin", data_dir=tmp_path / "cache",
        )


def test_fetch_resource_offline_requires_cache(tmp_path):
    with pytest.raises(FileNotFoundError, match="not cached"):
        fetch_resource(
            "fixture", url="https://example.org/atlas.bin", sha256="0" * 64,
            filename="atlas.bin", data_dir=tmp_path / "cache", download=False,
        )
