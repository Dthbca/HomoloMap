import hashlib
from pathlib import Path

import pytest

from HomoloMap.datasets.resources import fetch_resource, file_sha256, get_data_dir
from HomoloMap.datasets import atlas as atlas_module


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


def test_fetch_fslr_offline_never_calls_downloader(tmp_path, monkeypatch):
    """Strict offline mode must fail before neuromaps can access the network."""
    called = False

    def forbidden_download(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("neuromaps downloader was called in offline mode")

    import neuromaps.datasets

    monkeypatch.setattr(neuromaps.datasets, "fetch_fslr", forbidden_download)
    with pytest.raises(FileNotFoundError, match="offline"):
        atlas_module.fetch_fslr(
            density="32k", hemi="R", surf="very_inflated",
            base_dir=tmp_path, download=False, return_path=True,
        )
    assert not called
