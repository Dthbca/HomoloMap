import pandas as pd

from HomoloMap.datasets import atlas as atlas_module
from HomoloMap.transforms import load_data


def test_load_data_accepts_series_dataframe_and_csv(tmp_path):
    expected = pd.DataFrame({"idp": [0.1, 0.2, 0.3]}, index=[1, 3, 5])
    expected.index.name = "roi"

    from_frame = load_data(
        expected, atlas="BN", trg="BN", smooth=False,
    )
    from_series = load_data(
        expected["idp"], atlas="BN", trg="BN", smooth=False,
    )
    path = tmp_path / "brain_idps.csv"
    expected.to_csv(path)
    from_csv = load_data(path, atlas="BN", trg="BN", smooth=False)

    pd.testing.assert_frame_equal(from_frame, expected)
    pd.testing.assert_frame_equal(from_series, expected)
    pd.testing.assert_frame_equal(from_csv, expected)


def test_packaged_parcellation_does_not_initialize_user_cache(monkeypatch):
    def forbidden_cache(*args, **kwargs):
        raise AssertionError("user cache was accessed for a packaged atlas")

    monkeypatch.setattr(atlas_module, "get_data_dir", forbidden_cache)
    parc = atlas_module.fetch_parc(key="BN", hemi="L")
    assert parc.darrays
