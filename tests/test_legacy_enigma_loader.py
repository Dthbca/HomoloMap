import pandas as pd
import pytest

from HomoloMap.datasets import atlas as atlas_module


def test_fetch_enigma_is_not_top_level_public_api():
    import HomoloMap.datasets as datasets

    assert not hasattr(datasets, "fetch_enigma")
    assert "fetch_enigma" not in datasets.__all__


def test_fetch_enigma_requires_explicit_external_table():
    with pytest.warns(FutureWarning, match="legacy compatibility loader"):
        with pytest.raises(FileNotFoundError, match="does not bundle"):
            atlas_module.fetch_enigma()


def test_fetch_enigma_reads_and_filters_legacy_dk_table(tmp_path):
    path = tmp_path / "enigma.csv"
    pd.DataFrame(
        {"hemi": ["L", "R"], "disorder": [0.25, -0.10]},
        index=[1, 2],
    ).to_csv(path)
    with pytest.warns(FutureWarning):
        result = atlas_module.fetch_enigma(path=path, atlas="DK")
    assert list(result.index) == [1]
    assert list(result.columns) == ["disorder"]
    assert result.loc[1, "disorder"] == pytest.approx(0.25)