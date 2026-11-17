from pathlib import Path

import numpy as np
import pytest

from src.paths import DATA_DIR
from src.preprocessing.build_dataset import load_catalogue, main

DR24 = DATA_DIR / "tce_catalogue.csv"
KOI = DATA_DIR / "koi_catalogue.csv"


def _check_outputs(out_dir: Path, catalogue_path: Path):
    """train/test share no stars, views have the right shapes, labels match the catalogue"""
    train, test = np.load(out_dir / "train.npz"), np.load(out_dir / "test.npz")
    assert set(train["kepid"]).isdisjoint(test["kepid"])

    catalogue = load_catalogue(catalogue_path).set_index(["kepid", "planet_num"])
    for d in (train, test):
        n = len(d["label"])
        assert d["global_view"].shape == (n, 2001) and d["local_view"].shape == (n, 201)
        for kepid, planet_num, label in zip(d["kepid"], d["planet_num"], d["label"]):
            assert catalogue.loc[(kepid, planet_num), "is_exoplanet"] == label


@pytest.mark.network
def test_build_dataset_and_resume(tmp_path):
    main(DR24, limit=3, workers=2, out_dir=tmp_path)
    _check_outputs(tmp_path, DR24)

    # a second run reuses the saved chunks instead of rebuilding them
    chunk = next((tmp_path / "chunks").glob("*/train_0000.npz"))
    mtime = chunk.stat().st_mtime
    main(DR24, limit=3, workers=2, out_dir=tmp_path)
    assert chunk.stat().st_mtime == mtime


@pytest.mark.network
def test_build_dataset_from_koi_table(tmp_path):
    koi_csv = KOI
    main(koi_csv, limit=3, workers=2, out_dir=tmp_path)
    _check_outputs(tmp_path, koi_csv)


def test_load_catalogue_generic_columns():
    for path in (DR24, KOI):
        catalogue = load_catalogue(path)
        assert {"kepid", "period", "t0", "duration", "planet_num", "is_exoplanet"} <= set(catalogue.columns)
        assert set(catalogue.is_exoplanet.dropna().unique()) == {0, 1}
