from pathlib import Path

import numpy as np

from src.preprocessing.build_dataset import main

ROOT = Path(__file__).resolve().parents[1]
STARS = {11446443: 1, 4912991: 0, 9906802: 0}  # PC, AFP, NTP in data/test


def test_build_dataset_from_test_stars(tmp_path):
    main(ROOT / "data" / "dr24-tce-2.csv", mast_dir=ROOT / "data" / "test", workers=2, out_dir=tmp_path)
    train, test = np.load(tmp_path / "train.npz"), np.load(tmp_path / "test.npz")

    assert set(train["kepid"]).isdisjoint(test["kepid"])
    assert set(train["kepid"]) | set(test["kepid"]) == set(STARS)
    for d in (train, test):
        n = len(d["label"])
        assert d["global_view"].shape == (n, 2001) and d["local_view"].shape == (n, 201)
        assert {int(k): int(l) for k, l in zip(d["kepid"], d["label"])}.items() <= STARS.items()


def test_build_dataset_resumes_from_chunks(tmp_path):
    main(ROOT / "data" / "dr24-tce-2.csv", mast_dir=ROOT / "data" / "test", workers=2, out_dir=tmp_path)
    chunk = next((tmp_path / "chunks").glob("*/train_0000.npz"))
    mtime = chunk.stat().st_mtime
    main(ROOT / "data" / "dr24-tce-2.csv", mast_dir=ROOT / "data" / "test", workers=2, out_dir=tmp_path)
    assert chunk.stat().st_mtime == mtime
