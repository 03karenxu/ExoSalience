
import numpy as np
import pytest

from src.paths import DATA_DIR
from src.preprocessing.load import from_dir, from_s3

TEST_DIR = DATA_DIR / "test"
KEPID = 11446443  # Kepler-1b, 15 quarter files in TEST_DIR
QUARTERS = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17]


def test_from_dir():
    qs = from_dir(KEPID, TEST_DIR)
    assert [q.quarter for q in qs] == QUARTERS
    for q in qs:
        assert len(q.time) == len(q.flux) == len(q.quality) > 0
        assert q.flux.dtype == np.float64


def test_from_dir_missing_star():
    with pytest.raises(FileNotFoundError):
        from_dir(1, TEST_DIR)


@pytest.mark.network
def test_from_s3_matches_dir():
    local = {q.quarter: q for q in from_dir(KEPID, TEST_DIR)}
    remote = [q for q in from_s3(KEPID) if q.quarter in local]
    assert len(remote) == len(local)
    for q in remote:
        np.testing.assert_array_equal(q.flux, local[q.quarter].flux)
