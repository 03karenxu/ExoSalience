"""load one star's Kepler long-cadence light curves

from_dir(kepid) and from_s3(kepid) both return one Quarter per quarter file, sorted by quarter:

    [
        Quarter(
            quarter=3,                        # int, Kepler quarter (0-17)
            time=array([260.2, ...]),         # float64, days (BKJD)
            flux=array([1.53e5, ...]),        # float64, PDCSAP_FLUX in e-/s; may contain NaN
            flux_err=array([...]),            # float64, uncertainty of flux
            quality=array([0, 0, 128, ...]),  # int64, 0 = good, nonzero = flagged problem
        ),
        Quarter(quarter=4, ...),
        ...
    ]

"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import BinaryIO

from astropy.io import fits
from src.preprocessing.datatypes import Quarter

_S3_DIR = "stpubdata/kepler/public/lightcurves"

def _star_file_pattern(kepid: int | str) -> str:
    # MAST file layout: XXXX/XXXXXXXXX/kplrXXXXXXXXX-<timestamp>_llc.fits
    kid = f"{int(kepid):09d}" # mast uses 9-digit kepid padded with leading 0s
    return f"{kid[:4]}/{kid}/kplr{kid}-*_llc.fits"


def _read(f: str | Path | BinaryIO) -> Quarter:
    """read one quarter file into a Quarter of float64/int64 numpy arrays"""
    with fits.open(f, memmap=False) as hdul:
        lc = hdul["LIGHTCURVE"].data
        return Quarter(
            quarter=hdul[0].header["QUARTER"],
            time=np.array(lc["TIME"], dtype=float),  # BKJD
            flux=np.array(lc["PDCSAP_FLUX"], dtype=float),  # flux corrected for error
            flux_err=np.array(lc["PDCSAP_FLUX_ERR"], dtype=float),
            quality=np.array(lc["SAP_QUALITY"], dtype=int),
        )


def _sorted(quarters: list[Quarter], kepid: int | str, where: str | Path) -> list[Quarter]:
    if not quarters:
        raise FileNotFoundError(f"no long-cadence light curves for kepid {kepid} in {where}")
    return sorted(quarters, key=lambda q: (q.quarter, q.time[0]))


def from_dir(kepid: int | str, data_dir: str | Path) -> list[Quarter]:
    """load all quarters for a star from a local directory"""
    paths = Path(data_dir).glob(_star_file_pattern(kepid))
    return _sorted([_read(p) for p in paths], kepid, data_dir)


def from_s3(kepid: int | str, fs=None) -> list[Quarter]:
    """stream all quarters for a star from the MAST S3 bucket"""
    import s3fs

    fs = fs or s3fs.S3FileSystem(anon=True)
    quarters = []
    for path in fs.glob(f"{_S3_DIR}/{_star_file_pattern(kepid)}"):
        with fs.open(path, "rb") as f:
            quarters.append(_read(f))
    return _sorted(quarters, kepid, f"s3://{_S3_DIR}")
