"""load one star's Kepler long-cadence light curves

from_dir(kepid) and from_s3(kepid) both return one dict per quarter file, sorted by quarter:

    [
        {
            "quarter":  3,                        # int, Kepler quarter (0-17)
            "time":     array([260.2, ...]),      # float64, days (BKJD)
            "flux":     array([1.53e5, ...]),     # float64, PDCSAP_FLUX in e-/s; may contain NaN
            "flux_err": array([...]),             # float64, uncertainty of flux
            "quality":  array([0, 0, 128, ...]),  # int64, 0 = good, nonzero = flagged problem
        },
        {"quarter": 4, ...},
        ...
    ]
"""

import os
import numpy as np
from glob import glob
from astropy.io import fits


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(ROOT, "data/raw")
S3_DIR = "stpubdata/kepler/public/lightcurves"


def _star_file_pattern(kepid):
    # MAST file layout: XXXX/XXXXXXXXX/kplrXXXXXXXXX-<timestamp>_llc.fits
    kid = f"{int(kepid):09d}" # mast uses 9-digit kepid padded with leading 0s
    return f"{kid[:4]}/{kid}/kplr{kid}-*_llc.fits"


def _read(f):
    """read one quarter file (path or file object) into a dict of float64/int64 numpy arrays (copied out of the FITS file)."""
    with fits.open(f, memmap=False) as hdul:
        lc = hdul["LIGHTCURVE"].data
        return {
            "quarter": hdul[0].header["QUARTER"],
            "time": np.array(lc["TIME"], dtype=float),  # BKJD
            "flux": np.array(lc["PDCSAP_FLUX"], dtype=float),  # flux corrected for error
            "flux_err": np.array(lc["PDCSAP_FLUX_ERR"], dtype=float),
            "quality": np.array(lc["SAP_QUALITY"], dtype=int),
        }


def _sorted(quarters, kepid, where):
    if not quarters:
        raise FileNotFoundError(f"no long-cadence light curves for kepid {kepid} in {where}")
    return sorted(quarters, key=lambda q: (q["quarter"], q["time"][0]))


def from_dir(kepid, data_dir=RAW_DIR):
    """load all quarters for a star from a local directory"""
    paths = glob(os.path.join(data_dir, _star_file_pattern(kepid)))
    return _sorted([_read(p) for p in paths], kepid, data_dir)


def from_s3(kepid, fs=None):
    """stream all quarters for a star from the MAST S3 bucket"""
    import s3fs

    fs = fs or s3fs.S3FileSystem(anon=True)
    quarters = []
    for path in fs.glob(f"{S3_DIR}/{_star_file_pattern(kepid)}"):
        with fs.open(path, "rb") as f:
            quarters.append(_read(f))
    return _sorted(quarters, kepid, f"s3://{S3_DIR}")
