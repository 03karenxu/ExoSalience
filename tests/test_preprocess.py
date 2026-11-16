import numpy as np
import pandas as pd

from src.preprocessing.datatypes import StarCurve
from src.preprocessing.preprocess import bin, fill_gaps, fit_curve_to_spline, phase_fold, rescale

# fake star: slow 1% wiggle, 0.5% transits every 3 days lasting 2.4 hours, a 5-day gap
TCES = pd.DataFrame({"tce_period": [3.0], "tce_time0bk": [1.0], "tce_duration": [2.4]})


def test_spline_flattens_baseline_and_keeps_dips():
    rng = np.random.default_rng(0)
    t = np.arange(0, 90, 0.0204)
    t = t[(t < 40) | (t > 45)]
    in_transit = np.abs((t - 1.0 + 1.5) % 3.0 - 1.5) < 0.05
    flux = (1 + 0.01 * np.sin(2 * np.pi * t / 10)) * np.where(in_transit, 0.995, 1.0)
    flux += rng.normal(0, 1e-4, len(t))

    flat = fit_curve_to_spline(StarCurve(kepid=1, time=t, flux=flux, flux_err=np.full(len(t), 1e-4)), TCES)

    assert len(flat.time) == len(t)
    assert np.std(flat.flux[~in_transit]) < 2e-4  # 1% wiggle removed, only noise left
    assert abs(np.median(flat.flux[in_transit]) - 0.995) < 3e-4  # transit depth preserved


def test_fold_bin_fill_rescale():
    t = np.arange(0, 30, 0.01)
    flux = np.where(np.abs((t - 1.0 + 1.5) % 3.0 - 1.5) < 0.05, 0.99, 1.0)
    folded = phase_fold(StarCurve(kepid=1, time=t, flux=flux, flux_err=np.ones(len(t))), period=3.0, t0=1.0)
    assert np.all(np.diff(folded.phase) >= 0) and folded.phase.min() >= -1.5 and folded.phase.max() < 1.5

    view = bin(folded, num_bins=31, half_width=0.31)
    assert len(view) == 31 and view[15] == 0.99 and view[0] == 1.0

    assert list(fill_gaps(np.array([1.0, np.nan, 3.0]))) == [1.0, 2.0, 3.0]
    assert list(rescale(np.array([1.0, 1.0, 0.98, 1.0]))) == [0.0, 0.0, -1.0, 0.0]
