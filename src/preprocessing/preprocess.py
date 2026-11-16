import numpy as np
import pandas as pd
from dataclasses import replace
from lightkurve.utils import KeplerQualityFlags
from scipy.interpolate import LSQUnivariateSpline
from scipy.stats import binned_statistic

from src.preprocessing.datatypes import FoldedCurve, Quarter, StarCurve


def drop_bad_points(q: Quarter) -> Quarter:
    """drop measurements with NaN time/flux or a serious quality flag"""
    keep = (
        np.isfinite(q.time) &
        np.isfinite(q.flux) &
        np.isfinite(q.flux_err) &
        (q.quality & KeplerQualityFlags.DEFAULT_BITMASK == 0)
    )
    return replace(
        q,
        time=q.time[keep],
        flux=q.flux[keep],
        flux_err=q.flux_err[keep],
        quality=q.quality[keep]
    )


def scale(q: Quarter) -> Quarter:
    """scale quarter's flux and flux error so that median is 1"""
    median = np.median(q.flux)
    return replace(
        q,
        flux=q.flux / median,
        flux_err=q.flux_err / median
    )


def stitch_quarters(kepid: int, qs: list[Quarter]) -> StarCurve:
    """stitches all quarters together - assumes qs contains only quarters for the given kepid"""
    return StarCurve(
        kepid=kepid,
        time=np.concatenate([q.time for q in qs]),
        flux=np.concatenate([q.flux for q in qs]),
        flux_err=np.concatenate([q.flux_err for q in qs])
    )


def _phases(times: np.ndarray, period: float, t0: float) -> np.ndarray:
    return (times - t0 + period / 2) % period - period / 2

def _transit_mask(
        times: np.ndarray,
        period: float,
        t0: float,
        duration_h: float,
        width: float = 1.5) -> np.ndarray:
    """returns a mask -> True if point is in transit, False otherwise"""
    return np.abs(_phases(times, period, t0)) < width * (duration_h / 24) / 2


def _spline_knots(t: np.ndarray, spacing: float, min_points: int = 4) -> np.ndarray:
    """
    interior knots every `spacing` days across sorted times t,
    skipping any knot with fewer than min_points data points since the previous one (scipy rejects those
    """
    knots, last = [], t[0]
    for k in np.arange(t[0] + spacing, t[-1], spacing):
        if np.count_nonzero((t > last) & (t <= k)) >= min_points:
            knots.append(k)
            last = k
    if knots and np.count_nonzero(t > knots[-1]) < min_points:
        knots.pop()
    return np.array(knots)


def fit_curve_to_spline(
        c: StarCurve,
        tces: pd.DataFrame,
        spacing: float = 1.5,
        min_gap: float = 0.75,
        sigma: float = 3.0,
        max_iter: int = 5) -> StarCurve:
    """
    divide the curve by a cubic spline fitted to its out-of-transit points

    tces: all of the star's TCE rows
    spacing: days between knots
    min_gap: split into segments at gaps longer than this (days)
    sigma, max_iter: refit after dropping points more than sigma robust stds from the fit
    """
    # build transit mask
    in_transit = np.zeros(len(c.time), dtype=bool)
    for _, tce in tces.iterrows():
        # a point is in transit if mask returns True for any of this star's TCEs
        in_transit |= _transit_mask(c.time, tce.tce_period, tce.tce_time0bk, tce.tce_duration)

    baseline = np.full(len(c.time), np.nan)
    # split curve into segments
    segment_starts = np.where(np.diff(c.time) > min_gap)[0] + 1
    for s in np.split(np.arange(len(c.time)), segment_starts):
        time, flux, flux_err = c.time[s], c.flux[s], c.flux_err[s]
        fit = ~in_transit[s] # 
        if fit.sum() < 8:  # too few points for a cubic spline
            continue
        for _ in range(max_iter):
            # fit spline to segment excluding in-transit points
            spline = LSQUnivariateSpline(time[fit], flux[fit], _spline_knots(time[fit], spacing), w=1 / flux_err[fit], k=3)
            distance_from_fit = flux - spline(time)
            # drop outliers
            robust_std = 1.4826 * np.median(np.abs(distance_from_fit[fit] - np.median(distance_from_fit[fit])))
            new_fit = fit & (np.abs(distance_from_fit) < sigma * robust_std)
            if (new_fit == fit).all() or new_fit.sum() < 8:
                break
            fit = new_fit
        baseline[s] = spline(time)

    keep = np.isfinite(baseline)
    return replace(
        c,
        time=c.time[keep],
        flux=c.flux[keep] / baseline[keep],
        flux_err=c.flux_err[keep] / baseline[keep],
    )


def phase_fold(c: StarCurve, period: float, t0: float) -> FoldedCurve:
    """replace each time with its phase (days from the nearest transit centre, -period/2 to period/2), sorted by phase"""
    phase = _phases(c.time, period, t0)
    order = np.argsort(phase)
    return FoldedCurve(
        kepid=c.kepid,
        phase=phase[order],
        flux=c.flux[order],
        flux_err=c.flux_err[order]
    )


def bin(f: FoldedCurve, num_bins: int, half_width: float) -> np.ndarray:
    """
    divide the flux into equal-width bins and take the median value

    global view: num_bins=2001, half_width=period / 2
    local view: num_bins=201, half_width=min(4 * duration_days, period / 2)
    """
    return binned_statistic(f.phase, f.flux, statistic="median", bins=num_bins, range=(-half_width, half_width)).statistic


def fill_gaps(view: np.ndarray) -> np.ndarray:
    """replace empty (NaN) bins with the view's median"""
    return np.where(np.isnan(view), np.nanmedian(view), view)


def rescale(view: np.ndarray) -> np.ndarray:
    """shift and scale so the baseline is 0 and the deepest point is -1"""
    view = view - np.median(view)
    return view / np.abs(view.min())
