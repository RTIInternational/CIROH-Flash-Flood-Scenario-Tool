"""
Rating curve utilities for the CIROH Flash Flood Scenario Tool.

Provides functions to fetch USGS NWIS stage-discharge rating curves
and convert model discharge output (m³/s) back to stage height (ft).

USGS publishes "extended shift-adjusted" (exsa) rating curves for active
gauges via the NWIS web service.  Each curve is a lookup table mapping
stage (ft) to discharge (cfs); we reverse-interpolate it to go from
model-simulated Q back to the corresponding stage.
"""

import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor

_NWIS_RATING_URL = (
    "https://waterdata.usgs.gov/nwisweb/get_ratings"
    "?file_type=exsa&site_no={site_no}"
)
_FETCH_TIMEOUT = 15    # seconds per request
_FETCH_WORKERS = 10    # parallel HTTP connections

M3S_TO_CFS = 35.3147


def fetch_rating_curve(site_no: str):
    """
    Fetch the current extended shift-adjusted (exsa) rating curve from
    USGS NWIS for one gauge site.

    The exsa format stores columns as STOR (shift applied), INDEP (stage, ft),
    DEP (discharge, ft³/s).  Each stage value is repeated for every historical
    shift period, so the file may contain thousands of rows for one gauge.
    We detect the column order from the header row, filter to non-negative
    discharges, then average discharge values per unique stage to produce a
    clean monotone stage-discharge curve.

    Returns
    -------
    dict with keys:
        'stage_ft'      : np.ndarray – stage values in feet
        'discharge_cfs' : np.ndarray – discharge values in cfs
                          (both arrays sorted by discharge for interpolation)
        'site_no'       : str
    or None if the curve is unavailable or cannot be parsed.
    """
    url = _NWIS_RATING_URL.format(site_no=site_no)
    try:
        resp = requests.get(url, timeout=_FETCH_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        print(f"  [warn] rating curve fetch failed ({site_no}): {exc}")
        return None

    stages, discharges = [], []
    header_seen = False
    indep_col, dep_col = 0, 1   # fallback default (INDEP DEP … column order)

    for line in resp.text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        if not header_seen:
            header_seen = True
            # Detect actual column positions from the header row.
            # The exsa format uses STOR INDEP DEP order; other formats may differ.
            cols = stripped.upper().split()
            try:
                indep_col = cols.index('INDEP')
                dep_col   = cols.index('DEP')
            except ValueError:
                pass   # keep default
            continue
        parts = stripped.split()
        if len(parts) > max(indep_col, dep_col):
            try:
                stages.append(float(parts[indep_col]))
                discharges.append(float(parts[dep_col]))
            except ValueError:
                continue

    if len(stages) < 2:
        print(f"  [warn] no usable rating curve data for {site_no}")
        return None

    stage_arr     = np.asarray(stages,     dtype=np.float64)
    discharge_arr = np.asarray(discharges, dtype=np.float64)

    # Drop rows with negative discharge (artifacts from garbled or missing data)
    valid = discharge_arr >= 0
    stage_arr, discharge_arr = stage_arr[valid], discharge_arr[valid]

    if len(stage_arr) < 2:
        print(f"  [warn] insufficient valid data for {site_no}")
        return None

    # De-duplicate: the exsa format lists each stage once per historical shift
    # period, so the same stage height can appear hundreds of times.  Average
    # the discharge across all shift periods to obtain one value per stage.
    sort_s        = np.argsort(stage_arr)
    stage_arr     = stage_arr[sort_s]
    discharge_arr = discharge_arr[sort_s]
    unique_stages, inv, counts = np.unique(stage_arr, return_inverse=True,
                                           return_counts=True)
    unique_q  = np.bincount(inv, weights=discharge_arr) / counts
    stage_arr = unique_stages
    discharge_arr = unique_q

    if len(discharge_arr) < 2:
        print(f"  [warn] insufficient rating curve data for {site_no}")
        return None

    # Sort by discharge for reverse interpolation (Q → stage)
    sort_q = np.argsort(discharge_arr)
    return {
        'stage_ft':      stage_arr[sort_q],
        'discharge_cfs': discharge_arr[sort_q],
        'site_no':       site_no,
    }


def fetch_all_rating_curves(basin_ids):
    """
    Fetch rating curves for all basin IDs in parallel.

    Parameters
    ----------
    basin_ids : list of str – USGS site numbers

    Returns
    -------
    dict mapping basin_id → rating dict (or None if unavailable)
    """
    print(f"Fetching rating curves for {len(basin_ids)} basins…")

    def _fetch(basin_id):
        return basin_id, fetch_rating_curve(basin_id)

    results = {}
    with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as pool:
        for basin_id, rc in pool.map(_fetch, basin_ids):
            results[basin_id] = rc

    n_ok = sum(1 for v in results.values() if v is not None)
    print(f"  Rating curves loaded: {n_ok}/{len(basin_ids)}")
    return results


def discharge_to_stage(q_m3s, rating):
    """
    Convert a discharge array (m³/s) to stage (ft) by reverse-interpolating
    the supplied rating curve.

    - Within the measured range  : linear interpolation
    - Above the maximum discharge: linear extrapolation using the slope of
      the upper end of the curve (better than clamping to max stage, which
      would make all extreme flows appear identical)
    - Below the minimum discharge: clamped to the minimum stage

    Parameters
    ----------
    q_m3s  : array-like – discharge in m³/s
    rating : dict       – as returned by fetch_rating_curve

    Returns
    -------
    np.ndarray of stage values in feet.
    """
    q_cfs = np.asarray(q_m3s, dtype=np.float64) * M3S_TO_CFS
    q_arr = rating['discharge_cfs']
    h_arr = rating['stage_ft']

    # Interpolate within range; np.interp clamps at the boundaries
    stage = np.interp(q_cfs, q_arr, h_arr)

    # Linear extrapolation above the maximum measured discharge
    if len(q_arr) >= 2:
        dq_top = q_arr[-1] - q_arr[-2]
        if dq_top > 0:
            slope_top = (h_arr[-1] - h_arr[-2]) / dq_top
            above = q_cfs > q_arr[-1]
            if above.any():
                stage[above] = h_arr[-1] + slope_top * (q_cfs[above] - q_arr[-1])

    return stage
