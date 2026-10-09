"""
MRMS QPE Transposition Dashboard

Displays a single MRMS 1-hour QPE file over Hawaii and lets the user
interactively transpose the precipitation grid to a new location.
Optionally overlays Hawaii watershed boundaries from a shapefile and
computes spatial mean QPE inside the selected watershed for both the
original and transposed grids.

Deliberately avoids GeoViews / cartopy — uses plain HoloViews with pyproj
for coordinate conversions and hv.Tiles for the basemap.

Usage:
    panel serve dashboard_mrms.py --show
    panel serve dashboard_mrms.py --show --args path/to/mrms_file.nc path/to/watersheds.shp

Workflow:
    1. Click a cell on the LEFT pane  → sets the SOURCE point  (red cross)
    2. Click a location on the RIGHT pane → sets the DESTINATION point (blue cross)
    3. Click "Transpose" — the grid shifts so source aligns with destination
    4. Click "Reset"    — returns to the original placement
    5. Choose a watershed from the dropdown to see mean QPE stats
"""

import argparse
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

# SAC-SMA runs in a separate subprocess (hydro-model env) — no pysacsnow needed here
_SAC_WORKER     = Path(__file__).parent / "sac_worker.py"
_HYDRO_MODEL_PY = Path(r"C:\Users\domenicbrooks\.conda\envs\hydro-model\python.exe")
# SAC-SMA / UHG base directories
HAWAII_RESULTS_DIR = Path(
    r"C:\Users\domenicbrooks\OneDrive - Research Triangle Institute"
    r"\CIROH Flash Flooding\Hawaii results"
)
HYDRO_MODEL_CONFIG_DIR = Path(
    r"C:\Users\domenicbrooks\OneDrive - Research Triangle Institute"
    r"\CIROH Flash Flooding\hydro-model-config\hawaii\hawaii"
)
_SAC_CTRL_TEMPLATE = HYDRO_MODEL_CONFIG_DIR / "configurations/sac/sac.control.16103000.yaml"
_SAC_AVAILABLE  = (_SAC_WORKER.exists() and _HYDRO_MODEL_PY.exists()
                   and _SAC_CTRL_TEMPLATE.exists())
import numpy as np
import pandas as pd
import xarray as xr
import pyproj
import param
import geopandas as gpd
import panel as pn
import holoviews as hv
from holoviews import streams
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import from_bounds as rio_from_bounds
from bokeh.models import (FixedTicker, CustomJSTickFormatter, HoverTool, CustomJSHover,
                           ColumnDataSource, TapTool, Span)
from bokeh.plotting import figure as bk_figure
from utils import fetch_all_rating_curves, discharge_to_stage

pn.extension("bokeh")
hv.extension("bokeh")

# ---------------------------------------------------------------------------
# Projection helpers  (EPSG:4326 ↔ EPSG:3857 / Web Mercator)
# ---------------------------------------------------------------------------

_to_merc = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
_to_ll   = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

def to_merc(lon, lat):
    """Scalar or array lon/lat (degrees) → Web Mercator (metres)."""
    return _to_merc.transform(lon, lat)

def to_ll(mx, my):
    """Web Mercator (metres) → lon/lat (degrees)."""
    return _to_ll.transform(mx, my)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Colormap — 15-bin discrete radar-style palette
# ---------------------------------------------------------------------------

PRECIP_BIN_EDGES = [
    0.25, 0.5, 1.0, 2.5, 5.0, 7.5,
    10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0,
] # mm

PRECIP_BIN_EDGES_inches = [
    0.01, 0.02, 0.04, 0.10, 0.20, 0.40,
    0.60, 0.80, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00
] # inches

PRECIP_PALETTE = [
    "#b4f0a7",  # 0.25 – 0.5    pale green
    "#66e666",  # 0.50 – 1.0    light green
    "#00c800",  # 1.0  – 2.5    green
    "#006600",  # 2.5  – 5.0    dark green
    "#c8ff00",  # 5.0  – 7.5    yellow-green
    "#e6e600",  # 7.5  – 10.0   yellow
    "#ffcc00",  # 10.0 – 15.0   gold
    "#ff9900",  # 15.0 – 20.0   orange
    "#ff6600",  # 20.0 – 30.0   dark orange
    "#ff0000",  # 30.0 – 40.0   red
    "#cc0000",  # 40.0 – 50.0   dark red
    "#990000",  # 50.0 – 75.0   very dark red
    "#ff00ff",  # 75.0 – 100.0  magenta
    "#cc00cc",  # 100.0+         dark magenta / purple
]

# Long-duration (> 6 hr) scale: same palette, but stretched to cover the
# larger accumulated totals seen in 12/24/48-hr periods (10–300 mm).
PRECIP_BIN_EDGES_LONG = [
    10.0, 20.0, 30.0, 50.0, 75.0, 100.0,
    125.0, 150.0, 175.0, 200.0, 225.0, 250.0, 275.0, 300.0,
] # mm

PRECIP_BIN_EDGES_LONG_INCHES = [
    0.4, 0.8, 1.2, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
] # inches

# 256-entry staircase palette: each entry maps a linearly-sampled mm value
# to its bin colour.  Passing this to hv.Image with clim=(edge[0], edge[-1])
# gives truly discrete colour steps at the correct mm thresholds.
_STEPPED_PALETTE = []
for _v in np.linspace(PRECIP_BIN_EDGES[0], PRECIP_BIN_EDGES[-1], 256):
    _k = min(int(np.searchsorted(PRECIP_BIN_EDGES, _v, side="right")) - 1,
             len(PRECIP_PALETTE) - 1)
    _STEPPED_PALETTE.append(PRECIP_PALETTE[max(_k, 0)])
del _v, _k

# Pre-digitized approach: each bin → integer index 1…14, equal height on
# the colorbar.  Colorbar ticks are placed at bin boundaries (i + 0.5) and
# labelled with actual mm values.
_N_BINS = len(PRECIP_PALETTE)


def _digitize_precip(data, bin_edges=PRECIP_BIN_EDGES):
    """
    Map raw QPE values to integer bin indices 1…_N_BINS using bin_edges.
    Dry / sub-threshold pixels (< first edge) become NaN.
    """
    arr = data.copy().astype(np.float32)
    dry = ~np.isfinite(arr) | (arr < bin_edges[0])
    binned = np.digitize(arr, bin_edges).astype(np.float32)
    binned[dry] = np.nan
    return binned


def _fmt_edge(v):
    return str(int(v)) if v == int(v) else str(v)


# Colorbar boundary ticks (shared: both scales have the same bin count)
_CB_TICKS  = [i + 0.5 for i in range(_N_BINS + 1)]
_CB_LABELS = (
    [_fmt_edge(PRECIP_BIN_EDGES[0])]
    + [_fmt_edge(e) for e in PRECIP_BIN_EDGES[1:]]
    + ["\u2265100"]
)
_CB_LABELS_IN = (
    [_fmt_edge(PRECIP_BIN_EDGES_inches[0])]
    + [_fmt_edge(e) for e in PRECIP_BIN_EDGES_inches[1:]]
    + ["\u22655"]
)
_CB_LABELS_LONG = (
    [_fmt_edge(PRECIP_BIN_EDGES_LONG[0])]
    + [_fmt_edge(e) for e in PRECIP_BIN_EDGES_LONG[1:]]
    + ["\u2265300"]
)
_CB_LABELS_LONG_IN = (
    [_fmt_edge(PRECIP_BIN_EDGES_LONG_INCHES[0])]
    + [_fmt_edge(e) for e in PRECIP_BIN_EDGES_LONG_INCHES[1:]]
    + ["\u226512"]
)

# Average Recurrence Interval (ARI) display scale — shows how rare the
# observed precip is per Atlas14 climatology, independent of the
# transposition method in use.  Reuses PRECIP_PALETTE (same bin count).
ARI_BIN_EDGES = [1, 2, 3, 4, 6, 8, 10, 20, 30, 40, 50, 75, 100, 200]  # years
_CB_LABELS_ARI = (
    [_fmt_edge(ARI_BIN_EDGES[0])]
    + [_fmt_edge(e) for e in ARI_BIN_EDGES[1:]]
    + ["\u2265200"]
)

PLOT_W      = 620
PLOT_H      = 500
DEFAULT_NC  = (
    r"C:\Users\domenicbrooks\OneDrive - Research Triangle Institute"
    r"\CIROH Flash Flooding\data\MRMS\mrms_qpe_1hr_hawaii_20260320_1200.nc"
)
DEFAULT_PARQUET = str(
    HAWAII_RESULTS_DIR / "usgs_basin_geometry_with_source_all.parquet"
)
WATERSHED_NAME_COL = "name"

# Basin IDs that have been calibrated (derived from output subfolder names)
try:
    _AVAILABLE_BASIN_IDS = sorted([
        d.replace("basin_", "")
        for d in os.listdir(HAWAII_RESULTS_DIR / "output")
        if d.startswith("basin_")
        and (HAWAII_RESULTS_DIR / "output" / d).is_dir()
    ])
except OSError:
    _AVAILABLE_BASIN_IDS = []

DEFAULT_COASTLINE_SHP = (
    r"C:\Users\domenicbrooks\OneDrive - Research Triangle Institute"
    r"\CIROH Flash Flooding\data\GIS\Coastline.shp"
)
DEFAULT_GAUGE_PARQUET = str(
    HAWAII_RESULTS_DIR / "usgs_lt500_min2014-2024.parquet"
)

# Atlas 14 precipitation-frequency grid, used for AEP storm transposition.
# Fixed, locally-downloaded file — no on-the-fly retrieval.
DEFAULT_ATLAS14_NC = r"D:\Capstone Project\Precip_Grids\Atlas_14\a14_hawaii.nc"

# Dashboard "Period" selector label → matching Atlas14 duration
_ATLAS14_DURATION_MAP = {
    "1 hr":  pd.Timedelta(hours=1),
    "3 hr":  pd.Timedelta(hours=3),
    "6 hr":  pd.Timedelta(hours=6),
    "12 hr": pd.Timedelta(hours=12),
    "24 hr": pd.Timedelta(hours=24),
    "48 hr": pd.Timedelta(hours=48),
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# HDF5/NetCDF4 C library is not thread-safe; serialize open+copy across threads.
_NC_LOCK = threading.Lock()


def load_mrms(path: str):
    """
    Load the first variable from a MRMS NetCDF file.

    Returns
    -------
    lons : 1-D array, longitudes in –180…+180 degrees
    lats : 1-D array, latitudes  ascending
    data : 2-D array (lat × lon), QPE in mm; missing values → NaN
    """
    with _NC_LOCK:
        ds   = xr.open_dataset(path)
        da   = ds[list(ds.data_vars)[0]]

        lats = da["latitude"].values.copy()
        lons = ((da["longitude"].values + 180.0) % 360.0) - 180.0   # 0-360 → ±180
        data = da.values.copy().astype(float)

        missing = da.attrs.get("GRIB_missingValue", 9.999e+20)

        # Extract valid time (first time step, or fall back to file mtime)
        try:
            t_vals = da["time"].values if "time" in da.dims else da["valid_time"].values
            valid_time = pd.Timestamp(t_vals.flat[0]).strftime("%Y-%m-%d %Hz")
        except Exception:
            valid_time = ""

        ds.close()

    data[(data < 0) | (data >= missing * 0.99)] = np.nan

    if lats[0] > lats[-1]:          # ensure ascending so row-0 == lat_min
        lats = lats[::-1]
        data = data[::-1, :]

    return lons, lats, data, valid_time


def find_mrms_files(data_dir: str, end_dt: datetime, n_hours: int):
    """
    Find MRMS 1-hr QPE files covering the *n_hours* ending at *end_dt* (inclusive).
    Filenames are expected to match mrms_qpe_1hr_hawaii_YYYYMMDD_HHMM.nc.
    Returns a list of (datetime, filepath) sorted oldest-first; missing hours are
    skipped with a printed warning.
    """
    results = []
    for h in range(n_hours - 1, -1, -1):
        dt = end_dt - timedelta(hours=h)
        fname = f"mrms_qpe_1hr_hawaii_{dt.strftime('%Y%m%d_%H%M')}.nc"
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            results.append((dt, fpath))
        else:
            print(f"  [warn] missing: {fname}")
    return results


def point_value_at(lons, lats, data, target_lon, target_lat):
    """
    Extract QPE (mm) at the nearest grid cell to (target_lon, target_lat).
    Returns 0.0 if coordinates are None or the cell value is NaN.
    """
    if target_lon is None or target_lat is None:
        return 0.0
    j = int(np.argmin(np.abs(lons - target_lon)))
    i = int(np.argmin(np.abs(lats - target_lat)))
    v = data[i, j]
    return float(v) if np.isfinite(v) else 0.0


# ---------------------------------------------------------------------------
# Watershed helpers
# ---------------------------------------------------------------------------

def load_watersheds(path: str):
    """
    Load watershed geometry from a parquet or shapefile and return two GeoDataFrames:
      - gdf_4326 : reprojected to EPSG:4326 (for masking)
      - gdf_3857 : reprojected to EPSG:3857 (for map display)
    When loading a parquet file the result is filtered to only the basins
    that have calibrated SAC output (i.e. entries in _AVAILABLE_BASIN_IDS).
    """
    ext = Path(path).suffix.lower()
    if ext == ".parquet":
        gdf = gpd.read_parquet(path)
        available_ids = {"usgs-" + b for b in _AVAILABLE_BASIN_IDS}
        gdf = gdf[gdf["id"].isin(available_ids)].reset_index(drop=True)
    else:
        gdf = gpd.read_file(path)
    gdf_4326 = gdf.to_crs("EPSG:4326")
    gdf_3857 = gdf.to_crs("EPSG:3857")
    return gdf_4326, gdf_3857


def build_coastline_overlay(shp_path: str):
    """
    Load coastline shapefile, reproject to EPSG:3857, and return a faint
    outline hv.Polygons element for map display.
    """
    gdf = gpd.read_file(shp_path).to_crs("EPSG:3857")
    all_paths = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        all_paths.extend(_geom_to_paths(geom))
    return hv.Polygons(all_paths, kdims=["x", "y"]).opts(
        fill_alpha=0.0,
        line_color="#222222",
        line_alpha=0.9,
        line_width=1.5,
    )


def load_gauge_gdf(parquet_path: str):
    """
    Load gauge location parquet, filter to calibrated basins, and return
    a GeoDataFrame in EPSG:3857 with a stripped 'gauge_id' column.
    """
    gdf = gpd.read_parquet(parquet_path)
    available_ids = {"usgs-" + b for b in _AVAILABLE_BASIN_IDS}
    gdf = gdf[gdf["id"].isin(available_ids)].copy()
    gdf["gauge_id"] = gdf["id"].str.replace("usgs-", "", regex=False)
    return gdf.to_crs("EPSG:3857")


def build_gauge_points(gdf_3857):
    """
    Build an hv.Points element for gauge locations with hover tooltip
    showing gauge ID and stream name.
    """
    data = {
        "x":    gdf_3857.geometry.x.values,
        "y":    gdf_3857.geometry.y.values,
        "name": gdf_3857["name"].values,
        "id":   gdf_3857["gauge_id"].values,
    }
    return hv.Points(data, kdims=["x", "y"], vdims=["name", "id"], label="USGS Gauges").opts(
        size=8,
        color="#b6b6b6",
        marker="circle",
        line_color="black",
        line_width=0.8,
        tools=["hover"],
    )


def _geom_to_paths(geom):
    """
    Convert a shapely Polygon or MultiPolygon exterior to a list of
    dicts suitable for hv.Polygons: [{'x': array, 'y': array}, ...].
    """
    from shapely.geometry import MultiPolygon
    geoms = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    paths = []
    for poly in geoms:
        xs, ys = poly.exterior.xy
        paths.append({"x": np.asarray(xs), "y": np.asarray(ys)})
    return paths


def build_all_watershed_polygons(gdf_3857):
    """
    Build a single faint hv.Polygons element containing all watersheds.
    Precomputed once at startup.
    """
    all_paths = []
    for geom in gdf_3857.geometry:
        if geom is None or geom.is_empty:
            continue
        all_paths.extend(_geom_to_paths(geom))
    return hv.Polygons(all_paths, kdims=["x", "y"]).opts(
        fill_alpha=0.0,
        line_color="#383838",
        line_alpha=0.9,
        line_width=1.0,
    )


def build_selected_watershed_polygon(geom_3857):
    """
    Build a highlighted hv.Polygons element for the selected watershed.
    Returns an empty element if geom is None.
    """
    if geom_3857 is None:
        return hv.Polygons([], kdims=["x", "y"]).opts(
            fill_alpha=0.0, line_color="blue", line_width=2.5
        )
    paths = _geom_to_paths(geom_3857)
    return hv.Polygons(paths, kdims=["x", "y"]).opts(
        fill_alpha=0.15,
        fill_color="blue",
        line_color="blue",
        line_width=2.5,
    )


def compute_watershed_mask(geom_4326, lons, lats):
    """
    Rasterize a shapely geometry (EPSG:4326) onto the MRMS grid.

    Returns a boolean array with shape (len(lats), len(lons)) where True
    indicates the cell centre is inside the watershed.  Row-0 = lat_min
    (ascending, matching the data array orientation).
    """
    if geom_4326 is None:
        return None

    nlat, nlon = len(lats), len(lons)

    # rio_from_bounds treats the given extents as the OUTER EDGES of the raster,
    # but lons/lats hold cell CENTRES.  Extend bounds by half a cell so that
    # each cell's centre falls at the correct rasterio pixel position.
    dlon_h = float(lons[1] - lons[0]) / 2.0 if nlon > 1 else 0.0025
    dlat_h = float(lats[1] - lats[0]) / 2.0 if nlat > 1 else 0.0025

    # rasterio's from_bounds creates a north-up transform (row-0 = north)
    transform = rio_from_bounds(
        west=float(lons.min())  - dlon_h,
        south=float(lats.min()) - dlat_h,
        east=float(lons.max())  + dlon_h,
        north=float(lats.max()) + dlat_h,
        width=nlon, height=nlat,
    )

    from shapely.geometry import mapping
    mask = rio_rasterize(
        [(mapping(geom_4326), 1)],
        out_shape=(nlat, nlon),
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    # rasterio rasterizes north-up (row-0 = north), but our data is south-up
    # (row-0 = lat_min).  Flip to align with the data array.
    return np.flipud(mask).astype(bool)


def spatial_mean(data, mask):
    """
    Return the mean QPE (mm) within the masked area, ignoring NaN cells.
    Returns None if the mask is None or contains no valid data.
    """
    if mask is None:
        return None
    if data.shape != mask.shape:
        return None   # grid resolution mismatch — skip this timestep
    values = data[mask]
    valid = values[~np.isnan(values)]
    return float(np.mean(valid)) if valid.size > 0 else None


# ---------------------------------------------------------------------------
# Atlas 14 / AEP (Annual Exceedance Probability) helpers
# ---------------------------------------------------------------------------

def load_atlas14(path: str):
    """
    Open the Atlas 14 precipitation-frequency grid (lazy) for AEP transposition.

    Returns
    -------
    ds        : xr.Dataset, dims (ARI, confidence, duration, reg, y, x)
    lons      : 1-D ascending longitude coords (Atlas14 'x')
    lats      : 1-D ascending latitude coords  (Atlas14 'y')
    land_mask : 2-D bool array (y, x); True where Atlas14 has valid (land) coverage.
                Assumed the same for every ARI/duration (verified: physical mask).
    """
    ds = xr.open_dataset(path)
    lons = ds["x"].values
    lats = ds["y"].values
    ref = ds["precip"].isel(ARI=0, confidence=0, duration=0, reg=0).values
    land_mask = np.isfinite(ref)
    return ds, lons, lats, land_mask


def _nearest_indices(coord_array, query_array):
    """Vectorised nearest-neighbour indices of query_array within a uniformly-spaced coord_array."""
    d = coord_array[1] - coord_array[0]
    idx = np.round((query_array - coord_array[0]) / d).astype(np.intp)
    return np.clip(idx, 0, len(coord_array) - 1)


def _invert_to_log_ari(ref, amounts, log_ari):
    """
    Given observed values `ref` and per-cell climatology curves `amounts`
    (shape (ARI, ...) matching ref broadcastable), return the equivalent
    log(ARI) via linear interpolation between curve points, log-linearly
    extrapolated beyond the table's ends.
    """
    log_ari_field = np.zeros(ref.shape, dtype=np.float32)
    filled = np.zeros(ref.shape, dtype=bool)
    for k in range(len(log_ari) - 1):
        y0, y1 = amounts[k], amounts[k + 1]
        x0, x1 = log_ari[k], log_ari[k + 1]
        seg = (ref >= y0) & (ref <= y1) & ~filled
        denom = np.where(y1 != y0, y1 - y0, 1.0)
        log_ari_field = np.where(seg, x0 + (ref - y0) / denom * (x1 - x0), log_ari_field)
        filled |= seg
    below = ref <= amounts[0]
    y0, y1, x0, x1 = amounts[0], amounts[1], log_ari[0], log_ari[1]
    denom = np.where(y1 != y0, y1 - y0, 1.0)
    log_ari_field = np.where(below, x0 + (ref - y0) / denom * (x1 - x0), log_ari_field)
    above = ref >= amounts[-1]
    y0, y1, x0, x1 = amounts[-2], amounts[-1], log_ari[-2], log_ari[-1]
    denom = np.where(y1 != y0, y1 - y0, 1.0)
    log_ari_field = np.where(above, x0 + (ref - y0) / denom * (x1 - x0), log_ari_field)
    return log_ari_field


def _forward_from_log_ari(log_ari_field, amounts, log_ari):
    """Inverse of _invert_to_log_ari: evaluate climatology curves `amounts` at given log(ARI) positions."""
    v_new = np.zeros(log_ari_field.shape, dtype=np.float32)
    filled = np.zeros(log_ari_field.shape, dtype=bool)
    for k in range(len(log_ari) - 1):
        x0, x1 = log_ari[k], log_ari[k + 1]
        y0, y1 = amounts[k], amounts[k + 1]
        seg = (log_ari_field >= x0) & (log_ari_field <= x1) & ~filled
        v_new = np.where(seg, y0 + (log_ari_field - x0) / (x1 - x0) * (y1 - y0), v_new)
        filled |= seg
    below = log_ari_field <= log_ari[0]
    x0, x1, y0, y1 = log_ari[0], log_ari[1], amounts[0], amounts[1]
    v_new = np.where(below, y0 + (log_ari_field - x0) / (x1 - x0) * (y1 - y0), v_new)
    above = log_ari_field >= log_ari[-1]
    x0, x1, y0, y1 = log_ari[-2], log_ari[-1], amounts[-2], amounts[-1]
    v_new = np.where(above, y0 + (log_ari_field - x0) / (x1 - x0) * (y1 - y0), v_new)
    return v_new


def redistribute_ts(values, idx, new_val, method="Proportional", sigma=3.0, window=3):
    """
    Change values[idx] to new_val and redistribute the difference across
    other timesteps so the total sum is preserved.

    Parameters
    ----------
    values : array-like  – per-hour values (base snapshot at time of selection)
    idx    : int         – index of the timestep to modify
    new_val: float       – desired new value for that timestep
    method : str         – "Proportional", "Gaussian", or "Window"
    sigma  : float       – Gaussian half-width in hours (Gaussian method only)
    window : int         – half-window in hours (Window method only)

    Returns
    -------
    new array with the same sum as the input and result[idx] == new_val.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    total = float(values.sum())

    # Clamp new_val so total can still be preserved with non-negative others
    new_val = float(np.clip(new_val, 0.0, total))

    # Build unnormalized weights for compensation
    idx_arr = np.arange(n)
    if method == "Proportional":
        weights = np.where(idx_arr != idx, np.maximum(values, 0.0), 0.0)
    elif method == "Gaussian":
        gauss = np.exp(-0.5 * ((idx_arr - idx) / float(sigma)) ** 2)
        weights = np.where(idx_arr != idx, np.maximum(values, 0.0) * gauss, 0.0)
    else:  # "Window"
        in_win = (idx_arr != idx) & (np.abs(idx_arr - idx) <= int(window))
        weights = np.where(in_win, np.maximum(values, 0.0), 0.0)

    wsum = weights.sum()
    if wsum <= 0.0:
        # Fallback: uniform across all other hours
        weights = np.where(idx_arr != idx, 1.0, 0.0)
        wsum = float(weights.sum())
        if wsum <= 0.0:
            result = values.copy()
            result[idx] = new_val
            return result

    result = np.zeros(n)
    result[idx] = new_val
    # Remaining total distributed proportionally via weights
    result += weights / wsum * (total - new_val)
    return result


# ---------------------------------------------------------------------------
# Transpose logic
# ---------------------------------------------------------------------------

def transpose_data(lons, lats, data, d_lon, d_lat):
    """
    Shift the precipitation field by (d_lon, d_lat) degrees.
    The entire data array is preserved — only the coordinate arrays are shifted,
    so the whole storm pattern moves to the new location without any clipping.
    """
    if d_lon == 0.0 and d_lat == 0.0:
        return lons, lats, data
    return lons + d_lon, lats + d_lat, data


def rotate_data(lons, lats, data, angle_deg, pivot_lon=None, pivot_lat=None):
    """
    Rotate the precipitation grid by angle_deg degrees clockwise (as viewed
    on a north-up map) around pivot_lon/pivot_lat (degrees).  If no pivot is
    provided the grid centre is used.

    Strategy: pad the array asymmetrically so the pivot lies exactly at the
    array centre, then call scipy.ndimage.rotate (which always rotates around
    the array centre).  The returned coordinate arrays are built so that the
    pivot sits at their centre, preserving its geographic position.
    """
    if angle_deg == 0.0:
        return lons, lats, data

    from scipy.ndimage import rotate as ndimage_rotate

    H, W = data.shape
    dlat = (float(lats.max()) - float(lats.min())) / max(H - 1, 1)
    dlon = (float(lons.max()) - float(lons.min())) / max(W - 1, 1)

    # Default pivot = grid centre
    if pivot_lon is None:
        pivot_lon = 0.5 * (float(lons.min()) + float(lons.max()))
    if pivot_lat is None:
        pivot_lat = 0.5 * (float(lats.min()) + float(lats.max()))

    # Fractional pixel position of pivot (row 0 = south/lat_min)
    col_piv = (pivot_lon - float(lons.min())) / dlon
    row_piv = (pivot_lat - float(lats.min())) / dlat

    # Distance from pivot to each edge (pixels)
    d_south = row_piv
    d_north = (H - 1) - row_piv
    d_west  = col_piv
    d_east  = (W - 1) - col_piv

    # Pad each side so pivot ends up at the array centre.
    # row 0 = south → pad_south prepended at [0], pad_north appended at end.
    pad_south = max(0, int(np.ceil(d_north - d_south)))
    pad_north = max(0, int(np.ceil(d_south - d_north)))
    pad_west  = max(0, int(np.ceil(d_east  - d_west)))
    pad_east  = max(0, int(np.ceil(d_west  - d_east)))

    padded = np.pad(
        np.where(np.isnan(data), 0.0, data),
        ((pad_south, pad_north), (pad_west, pad_east)),
        constant_values=0.0,
    )
    valid_mask = np.pad(
        (~np.isnan(data)).astype(float),
        ((pad_south, pad_north), (pad_west, pad_east)),
        constant_values=0.0,
    )

    # scipy CCW positive; our south-up array means CCW pixel = CW on map.
    rotated   = ndimage_rotate(padded,     angle_deg, reshape=True, cval=0.0, order=1)
    rot_valid = ndimage_rotate(valid_mask, angle_deg, reshape=True, cval=0.0, order=1)
    rotated   = np.where(rot_valid < 0.5, np.nan, rotated)

    H_new, W_new = rotated.shape

    # Pivot is at the centre of the rotated (reshaped) array.
    new_lats = np.linspace(
        pivot_lat - H_new * dlat / 2.0,
        pivot_lat + H_new * dlat / 2.0,
        H_new,
    )
    new_lons = np.linspace(
        pivot_lon - W_new * dlon / 2.0,
        pivot_lon + W_new * dlon / 2.0,
        W_new,
    )

    # Crop back to the original (pre-rotation) extent with a small buffer so
    # that hv.Image maps pixels over a small lat/lon range.  hv.Image maps
    # pixels *linearly in Mercator y*, but our coordinates are linear in
    # geographic degrees.  Over a large lat extent this causes a significant
    # north/south pixel shift.  Keeping extents small (≤ original grid size)
    # keeps the error well below one grid-cell.
    margin = max(dlat, dlon) * 2
    col_lo = max(0, int(np.searchsorted(new_lons, float(lons.min()) - margin)))
    col_hi = min(W_new, int(np.searchsorted(new_lons, float(lons.max()) + margin)) + 1)
    row_lo = max(0, int(np.searchsorted(new_lats, float(lats.min()) - margin)))
    row_hi = min(H_new, int(np.searchsorted(new_lats, float(lats.max()) + margin)) + 1)

    return (new_lons[col_lo:col_hi],
            new_lats[row_lo:row_hi],
            rotated[row_lo:row_hi, col_lo:col_hi])


def regrid_to_mercator_lats(lons, lats, data):
    """
    Resample data rows onto Mercator-uniform latitudes.

    hv.Image renders image rows LINEARLY in Mercator y-space, but our grids
    are uniformly spaced in geographic degrees.  At ~20°N over a 5° extent
    this causes a systematic ~0.07° northward pixel shift: heavy-rain cells
    near the southern edge of a watershed appear visually inside it, while
    the geographic mask correctly places them at the edge, causing a
    mismatch between the displayed colours and the computed areal mean.

    This function resamples the data so each row corresponds to the
    geographic latitude of an equally-spaced Mercator y value.  After this,
    hv.Image renders each row at exactly the right latitude and the mask
    is consistent with what appears on screen.
    """
    H, W = data.shape
    if H < 2:
        return lons, lats, data

    # Mercator y at south and north edges of the grid
    R = 6378137.0
    lat_min_rad = np.deg2rad(float(lats[0]))
    lat_max_rad = np.deg2rad(float(lats[-1]))
    y_south = R * np.log(np.tan(np.pi / 4.0 + lat_min_rad / 2.0))
    y_north = R * np.log(np.tan(np.pi / 4.0 + lat_max_rad / 2.0))

    # H equally-spaced Mercator y values → their geographic latitudes
    merc_ys  = np.linspace(y_south, y_north, H)
    merc_lats = np.degrees(2.0 * np.arctan(np.exp(merc_ys / R)) - np.pi / 2.0)

    # Fractional row indices in the original (degree-uniform) grid
    lat_min = float(lats[0])
    lat_range = float(lats[-1]) - lat_min
    src_rows = np.clip((merc_lats - lat_min) / lat_range * (H - 1), 0, H - 1)

    # Fast vectorised bilinear interpolation in the latitude direction only
    row_lo  = np.clip(np.floor(src_rows).astype(np.intp), 0, H - 2)
    row_hi  = row_lo + 1
    frac    = (src_rows - row_lo)[:, np.newaxis]   # shape (H, 1)

    data_clean = np.where(np.isnan(data), 0.0, data)
    resampled  = data_clean[row_lo, :] * (1.0 - frac) + data_clean[row_hi, :] * frac

    # NaN wherever either source row was NaN
    nan_mask  = np.isnan(data)
    resampled = np.where(nan_mask[row_lo, :] | nan_mask[row_hi, :], np.nan, resampled)

    return lons, merc_lats, resampled

# ---------------------------------------------------------------------------
# HoloViews element builders  (everything in Web Mercator)
# ---------------------------------------------------------------------------

def _merc_bounds(lons, lats):
    """Return (x0, y0, x1, y1) of the data extent in Web Mercator."""
    x0, y0 = to_merc(lons.min(), lats.min())
    x1, y1 = to_merc(lons.max(), lats.max())
    return x0, y0, x1, y1


def make_tiles():
    """CartoLight basemap tile element."""
    return hv.element.tiles.CartoLight().opts(width=PLOT_W, height=PLOT_H)


def make_image(lons, lats, data, _already_merc=False, use_imperial=False, long_duration=False,
               ari_mode=False):
    """
    Build an hv.Image in Web Mercator coordinates using the discrete
    radar-style colormap.  Each bin occupies equal height on the colorbar;
    tick labels show actual mm thresholds.  Dry pixels are transparent.
    Hover shows lat/lon and the actual mm value from a second vdim.

    long_duration=True switches to the 10–300 mm scale used for periods
    longer than 6 hours; otherwise the default 0.25–100 mm scale is used.
    ari_mode=True treats `data` as an Atlas14 average recurrence interval
    (years) field instead of raw QPE, using a fixed 1–200 yr scale.
    """
    if not _already_merc:
        lons, lats, data = regrid_to_mercator_lats(lons, lats, data)
    x0, y0, x1, y1 = _merc_bounds(lons, lats)
    # Keep south-up orientation so row 0 aligns with ys[0] = y0 (south)
    south_up = data.astype(np.float32)

    if ari_mode:
        bin_edges   = ARI_BIN_EDGES
        binned      = _digitize_precip(south_up, bin_edges)
        raw_display = south_up.copy()
        raw_display[~np.isfinite(raw_display) | (raw_display < bin_edges[0])] = np.nan
        disp_unit   = "yr"
        cb_labels   = _CB_LABELS_ARI
        hover_fmt   = "0.1f"
        cb_title    = "ARI (yr)"
        hover_label = "ARI"
    else:
        bin_edges = PRECIP_BIN_EDGES_LONG if long_duration else PRECIP_BIN_EDGES
        binned = _digitize_precip(south_up, bin_edges)
        raw_mm  = south_up.copy()
        raw_mm[~np.isfinite(raw_mm) | (raw_mm < bin_edges[0])] = np.nan

        disp_unit   = "in" if use_imperial else "mm"
        raw_display = raw_mm * 0.0393701 if use_imperial else raw_mm
        if long_duration:
            cb_labels = _CB_LABELS_LONG_IN if use_imperial else _CB_LABELS_LONG
        else:
            cb_labels = _CB_LABELS_IN if use_imperial else _CB_LABELS
        hover_fmt   = "0.3f" if use_imperial else "0.2f"
        cb_title    = f"QPE ({disp_unit})"
        hover_label = "QPE"

    ny, nx = binned.shape
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)  # south to north, matches south-up rows
    ds = xr.Dataset(
        {
            "Bin":     xr.DataArray(binned, dims=["y", "x"]),
            "QPE_val": xr.DataArray(raw_display, dims=["y", "x"]),
        },
        coords={"x": xs, "y": ys},
    )

    ticker = FixedTicker(ticks=_CB_TICKS)
    formatter = CustomJSTickFormatter(
        args=dict(labels=cb_labels),
        code="return labels[Math.round(tick - 0.5)] || '';",
    )
    # Hover: convert Web Mercator x/y to lat/lon
    _lat_fmt = CustomJSHover(code="""
        const lat = (Math.atan(Math.exp(special_vars.y * Math.PI / 20037508.342))
                     * 360 / Math.PI) - 90;
        return lat.toFixed(4) + '\u00b0N';
    """)
    _lon_fmt = CustomJSHover(code="""
        const lon = (special_vars.x / 20037508.342) * 180;
        return lon.toFixed(4) + '\u00b0E';
    """)
    hover_tool = HoverTool(
        tooltips=[
            ("Lat", "$y{custom}"),
            ("Lon", "$x{custom}"),
            (hover_label, f"@QPE_val{{{hover_fmt}}} {disp_unit}"),
        ],
        formatters={
            "$y": _lat_fmt,
            "$x": _lon_fmt,
        },
    )
    return hv.Image(
        ds,
        kdims=["x", "y"],
        vdims=["Bin", "QPE_val"],
    ).opts(
        cmap=PRECIP_PALETTE,
        clim=(0.5, _N_BINS + 0.5),
        alpha=0.65,
        colorbar=True,
        colorbar_opts={
            "title": cb_title,
            "ticker": ticker,
            "formatter": formatter,
            "major_tick_line_color": "black",
            "major_tick_line_width": 1,
            "major_tick_in": 0,
            "major_tick_out": 5,
        },
        tools=["tap", hover_tool],
    )


def make_marker(mx=None, my=None, color="black"):
    """Cross marker at a Web Mercator position, or empty if not yet set."""
    pts = [(mx, my)] if mx is not None else []
    return hv.Points(pts, kdims=["Easting (m)", "Northing (m)"]).opts(
        size=14, color=color, marker="cross", line_width=3,
    )

# ---------------------------------------------------------------------------
# Dashboard class
# ---------------------------------------------------------------------------

class MRMSTransposeDashboard(param.Parameterized):
    """Interactive dashboard for MRMS QPE transposition with watershed stats."""

    selected_watershed = param.String(default="")
    rotation_deg       = param.Number(default=0.0)
    fine_lon           = param.Number(default=0.0)
    fine_lat           = param.Number(default=0.0)
    data_version       = param.Integer(default=0)

    def __init__(self, nc_path: str, shp_path: str = None,
                 coastline_path: str = DEFAULT_COASTLINE_SHP,
                 gauge_path: str = DEFAULT_GAUGE_PARQUET,
                 atlas14_path: str = DEFAULT_ATLAS14_NC, **params):
        super().__init__(**params)
        self.nc_path  = nc_path
        self.shp_path = shp_path
        self.lons, self.lats, self.data, self._valid_time = load_mrms(nc_path)

        # Click positions stored in Web Mercator (what Tap returns)
        self._src_mx = self._src_my = None
        self._dst_mx = self._dst_my = None

        # Applied transposition in degrees
        self._d_lon = self._d_lat = 0.0

        # ── Watershed data ────────────────────────────────────────────────
        self._has_watersheds = False
        self._gdf_4326 = self._gdf_3857 = None
        self._all_ws_polys = hv.Polygons([], kdims=["x", "y"])
        self._ws_mask    = None   # boolean array: watershed on original grid
        self._trans_mean = None   # cached transposed mean QPE (updated by _recompute_trans_mean)

        if shp_path:
            print("Loading watersheds…")
            self._gdf_4326, self._gdf_3857 = load_watersheds(shp_path)
            self._all_ws_polys = build_all_watershed_polygons(self._gdf_3857)
            self._has_watersheds = True
            print(f"  Loaded {len(self._gdf_4326)} watersheds")

        # Build name → gauge ID mapping for SAC basin selection
        self._basin_id_for_name = {}
        if self._has_watersheds and "id" in self._gdf_4326.columns:
            for _, row in self._gdf_4326.iterrows():
                self._basin_id_for_name[row[WATERSHED_NAME_COL]] = (
                    row["id"].replace("usgs-", ""))

        # ── Coastline overlay ─────────────────────────────────────────────────────
        self._coastline_overlay = hv.Polygons([], kdims=["x", "y"])
        if coastline_path:
            try:
                self._coastline_overlay = build_coastline_overlay(coastline_path)
                print("  Loaded coastline")
            except Exception as _e:
                print(f"  [warn] coastline not loaded: {_e}")

        # ── Gauge points overlay ──────────────────────────────────────────────────
        self._gauge_gdf_3857 = None
        self._gauge_points   = hv.Points([], kdims=["x", "y"])
        if gauge_path:
            try:
                self._gauge_gdf_3857 = load_gauge_gdf(gauge_path)
                self._gauge_points   = build_gauge_points(self._gauge_gdf_3857)
                print(f"  Loaded {len(self._gauge_gdf_3857)} gauge locations")
            except Exception as _e:
                print(f"  [warn] gauges not loaded: {_e}")

        # ── Atlas 14 climatology (for AEP transposition) ───────────────────────────
        self._has_atlas14 = False
        self._a14_ds = self._a14_lons = self._a14_lats = self._a14_land_mask = None
        if atlas14_path:
            try:
                self._a14_ds, self._a14_lons, self._a14_lats, self._a14_land_mask = \
                    load_atlas14(atlas14_path)
                self._has_atlas14 = True
                print(f"  Loaded Atlas 14 climatology ({atlas14_path})")
            except Exception as _e:
                print(f"  [warn] Atlas 14 not loaded: {_e}")

        # AEP (Annual Exceedance Probability) transposition state
        self._aep_enabled        = False
        self._aep_field          = None   # 2-D array, same shape as self.data
        self._aep_version        = 0      # bumped each recompute; used in cache keys
        self._a14_duration_cache = {}     # {pd.Timedelta: (precip_allari, ari_values)}

        # ── Watershed stream (triggers redraw on dropdown/rotation/fine-tune changes) ──
        # Split: orig map only needs watershed changes; trans map needs all params.
        self._orig_stream  = streams.Params(self, ["selected_watershed", "data_version"])
        self._trans_stream = streams.Params(
            self, ["selected_watershed", "rotation_deg", "fine_lon", "fine_lat", "data_version"])

        # ── Tap streams ───────────────────────────────────────────────────
        self._tap_orig  = streams.Tap(x=None, y=None)
        self._tap_trans = streams.Tap(x=None, y=None)

        # ── Dynamic maps ──────────────────────────────────────────────────
        self._orig_dmap  = hv.DynamicMap(
            self._orig_cb, streams=[self._tap_orig, self._orig_stream])
        self._trans_dmap = hv.DynamicMap(
            self._trans_cb, streams=[self._tap_trans, self._trans_stream])
        # Explicit pane refs so _on_units_change can replace the object
        # and force a fresh Bokeh figure (updates CustomJSTickFormatter labels)
        self._orig_pane  = pn.pane.HoloViews(self._orig_dmap)
        self._trans_pane = pn.pane.HoloViews(self._trans_dmap)

        # ── Widgets ───────────────────────────────────────────────────────
        self._status = pn.pane.Markdown(self._status_md(), width=290)
        self._stats  = pn.pane.Markdown(self._stats_md(),  width=290)

        self._btn_transpose = pn.widgets.Button(
            name="▶  Transpose", button_type="success", width=138)
        self._btn_transpose.on_click(self._on_transpose)

        self._btn_reset = pn.widgets.Button(
            name="↺  Reset", button_type="warning", width=138)
        self._btn_reset.on_click(self._on_reset)

        # ── Watershed dropdown ────────────────────────────────────────────
        ws_names = (
            sorted(self._gdf_4326[WATERSHED_NAME_COL].dropna().unique().tolist())
            if self._has_watersheds else []
        )
        self._ws_select = pn.widgets.AutocompleteInput(
            name="Watershed",
            options=[""] + ws_names,
            value="",
            placeholder="Type to search…",
            min_characters=0,
            case_sensitive=False,
            width=290,
        )
        self._ws_select.param.watch(self._on_watershed_select, "value")

        # ── Rotation slider ───────────────────────────────────────────────
        self._rot_slider = pn.widgets.EditableFloatSlider(
            name="Rotation (° CW)",
            start=-180, end=180, step=1, value=0,
            width=290,
        )
        self._rot_slider.param.watch(self._on_rotation_change, "value_throttled")

        # Recompute t_ws_mask whenever rotation changes
        self.param.watch(self._on_rotation_param_change, "rotation_deg")

        # ── Fine-tune offset sliders ──────────────────────────────────────
        self._fine_lon_slider = pn.widgets.EditableFloatSlider(
            name="Fine-tune Δlon (°)",
            start=-1.0, end=1.0, step=0.01, value=0.0,
            width=290,
        )
        self._fine_lat_slider = pn.widgets.EditableFloatSlider(
            name="Fine-tune Δlat (°)",
            start=-1.0, end=1.0, step=0.01, value=0.0,
            width=290,
        )
        self._fine_lon_slider.param.watch(self._on_fine_lon_change, "value_throttled")
        self._fine_lat_slider.param.watch(self._on_fine_lat_change, "value_throttled")
        self.param.watch(self._on_fine_param_change, ["fine_lon", "fine_lat"])

        # ── Transposition method controls ─────────────────────────────────
        self._trans_method_toggle = pn.widgets.RadioButtonGroup(
            name="Transposition Method",
            options=["Simple Translation", "AEP Transposition (Atlas 14)"],
            value="Simple Translation",
            button_type="default",
            width=360,
            disabled=not self._has_atlas14,
        )
        self._trans_method_toggle.param.watch(self._on_trans_method_change, "value")

        self._aep_status = pn.pane.Markdown(
            ("*Select AEP Transposition to transpose using Atlas 14 climatology "
             "(source point must be on land).*") if self._has_atlas14 else
            "*Atlas 14 climatology file not found — AEP method unavailable.*",
            width=480,
        )

        # ── Spatial-plot display mode (observed QPE vs. Atlas14 recurrence interval) ──
        self._show_ari = False
        self._ari_display_toggle = pn.widgets.RadioButtonGroup(
            name="Spatial Plot Display",
            options=["Precipitation", "Recurrence Interval (Atlas 14)"],
            value="Precipitation",
            button_type="default",
            width=280,
            disabled=not self._has_atlas14,
        )
        self._ari_display_toggle.param.watch(self._on_display_mode_change, "value")

        # ── Manual coordinate inputs ──────────────────────────────────────
        self._src_lat_input = pn.widgets.FloatInput(
            name="Lat (°N)", value=None, placeholder="e.g. 21.3065",
            step=0.0001, width=130,
        )
        self._src_lon_input = pn.widgets.FloatInput(
            name="Lon (°E)", value=None, placeholder="e.g. -157.81",
            step=0.0001, width=130,
        )
        self._dst_lat_input = pn.widgets.FloatInput(
            name="Lat (°N)", value=None, placeholder="e.g. 21.3065",
            step=0.0001, width=130,
        )
        self._dst_lon_input = pn.widgets.FloatInput(
            name="Lon (°E)", value=None, placeholder="e.g. -157.81",
            step=0.0001, width=130,
        )
        self._btn_set_src = pn.widgets.Button(
            name="Set Source", button_type="primary", width=290)
        self._btn_set_dst = pn.widgets.Button(
            name="Set Dest", button_type="primary", width=290)
        self._btn_set_src.on_click(self._on_set_src)
        self._btn_set_dst.on_click(self._on_set_dst)

        # ── Time-series state ─────────────────────────────────────────────
        self._ts_grids = []
        self._ts_grids_version = 0   # incremented each reload; invalidates caches
        self._ts_accum_lons = None
        self._ts_accum_lats = None
        self._ts_accum_data = None

        # Cache for accumulated transposed grid — avoids 24× rotate_data per callback.
        # Key: (total_d_lon, total_d_lat, rotation_deg, piv_lon, piv_lat, ts_version)
        self._trans_accum_cache_key = None
        self._trans_accum_cache = None  # (t_lons, t_lats, t_data)

        # Cache for Mercator-regrided original single-hour data (never changes).
        self._orig_merc = regrid_to_mercator_lats(self.lons, self.lats, self.data)
        # Cache for Mercator-regrided accumulated original (set in _on_load_ts).
        self._accum_orig_merc = None   # pixel-sum of all original grids

        # ── Time-series widgets ───────────────────────────────────────────
        _init_dt = self._parse_dt_from_path(nc_path)
        self._dt_picker = pn.widgets.DatetimePicker(
            name="End date/time (UTC)",
            value=_init_dt,
            width=220,
        )
        self._period_select = pn.widgets.RadioButtonGroup(
            name="Period",
            options=["1 hr", "3 hr", "6 hr", "12 hr", "24 hr", "48 hr"],
            value="1 hr",
            button_type="default",
        )
        self._btn_load_ts = pn.widgets.Button(
            name="Load", button_type="primary", width=90)
        self._btn_load_ts.on_click(self._on_load_ts)
        self._period_select.param.watch(self._on_period_change, "value")

        self._ts_status = pn.pane.Markdown(
            f"*Showing: {self._valid_time}*", width=500)
        self._ts_pane = pn.pane.Bokeh(
            self._empty_ts_plot(), sizing_mode="stretch_width")
        self._ws_ts_pane = pn.pane.Bokeh(
            self._empty_ws_ts_plot(), sizing_mode="stretch_width")

        # ── Temporal modification state ───────────────────────────────────
        # Base (unmodified) per-hour areal mean values for primary series
        self._ws_ts_cur_vals  = []   # last computed (orig or trans)
        self._ws_ts_cur_times = []   # matching datetime list
        self._ws_ts_mod_vals  = None # modified array, or None
        self._ws_ts_snap_vals = None # snapshot at point selection for slider base
        self._ws_sel_hour_idx = None
        # _ws_tap_source: ColumnDataSource for the primary (tappable) series;
        # kept as an instance attribute so we can re-register the callback.
        self._ws_tap_source          = None
        # Modified-series renderers — kept so slider can push updates in-place.
        self._ws_mod_source          = None
        self._ws_mod_line_renderer   = None
        self._ws_mod_scatter_renderer = None

        # ── Modification widgets ──────────────────────────────────────────
        self._ws_mod_method = pn.widgets.Select(
            name="Method",
            options=["Proportional", "Gaussian", "Window"],
            value="Proportional",
            width=150,
        )
        self._ws_mod_sigma = pn.widgets.IntSlider(
            name="σ (hours)", start=1, end=12, value=3, width=150, visible=False)
        self._ws_mod_window = pn.widgets.IntSlider(
            name="Window (±hr)", start=1, end=12, value=3, width=150, visible=False)
        self._ws_mod_method.param.watch(self._on_mod_method_change, "value")
        self._ws_mod_sigma.param.watch(self._on_mod_param_change, "value")
        self._ws_mod_window.param.watch(self._on_mod_param_change, "value")

        self._ws_mod_slider = pn.widgets.EditableFloatSlider(
            name="Adjust selected hour (mm)",
            start=0, end=200, step=0.1, value=0,
            width=300, visible=False,
        )
        self._ws_mod_slider.param.watch(self._on_mod_slider_change, "value")

        self._ws_mod_info = pn.pane.Markdown(
            "*Click a point on the Watershed Areal Mean plot to select a timestep.*",
            width=400)
        self._btn_reset_mod = pn.widgets.Button(
            name="Reset Modifications", button_type="warning", width=160, visible=False)
        self._btn_reset_mod.on_click(self._on_reset_mod)

        # ── SAC-SMA model state ──────────────────────────────────────────────────
        # Cached spin-up state (keyed by event_start datetime str)
        self._sac_spinup_cache      = {}     # {event_start_str: state_dict}
        self._ws_ts_orig_vals       = []     # original (untransposed) areal mean
        self._sac_pane = pn.pane.Bokeh(
            self._empty_sac_plot(), sizing_mode='stretch_width')
        self._sac_stage_pane = pn.pane.Bokeh(
            self._empty_stage_plot(), sizing_mode='stretch_width')
        self._btn_run_sac = pn.widgets.Button(
            name='▶  Run SAC-SMA Model',
            button_type='success', width=180,
            disabled=not _SAC_AVAILABLE)
        self._sac_status = pn.pane.Markdown(
            '*Load a time series and select a calibrated watershed to run.*'
            if _SAC_AVAILABLE else
            '*sac_worker.py or hydro-model Python not found — check paths.*')
        if _SAC_AVAILABLE:
            self._btn_run_sac.on_click(self._on_run_sac)
        self._sac_progress = pn.indicators.Progress(
            active=False, bar_color='success',
            sizing_mode='stretch_width', visible=False, height=8,
        )
        # ── Units toggle ──────────────────────────────────────────────────────────
        self._use_imperial = False
        self._last_sac_data = None  # stored after SAC run for unit re-render
        self._units_toggle = pn.widgets.Toggle(
            name="Imperial units  (in, ft³/s, mi)",
            value=False,
            button_type="default",
            width=220,
        )
        self._units_toggle.param.watch(self._on_units_change, "value")

        # ── Rating curves (fetched once at startup for all calibrated basins) ──
        self._rating_curves = {}
        if _AVAILABLE_BASIN_IDS:
            self._rating_curves = fetch_all_rating_curves(_AVAILABLE_BASIN_IDS)

    # ── Time-series helpers ───────────────────────────────────────────────

    @staticmethod
    def _parse_dt_from_path(nc_path: str) -> datetime:
        """Extract datetime from mrms_qpe_1hr_hawaii_YYYYMMDD_HHMM.nc filename."""
        m = re.search(r"(\d{8})_(\d{4})", os.path.basename(nc_path))
        if m:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")
        return datetime.utcnow().replace(minute=0, second=0, microsecond=0)

    def _empty_ts_plot(self):
        """Placeholder Bokeh figure when no point time series data is loaded."""
        p = bk_figure(
            x_axis_type="datetime",
            height=260,
            sizing_mode="stretch_width",
            title="Time series — load a period and set source/dest points",
            x_axis_label="Time (UTC)",
            y_axis_label="QPE (mm)",
        )
        p.grid.grid_line_alpha = 0.3
        return p
    def _empty_ws_ts_plot(self):
        """Placeholder Bokeh figure when no watershed TS data is loaded."""
        p = bk_figure(
            x_axis_type="datetime",
            height=280,
            sizing_mode="stretch_width",
            title="Watershed areal mean — select a watershed and load a period",
            x_axis_label="Time (UTC)",
            y_axis_label="Areal Mean QPE (mm)",
        )
        p.grid.grid_line_alpha = 0.3
        return p
    def _on_load_ts(self, _):
        """Load MRMS files for the selected time period and update display."""
        end_dt = self._dt_picker.value
        if end_dt is None:
            return
        # DatetimePicker returns datetime; strip sub-hour components
        if hasattr(end_dt, "to_pydatetime"):
            end_dt = end_dt.to_pydatetime()
        end_dt = end_dt.replace(minute=0, second=0, microsecond=0)

        n_hours = int(self._period_select.value.split()[0])
        data_dir = os.path.dirname(self.nc_path)

        self._ts_status.object = "*Loading...*"
        file_list = find_mrms_files(data_dir, end_dt, n_hours)
        if not file_list:
            self._ts_status.object = "No files found for selected period."
            return

        # Load files in parallel (ThreadPoolExecutor), then sort by datetime
        with ThreadPoolExecutor() as pool:
            loaded = list(pool.map(load_mrms, [fp for _, fp in file_list]))
        self._ts_grids = [
            {"dt": dt, "lons": lons, "lats": lats, "data": data, "valid_time": vt}
            for (dt, _), (lons, lats, data, vt) in zip(file_list, loaded)
        ]
        self._ts_grids.sort(key=lambda g: g["dt"])

        # Increment version to invalidate the accumulated transposed cache
        self._ts_grids_version += 1
        self._trans_accum_cache_key = None
        self._trans_accum_cache = None

        # Compute accumulated original (pixel sum) and cache Mercator regrid
        ref = self._ts_grids[0]
        ref_shape = ref["data"].shape
        accum = np.zeros_like(ref["data"])
        for g in self._ts_grids:
            if g["data"].shape != ref_shape:
                continue  # skip grids with a different resolution/extent
            d = g["data"].copy()
            d[np.isnan(d)] = 0.0
            accum += d
        accum[accum == 0.0] = np.nan
        self._ts_accum_lons = ref["lons"]
        self._ts_accum_lats = ref["lats"]
        self._ts_accum_data = accum
        # Pre-compute Mercator regrid of accumulated original for fast redraws
        self._accum_orig_merc = regrid_to_mercator_lats(
            self._ts_accum_lons, self._ts_accum_lats, self._ts_accum_data)

        n = len(self._ts_grids)
        period_str = (
            f"1hr: {self._ts_grids[0]['valid_time']}"
            if n == 1
            else (
                f"{n}hr accum: {self._ts_grids[0]['valid_time']}"
                f" to {self._ts_grids[-1]['valid_time']}"
            )
        )
        self._ts_status.object = f"*Loaded {n} file(s) | {period_str}*"

        # Recompute watershed mean (uses accumulated data now)
        self._recompute_trans_mean()
        self._recompute_aep_field()
        self._stats.object = self._stats_md()

        # Redraw spatial maps with accumulated data — increment data_version
        # so both streams fire even if x/y haven't changed since last click.
        self.data_version += 1

        # Build and display time series
        self._clear_ws_mod_state()
        self._ts_pane.object = self._build_ts_plot()
        self._ws_ts_pane.object = self._build_ws_ts_plot()

    def _build_ts_plot(self):
        """Build hourly point time series as a Bokeh figure."""
        if not self._ts_grids:
            return self._empty_ts_plot()

        src_lon = src_lat = dst_lon = dst_lat = None
        if self._src_mx is not None:
            src_lon, src_lat = to_ll(self._src_mx, self._src_my)
        if self._dst_mx is not None:
            dst_lon, dst_lat = to_ll(self._dst_mx, self._dst_my)

        total_d_lon = self._d_lon + self.fine_lon
        total_d_lat = self._d_lat + self.fine_lat
        piv_lon, piv_lat = self._pivot_ll()

        transposed = (self._d_lon != 0.0 or self._d_lat != 0.0
                      or self.fine_lon != 0.0 or self.fine_lat != 0.0
                      or self.rotation_deg != 0.0)

        times_list = [g["dt"] for g in self._ts_grids]
        v_orig_list, v_trans_list, v_dst_orig_list = [], [], []
        for g in self._ts_grids:
            v_orig_list.append(
                point_value_at(g["lons"], g["lats"], g["data"], src_lon, src_lat))

            if dst_lon is not None and transposed:
                v_dst_orig_list.append(
                    point_value_at(g["lons"], g["lats"], g["data"], dst_lon, dst_lat))

            gd = self._apply_aep(g["data"])
            tl, tt, td = transpose_data(
                g["lons"], g["lats"], gd, total_d_lon, total_d_lat)
            tl, tt, td = rotate_data(
                tl, tt, td, self.rotation_deg,
                pivot_lon=piv_lon, pivot_lat=piv_lat)
            v_trans_list.append(point_value_at(tl, tt, td, dst_lon, dst_lat))

        n = len(self._ts_grids)
        title = (
            f"Hourly QPE — {self._ts_grids[0]['valid_time']}"
            if n == 1
            else f"Hourly QPE — {n}hr period ending {self._ts_grids[-1]['valid_time']}"
        )
        qs = self._qpe_scale()
        qu = self._qpe_unit()
        orig_total  = sum(v_orig_list) * qs
        trans_total = sum(v_trans_list) * qs
        v_orig_disp     = [v * qs for v in v_orig_list]
        v_trans_disp    = [v * qs for v in v_trans_list]
        v_dst_orig_disp = [v * qs for v in v_dst_orig_list]

        p = bk_figure(
            x_axis_type="datetime",
            height=260,
            sizing_mode="stretch_width",
            title=title,
            x_axis_label="Time (UTC)",
            y_axis_label=f"QPE ({qu})",
        )
        p.grid.grid_line_alpha = 0.3
        p.grid.grid_line_color = "gray"

        _COLORS = ["#FFB000", "#fc4f30", "#30a2da"]
        p.line(times_list, v_orig_disp, color=_COLORS[0], line_width=2,
               legend_label=f"Source point, original [Total: {orig_total:.2f} {qu}]")
        p.line(times_list, v_trans_disp, color=_COLORS[1], line_width=2,
               legend_label=f"Destination point, transposed [Total: {trans_total:.2f} {qu}]")
        if v_dst_orig_disp:
            dst_orig_total = sum(v_dst_orig_list) * qs
            p.line(times_list, v_dst_orig_disp, color=_COLORS[2], line_width=2,
                   legend_label=f"Destination point, original [Total: {dst_orig_total:.2f} {qu}]")

        p.legend.click_policy = "hide"
        p.add_layout(p.legend[0], "right")

        hover = HoverTool(
            tooltips=[("Time", "@x{%Y-%m-%d %H:%M}"), ("QPE", f"@y{{0.00}} {qu}")],
            formatters={"@x": "datetime"},
            mode="vline",
        )
        p.add_tools(hover)
        return p
    def _build_ws_ts_plot(self):
        """Build hourly watershed areal mean QPE as a Bokeh figure with tap support."""
        if (not self._ts_grids or not self._has_watersheds
                or not self.selected_watershed or self._ws_mask is None):
            self._ws_ts_cur_vals  = []
            self._ws_ts_cur_times = []
            return self._empty_ws_ts_plot()

        rows = self._gdf_4326[self._gdf_4326[WATERSHED_NAME_COL] == self.selected_watershed]
        if rows.empty:
            return self._empty_ws_ts_plot()
        geom_4326 = rows.iloc[0].geometry

        transposed = (
            self._d_lon != 0.0 or self._d_lat != 0.0
            or self.fine_lon != 0.0 or self.fine_lat != 0.0
            or self.rotation_deg != 0.0
        )
        total_d_lon = self._d_lon + self.fine_lon
        total_d_lat = self._d_lat + self.fine_lat
        piv_lon, piv_lat = self._pivot_ll()

        t_ws_mask = None
        if transposed and self._ts_grids:
            ref = self._ts_grids[0]
            gd_ref = self._apply_aep(ref["data"])
            tl, tt, td = transpose_data(
                ref["lons"], ref["lats"], gd_ref, total_d_lon, total_d_lat)
            tl, tt, td = rotate_data(
                tl, tt, td, self.rotation_deg, pivot_lon=piv_lon, pivot_lat=piv_lat)
            t_ws_mask = compute_watershed_mask(geom_4326, tl, tt)

        orig_vals, trans_vals = [], []
        for g in self._ts_grids:
            v_orig = spatial_mean(g["data"], self._ws_mask)
            orig_vals.append(v_orig if v_orig is not None else 0.0)

            if transposed and t_ws_mask is not None:
                gd = self._apply_aep(g["data"])
                tl, tt, td = transpose_data(
                    g["lons"], g["lats"], gd, total_d_lon, total_d_lat)
                tl, tt, td = rotate_data(
                    tl, tt, td, self.rotation_deg, pivot_lon=piv_lon, pivot_lat=piv_lat)
                v_trans = spatial_mean(td, t_ws_mask)
                trans_vals.append(v_trans if v_trans is not None else 0.0)

        times_list = [g["dt"] for g in self._ts_grids]
        n = len(self._ts_grids)

        # Primary (tappable) series is transposed when available, else original
        primary_vals = trans_vals if trans_vals else orig_vals
        self._ws_ts_cur_vals  = list(primary_vals)
        self._ws_ts_cur_times = list(times_list)
        self._ws_ts_orig_vals = list(orig_vals)  # always the untransposed series

        title = (
            f"Watershed Areal Mean QPE — {self.selected_watershed} — "
            + (f"{self._ts_grids[0]['valid_time']}" if n == 1
               else f"{n}hr period ending {self._ts_grids[-1]['valid_time']}")
        )

        qs = self._qpe_scale()
        qu = self._qpe_unit()
        p = bk_figure(
            x_axis_type="datetime",
            height=280,
            sizing_mode="stretch_width",
            title=title,
            x_axis_label="Time (UTC)",
            y_axis_label=f"Areal Mean QPE ({qu})",
        )
        p.grid.grid_line_alpha = 0.3
        p.grid.grid_line_color = "gray"

        _COLORS = ["#30a2da", "#fc4f30", "#2ca02c"]

        # ── Original series (line + circles) ─────────────────────────────
        orig_total = sum(orig_vals) * qs
        orig_src = ColumnDataSource(
            data={"x": times_list, "y": [v * qs for v in orig_vals],
                  "dt": [t.strftime("%Y-%m-%d %H:%Mz") for t in times_list]})
        p.line("x", "y", source=orig_src, color=_COLORS[0], line_width=2,
               legend_label=f"Original [Total: {orig_total:.2f} {qu}]")
        p.scatter("x", "y", source=orig_src, color=_COLORS[0], size=8,
                  line_color="white", line_width=1)

        # ── Transposed series (line + tappable circles) ───────────────────
        if trans_vals:
            trans_total = sum(trans_vals) * qs
            tap_src = ColumnDataSource(
                data={"x": times_list, "y": [v * qs for v in trans_vals],
                      "dt": [t.strftime("%Y-%m-%d %H:%Mz") for t in times_list]})
            p.line("x", "y", source=tap_src, color=_COLORS[1], line_width=2,
                   legend_label=f"Transposed [Total: {trans_total:.2f} {qu}]")
            tap_renderer = p.scatter("x", "y", source=tap_src, color=_COLORS[1],
                                     size=10, line_color="white", line_width=1)
        else:
            orig_total2 = sum(orig_vals) * qs
            tap_src = ColumnDataSource(
                data={"x": times_list, "y": [v * qs for v in orig_vals],
                      "dt": [t.strftime("%Y-%m-%d %H:%Mz") for t in times_list]})
            tap_renderer = p.scatter("x", "y", source=tap_src, color=_COLORS[0],
                                     size=10, line_color="white", line_width=1)

        # Register Bokeh server-side tap callback on the tappable source
        self._ws_tap_source = tap_src

        def _on_tap(attr, old, new_indices):
            if not new_indices:
                return
            idx = new_indices[0]
            if idx < 0 or idx >= len(self._ws_ts_cur_vals):
                return
            self._ws_sel_hour_idx = idx
            current = (list(self._ws_ts_mod_vals)
                       if self._ws_ts_mod_vals is not None
                       else list(self._ws_ts_cur_vals))
            self._ws_ts_snap_vals = current
            cur_val  = current[idx]
            total    = float(sum(current))
            dt_label = self._ws_ts_cur_times[idx].strftime("%Y-%m-%d %H:%Mz")                        if self._ws_ts_cur_times else str(idx)
            _qs = self._qpe_scale()
            _qu = self._qpe_unit()
            self._ws_mod_info.object = (
                f"**Selected:** {dt_label} — current value: **{cur_val * _qs:.3f} {_qu}**")
            self._ws_mod_slider.start   = 0.0
            self._ws_mod_slider.end     = round(max(total, cur_val * 3, 10.0) * _qs, 3)
            self._ws_mod_slider.value   = round(cur_val * _qs, 4)
            self._ws_mod_slider.visible = True
            self._btn_reset_mod.visible = True

        tap_src.selected.on_change("indices", _on_tap)
        p.add_tools(TapTool(renderers=[tap_renderer]))

        # ── Modified series — always created; shown/hidden in-place ──────
        _mod_active = (self._ws_ts_mod_vals is not None
                       and len(self._ws_ts_mod_vals) == n)
        mod_y = [v * qs for v in self._ws_ts_mod_vals] if _mod_active else [0.0] * n
        mod_total = sum(mod_y) if _mod_active else 0.0
        mod_label = (f"Modified [Total: {mod_total:.2f} {qu}]"
                     if _mod_active else "Modified")
        mod_dt = [t.strftime("%Y-%m-%d %H:%Mz") for t in times_list]
        self._ws_mod_source = ColumnDataSource(
            data={"x": times_list, "y": mod_y, "dt": mod_dt})
        self._ws_mod_line_renderer = p.line(
            "x", "y", source=self._ws_mod_source,
            color=_COLORS[2], line_width=2, line_dash="dashed",
            legend_label=mod_label, visible=_mod_active)
        self._ws_mod_scatter_renderer = p.scatter(
            "x", "y", source=self._ws_mod_source,
            color=_COLORS[2], size=8, line_color="white", line_width=1,
            visible=_mod_active)

        # Hover over any renderer
        hover = HoverTool(
            tooltips=[("Time", "@dt"), ("Value", f"@y{{0.00}} {qu}")],
            mode="mouse",
        )
        p.add_tools(hover)

        p.legend.click_policy = "hide"
        p.add_layout(p.legend[0], "right")
        return p
    # ── Temporal modification callbacks ───────────────────────────────────

    def _empty_sac_plot(self):
        """Placeholder Bokeh figure for SAC-SMA streamflow output."""
        p = bk_figure(
            x_axis_type='datetime', height=260,
            sizing_mode='stretch_width',
            title='SAC-SMA Streamflow — click "▶ Run SAC-SMA Model" to compute',
            x_axis_label='Time (UTC)',
            y_axis_label='Streamflow (m³/s)',
        )
        p.grid.grid_line_alpha = 0.3
        return p

    def _empty_stage_plot(self, msg: str = ''):
        """Placeholder Bokeh figure for stage height output."""
        title = msg or 'Stage height — run the SAC-SMA model to compute'
        p = bk_figure(
            x_axis_type='datetime', height=200,
            sizing_mode='stretch_width',
            title=title,
            x_axis_label='Time (UTC)',
            y_axis_label='Stage (ft)',
        )
        p.grid.grid_line_alpha = 0.3
        return p

    def _on_run_sac(self, _):
        """Run SAC-SMA spin-up + event via subprocess (hydro-model env)."""
        import subprocess
        import json as _json
        import threading as _threading

        _COLORS = ['#30a2da', '#fc4f30', '#2ca02c']

        if not _SAC_AVAILABLE:
            self._sac_status.object = '*SAC worker not available.*'
            return
        if not self._ts_grids or not self._ws_ts_cur_vals:
            self._sac_status.object = '*Load a time series first.*'
            return

        basin_id = self._basin_id_for_name.get(self.selected_watershed)
        if not basin_id:
            self._sac_status.object = '*Select a calibrated watershed first.*'
            return
        ws_name = self.selected_watershed

        times_list  = self._ws_ts_cur_times
        n           = len(times_list)

        # Build scenarios dict
        scenarios = {}
        if self._ws_ts_orig_vals and len(self._ws_ts_orig_vals) == n:
            scenarios['Original'] = list(self._ws_ts_orig_vals)
        transposed = (
            self._d_lon != 0.0 or self._d_lat != 0.0
            or self.fine_lon != 0.0 or self.fine_lat != 0.0
            or self.rotation_deg != 0.0
        )
        if transposed and self._ws_ts_cur_vals and len(self._ws_ts_cur_vals) == n:
            scenarios['Transposed'] = list(self._ws_ts_cur_vals)
        if self._ws_ts_mod_vals is not None and len(self._ws_ts_mod_vals) == n:
            scenarios['Modified'] = list(self._ws_ts_mod_vals)
        if not scenarios:
            self._sac_status.object = '*No precip scenarios available.*'
            return

        payload = _json.dumps({
            'basin':       basin_id,
            'n_extend':    12,
            'event_start': pd.Timestamp(times_list[0]).isoformat(),
            'times':       [pd.Timestamp(t).isoformat() for t in times_list],
            'scenarios':   scenarios,
        })

        n_scen = len(scenarios)
        self._btn_run_sac.disabled = True
        self._sac_progress.active  = True
        self._sac_progress.visible = True
        self._sac_status.object = (
            f'*Running spin-up (\u22481.7 s) + {n_scen} scenario(s)\u2026*'
        )

        def _run():
            try:
                result = subprocess.run(
                    [str(_HYDRO_MODEL_PY), str(_SAC_WORKER)],
                    input=payload, capture_output=True, text=True, timeout=180,
                )
                if result.returncode != 0:
                    self._sac_status.object = (
                        f'*Worker error (code {result.returncode}): '
                        f'{result.stderr[:400]}*')
                    return
                runs_raw = _json.loads(result.stdout)
                if 'error' in runs_raw:
                    self._sac_status.object = f'*{runs_raw["error"]}*'
                    return

                spinup_date = runs_raw.pop('_spinup_date', None)
                times_plot  = [pd.Timestamp(t) for t in
                               runs_raw.pop('_times_all', [])]
                if not times_plot:
                    times_plot = times_list
                n_event = int(runs_raw.pop('_n_event', n))
                runs = {label: np.array(q) for label, q in runs_raw.items()}

                # Store results so plot can be rebuilt when units change
                self._last_sac_data = {
                    'runs':        runs,
                    'times_plot':  times_plot,
                    'n_event':     n_event,
                    'ws_name':     ws_name,  # noqa: F821
                    'basin_id':    basin_id,  # noqa: F821
                    'spinup_date': spinup_date,
                }
                self._rebuild_sac_plot()

                # \u2500\u2500 Build Bokeh flow figure \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

            except subprocess.TimeoutExpired:
                self._sac_status.object = '*Timed out after 180 s.*'
            except Exception as exc:
                self._sac_status.object = f'*Failed: {exc}*'
            finally:
                self._btn_run_sac.disabled = False
                self._sac_progress.active  = False
                self._sac_progress.visible = False

        _threading.Thread(target=_run, daemon=True).start()

    def _apply_mod_inplace(self):
        """Push current _ws_ts_mod_vals into the mod ColumnDataSource in-place."""
        if self._ws_mod_source is None or self._ws_ts_mod_vals is None:
            return
        times_list = self._ws_ts_cur_times
        qs = self._qpe_scale()
        self._ws_mod_source.data = {
            "x": times_list,
            "y": [v * qs for v in self._ws_ts_mod_vals],
            "dt": [t.strftime("%Y-%m-%d %H:%Mz") for t in times_list],
        }
        if self._ws_mod_line_renderer is not None:
            self._ws_mod_line_renderer.visible = True
        if self._ws_mod_scatter_renderer is not None:
            self._ws_mod_scatter_renderer.visible = True

    def _on_mod_slider_change(self, event):
        """Apply redistribution based on slider value."""
        idx = self._ws_sel_hour_idx
        if idx is None or not self._ws_ts_snap_vals:
            return
        new_val = float(event.new) / self._qpe_scale()  # slider is in display units; convert to mm
        method  = self._ws_mod_method.value
        sigma   = float(self._ws_mod_sigma.value)
        window  = int(self._ws_mod_window.value)
        self._ws_ts_mod_vals = redistribute_ts(
            self._ws_ts_snap_vals, idx, new_val, method, sigma, window)
        self._apply_mod_inplace()

    def _on_mod_method_change(self, event):
        """Show/hide σ or window slider depending on method choice."""
        method = event.new
        self._ws_mod_sigma.visible  = (method == "Gaussian")
        self._ws_mod_window.visible = (method == "Window")
        # Reapply redistribution with new method if a point is selected
        self._on_mod_param_change(event)

    def _on_mod_param_change(self, event):
        """Re-run redistribution when σ or window changes."""
        if self._ws_sel_hour_idx is None or not self._ws_ts_snap_vals:
            return
        new_val = float(self._ws_mod_slider.value)
        method  = self._ws_mod_method.value
        sigma   = float(self._ws_mod_sigma.value)
        window  = int(self._ws_mod_window.value)
        self._ws_ts_mod_vals = redistribute_ts(
            self._ws_ts_snap_vals, self._ws_sel_hour_idx,
            new_val, method, sigma, window)
        self._apply_mod_inplace()

    def _on_reset_mod(self, _):
        """Clear all temporal modifications."""
        self._ws_ts_mod_vals  = None
        self._ws_ts_snap_vals = None
        self._ws_sel_hour_idx = None
        self._ws_mod_slider.visible   = False
        self._btn_reset_mod.visible   = False
        self._ws_mod_info.object = (
            "*Click a point on the Watershed Areal Mean plot to select a timestep.*")
        if self._ws_mod_line_renderer is not None:
            self._ws_mod_line_renderer.visible = False
        if self._ws_mod_scatter_renderer is not None:
            self._ws_mod_scatter_renderer.visible = False

    def _clear_ws_mod_state(self):
        """Reset modification state (called on Load/Reset/watershed change)."""
        self._ws_ts_mod_vals  = None
        self._ws_ts_snap_vals = None
        self._ws_sel_hour_idx = None
        self._ws_mod_slider.visible   = False
        self._btn_reset_mod.visible   = False
        self._ws_mod_info.object = (
            "*Click a point on the Watershed Areal Mean plot to select a timestep.*")
        if self._ws_mod_line_renderer is not None:
            self._ws_mod_line_renderer.visible = False
        if self._ws_mod_scatter_renderer is not None:
            self._ws_mod_scatter_renderer.visible = False

    def _on_period_change(self, event):
        """Auto-reload when 1hr/24hr toggle is switched."""
        self._on_load_ts(None)

    def _on_watershed_select(self, event):
        name = event.new.strip()

        # Recompute mask for the new selection
        if self._has_watersheds and name:
            rows = self._gdf_4326[self._gdf_4326[WATERSHED_NAME_COL] == name]
            if not rows.empty:
                geom_4326 = rows.iloc[0].geometry
                self._ws_mask = compute_watershed_mask(geom_4326, self.lons, self.lats)
            else:
                self._ws_mask = None
        else:
            self._ws_mask = None

        # Set param FIRST so _recompute_t_ws_mask sees the correct name
        self.selected_watershed = name

        # Recompute transposed mean with current shifts/rotation
        self._recompute_trans_mean()

        # Update stats text and watershed time series
        self._stats.object = self._stats_md()
        self._clear_ws_mod_state()
        self._ws_ts_pane.object = self._build_ws_ts_plot()

    # ── Transposed accumulation cache ────────────────────────────────────

    def _get_trans_accum(self, total_d_lon, total_d_lat, rotation_deg, piv_lon, piv_lat):
        """
        Return the accumulated (or single-hour) transposed+rotated grid,
        reusing the cached result if the parameters haven't changed.

        This is the key performance hotspot: in 24hr mode it runs
        24× rotate_data (scipy.ndimage.rotate), which is expensive.
        Caching means it only runs once per unique set of transposition
        parameters, not once per callback.
        """
        key = (
            round(total_d_lon, 7), round(total_d_lat, 7),
            round(rotation_deg, 5),
            None if piv_lon is None else round(piv_lon, 7),
            None if piv_lat is None else round(piv_lat, 7),
            self._ts_grids_version,
            self._aep_enabled, self._aep_version,
        )
        if key == self._trans_accum_cache_key:
            return self._trans_accum_cache

        if self._ts_grids:
            accum = None
            t_lons_ref = t_lats_ref = None
            for g in self._ts_grids:
                gd = self._apply_aep(g["data"])
                tl, tt, td = transpose_data(
                    g["lons"], g["lats"], gd, total_d_lon, total_d_lat)
                tl, tt, td = rotate_data(
                    tl, tt, td, rotation_deg,
                    pivot_lon=piv_lon, pivot_lat=piv_lat)
                d = td.copy()
                d[np.isnan(d)] = 0.0
                if accum is None:
                    accum = d
                elif accum.shape == d.shape:
                    accum += d
                t_lons_ref, t_lats_ref = tl, tt
            if accum is not None:
                accum[accum == 0.0] = np.nan
            result = (t_lons_ref, t_lats_ref, accum)
        else:
            t_lons, t_lats, t_data = transpose_data(
                self.lons, self.lats, self._apply_aep(self.data), total_d_lon, total_d_lat)
            t_lons, t_lats, t_data = rotate_data(
                t_lons, t_lats, t_data, rotation_deg,
                pivot_lon=piv_lon, pivot_lat=piv_lat)
            result = (t_lons, t_lats, t_data)

        self._trans_accum_cache_key = key
        self._trans_accum_cache = result
        return result

    # ── Transposed-mean helper ───────────────────────────────────────────

    def _recompute_trans_mean(self):
        """
        Compute and cache the mean QPE the selected watershed would receive
        after the current transposition + rotation + fine-tune.

        When a multi-hour time series is loaded the watershed mean is computed
        on the *accumulated* (pixel-summed) transposed grid so it represents
        total QPE over the selected period, consistent with what the spatial
        map shows.
        """
        if not self._has_watersheds or not self.selected_watershed or self._ws_mask is None:
            self._trans_mean = None
            return

        rows = self._gdf_4326[self._gdf_4326[WATERSHED_NAME_COL] == self.selected_watershed]
        if rows.empty:
            self._trans_mean = None
            return

        geom_4326 = rows.iloc[0].geometry
        total_d_lon = self._d_lon + self.fine_lon
        total_d_lat = self._d_lat + self.fine_lat
        piv_lon, piv_lat = self._pivot_ll()

        # Reuse cached accumulated transposed grid (no recompute if params unchanged)
        r_lons, r_lats, r_data = self._get_trans_accum(
            total_d_lon, total_d_lat, self.rotation_deg, piv_lon, piv_lat)

        r_mask = compute_watershed_mask(geom_4326, r_lons, r_lats)
        self._trans_mean = spatial_mean(r_data, r_mask)

    # ── Atlas 14 helpers ──────────────────────────────────────────────────

    def _is_land(self, lon, lat) -> bool:
        """True if (lon, lat) falls on a valid (non-water) Atlas 14 grid cell."""
        if not self._has_atlas14:
            return True
        dlon = self._a14_lons[1] - self._a14_lons[0]
        dlat = self._a14_lats[1] - self._a14_lats[0]
        if not (self._a14_lons[0] - dlon / 2 <= lon <= self._a14_lons[-1] + dlon / 2):
            return False
        if not (self._a14_lats[0] - dlat / 2 <= lat <= self._a14_lats[-1] + dlat / 2):
            return False
        ix = int(np.clip(round((lon - self._a14_lons[0]) / dlon), 0, len(self._a14_lons) - 1))
        iy = int(np.clip(round((lat - self._a14_lats[0]) / dlat), 0, len(self._a14_lats) - 1))
        return bool(self._a14_land_mask[iy, ix])

    def _get_a14_duration_slice(self, duration):
        """
        Return (precip_allari, ari_values) for the given Atlas14 duration,
        shape (ARI, y, x).  Cached per duration since each slice is ~75 MB.
        """
        if duration in self._a14_duration_cache:
            return self._a14_duration_cache[duration]
        try:
            da = self._a14_ds["precip"].sel(duration=duration, method="nearest")
            da = da.isel(confidence=0, reg=0).transpose("ARI", "y", "x")
            arr = da.values
            ari_values = self._a14_ds["ARI"].values.astype(float)
        except Exception as exc:
            print(f"  [warn] Atlas14 duration slice failed: {exc}")
            return None, None
        self._a14_duration_cache[duration] = (arr, ari_values)
        return arr, ari_values

    def _compute_display_ari_field(self, data, lons, lats):
        """
        Convert each grid cell's own observed magnitude into its equivalent
        Atlas14 average recurrence interval (years), for the "Recurrence
        Interval" spatial-plot display mode.  Returns None if Atlas14 or the
        selected duration is unavailable.  Dry cells and cells outside
        Atlas14 coverage come out as NaN.
        """
        if not self._has_atlas14:
            return None
        duration = _ATLAS14_DURATION_MAP.get(self._period_select.value, pd.Timedelta(hours=1))
        arr_allari, ari_values = self._get_a14_duration_slice(duration)
        if arr_allari is None:
            return None

        ix = _nearest_indices(self._a14_lons, lons)
        iy = _nearest_indices(self._a14_lats, lats)
        land = self._a14_land_mask[iy[:, None], ix[None, :]]
        amounts = arr_allari[:, iy[:, None], ix[None, :]].astype(np.float32)

        log_ari = np.log(ari_values)
        ref = data.astype(np.float32)
        dry_mask = ~np.isfinite(ref) | (ref < PRECIP_BIN_EDGES[0])

        ari_field = np.exp(_invert_to_log_ari(ref, amounts, log_ari))
        ari_field[dry_mask | ~land] = np.nan
        return ari_field

    def _recompute_aep_field(self):
        """
        Recompute the Atlas14-based AEP transposition scaling field: every wet
        cell's own observed magnitude is converted to an equivalent Atlas14
        return period (AEP) using the climatology AT ITS OWN position, then
        re-evaluated against the climatology at its *transposed* position to
        get the new magnitude.  Values outside Atlas14's ARI range are
        log-linearly extrapolated.

        Cells lacking Atlas14 coverage at either the source or the
        transposed position are blanked (NaN); dry cells are left at 0 so no
        rain is introduced where none existed.
        """
        self._aep_field = None
        if not self._aep_enabled:
            return
        self._aep_version += 1

        if not self._has_atlas14:
            self._aep_status.object = "*Atlas 14 climatology file not found — unavailable.*"
            return
        if self._src_mx is None:
            self._aep_status.object = "*Set a source point on land to compute AEP transposition.*"
            return

        src_lon, src_lat = to_ll(self._src_mx, self._src_my)
        if not self._is_land(src_lon, src_lat):
            self._aep_status.object = "⚠️ Source point is over water — unavailable."
            return

        ref_data = self._ts_accum_data if self._ts_accum_data is not None else self.data

        period_label = self._period_select.value
        duration = _ATLAS14_DURATION_MAP.get(period_label, pd.Timedelta(hours=1))

        arr_allari, ari_values = self._get_a14_duration_slice(duration)
        if arr_allari is None:
            self._aep_status.object = "⚠️ Atlas 14 duration lookup failed."
            return

        total_d_lon = self._d_lon + self.fine_lon
        total_d_lat = self._d_lat + self.fine_lat

        ix_src = _nearest_indices(self._a14_lons, self.lons)
        iy_src = _nearest_indices(self._a14_lats, self.lats)
        ix_dst = _nearest_indices(self._a14_lons, self.lons + total_d_lon)
        iy_dst = _nearest_indices(self._a14_lats, self.lats + total_d_lat)

        land_src = self._a14_land_mask[iy_src[:, None], ix_src[None, :]]
        land_dst = self._a14_land_mask[iy_dst[:, None], ix_dst[None, :]]

        amounts_src = arr_allari[:, iy_src[:, None], ix_src[None, :]].astype(np.float32)
        amounts_dst = arr_allari[:, iy_dst[:, None], ix_dst[None, :]].astype(np.float32)

        log_ari = np.log(ari_values)
        ref = ref_data.astype(np.float32)
        dry_mask = ~np.isfinite(ref) | (ref < PRECIP_BIN_EDGES[0])

        log_ari_field = _invert_to_log_ari(ref, amounts_src, log_ari)
        v_new = _forward_from_log_ari(log_ari_field, amounts_dst, log_ari)
        v_new = np.maximum(v_new, 0.0)

        ratio = np.full(ref.shape, np.nan, dtype=np.float32)
        ratio[dry_mask] = 0.0
        valid = (~dry_mask) & land_src & land_dst
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_valid = np.where(ref > 0, v_new / ref, 0.0)
        ratio[valid] = ratio_valid[valid]

        self._aep_field = ratio
        n_wet = int(np.count_nonzero(~dry_mask))
        n_blocked = int(np.count_nonzero(~dry_mask & ~(land_src & land_dst)))
        self._aep_status.object = (
            f"**AEP transposition active** — duration: {period_label} · "
            f"{n_blocked}/{max(n_wet, 1)} wet cell(s) blanked (no Atlas14 coverage)"
        )

    def _apply_aep(self, data):
        """Multiply by the cached AEP scaling field when active, else pass through."""
        if not self._aep_enabled or self._aep_field is None:
            return data
        if data.shape != self._aep_field.shape:
            return data
        return data * self._aep_field

    def _on_trans_method_change(self, event):
        """Switch between simple translation and full-field AEP transposition."""
        self._aep_enabled = (event.new == "AEP Transposition (Atlas 14)")

        # AEP transposition is pure-translation — disable rotation while active
        self._rot_slider.disabled = self._aep_enabled
        if self._aep_enabled and self.rotation_deg != 0.0:
            self._rot_slider.value = 0.0
            self.rotation_deg = 0.0

        # A previously-set source point over water is invalid in AEP mode
        if self._aep_enabled and self._src_mx is not None:
            slon, slat = to_ll(self._src_mx, self._src_my)
            if not self._is_land(slon, slat):
                self._src_mx = self._src_my = None
                self._src_lat_input.value = None
                self._src_lon_input.value = None
                self._aep_status.object = (
                    "⚠️ **Previous source point was over water and was cleared.** "
                    "Pick a source point on land.")

        self._recompute_aep_field()
        self.data_version += 1
        self._status.object = self._status_md()
        self._stats.object  = self._stats_md()
        self._clear_ws_mod_state()
        if self._ts_grids:
            self._ts_pane.object    = self._build_ts_plot()
            self._ws_ts_pane.object = self._build_ws_ts_plot()

    def _on_display_mode_change(self, event):
        """Toggle spatial-plot display between observed QPE and Atlas14 recurrence interval."""
        self._show_ari = (event.new == "Recurrence Interval (Atlas 14)")
        self.data_version += 1
        self._rebuild_spatial_dmaps()

    # ── Rotation callbacks ─────────────────────────────────────────────────

    def _pivot_ll(self, fine_lon=None, fine_lat=None):
        """Return (lon, lat) pivot for rotation: destination point if set, else transposed grid centre."""
        if fine_lon is None:
            fine_lon = self.fine_lon
        if fine_lat is None:
            fine_lat = self.fine_lat
        if self._dst_mx is not None and self._dst_my is not None:
            dst_lon, dst_lat = to_ll(self._dst_mx, self._dst_my)
            return dst_lon + fine_lon, dst_lat + fine_lat
        cx = 0.5 * (float(self.lons.min()) + float(self.lons.max())) + self._d_lon + fine_lon
        cy = 0.5 * (float(self.lats.min()) + float(self.lats.max())) + self._d_lat + fine_lat
        return cx, cy

    def _on_rotation_change(self, event):
        """Slider → param (drives DynamicMap redraw + mask update)."""
        self.rotation_deg = float(event.new)

    def _on_rotation_param_change(self, event):
        """param → recompute trans mean + stats + ts."""
        self._recompute_trans_mean()
        self._stats.object = self._stats_md()
        if self._src_mx is not None and self._dst_mx is not None:
            self._ts_pane.object = self._build_ts_plot()
        self._ws_ts_pane.object = self._build_ws_ts_plot()

    def _on_fine_lon_change(self, event):
        self.fine_lon = float(event.new)

    def _on_fine_lat_change(self, event):
        self.fine_lat = float(event.new)

    def _on_fine_param_change(self, event):
        """Recompute trans mean + stats + ts whenever fine_lon or fine_lat changes."""
        self._recompute_trans_mean()
        self._recompute_aep_field()
        self._stats.object = self._stats_md()
        if self._src_mx is not None and self._dst_mx is not None:
            self._ts_pane.object = self._build_ts_plot()
        self._ws_ts_pane.object = self._build_ws_ts_plot()

    def _on_set_src(self, _):
        """Apply manually entered source coordinates."""
        lat = self._src_lat_input.value
        lon = self._src_lon_input.value
        if lat is None or lon is None:
            return
        if self._aep_enabled and not self._is_land(lon, lat):
            self._aep_status.object = "⚠️ **AEP transposition requires a source point on land.** Pick another location."
            return
        self._src_mx, self._src_my = to_merc(lon, lat)
        self._recompute_aep_field()
        self._status.object = self._status_md()
        self.data_version += 1

    def _on_set_dst(self, _):
        """Apply manually entered destination coordinates."""
        lat = self._dst_lat_input.value
        lon = self._dst_lon_input.value
        if lat is None or lon is None:
            return
        self._dst_mx, self._dst_my = to_merc(lon, lat)
        self._status.object = self._status_md()
        self.data_version += 1

    # ── Plot callbacks ────────────────────────────────────────────────────

    def _selected_ws_geom_3857(self):
        """Return the EPSG:3857 geometry for the current selection, or None."""
        name = self.selected_watershed
        if not self._has_watersheds or not name:
            return None
        rows = self._gdf_3857[self._gdf_3857[WATERSHED_NAME_COL] == name]
        return rows.iloc[0].geometry if not rows.empty else None

    def _sel_gauge_point(self):
        """Return a highlighted hv.Points for the selected watershed's gauge, or empty."""
        name = self.selected_watershed
        if (not name or name not in self._basin_id_for_name
                or self._gauge_gdf_3857 is None):
            return hv.Points([], kdims=["x", "y"])
        gauge_id = "usgs-" + self._basin_id_for_name[name]
        rows = self._gauge_gdf_3857[self._gauge_gdf_3857["id"] == gauge_id]
        if rows.empty:
            return hv.Points([], kdims=["x", "y"])
        row = rows.iloc[0]
        return hv.Points(
            {"x": [row.geometry.x], "y": [row.geometry.y],
             "name": [row["name"]], "id": [self._basin_id_for_name[name]]},
            kdims=["x", "y"], vdims=["name", "id"], label="Selected Gauge",
        ).opts(size=14, color="blue", marker="star",
               line_color="white", line_width=1, tools=["hover"])

    def _orig_cb(self, x, y, selected_watershed, data_version):
        """Left pane: records source click, redraws original overlay."""
        if x is not None and y is not None:
            slon, slat = to_ll(x, y)
            if self._aep_enabled and not self._is_land(slon, slat):
                self._aep_status.object = (
                    "⚠️ **AEP transposition requires a source point on land.** Pick another location.")
            else:
                self._src_mx, self._src_my = x, y
                self._src_lat_input.value = round(slat, 4)
                self._src_lon_input.value = round(slon, 4)
                self._recompute_aep_field()
                self._status.object = self._status_md()
                # Only rebuild ts when both points are set
                if self._dst_mx is not None:
                    self._ts_pane.object = self._build_ts_plot()

        # Use accumulated data when a multi-hour period is loaded;
        # pass pre-computed Mercator regrid to avoid recomputing each call.
        if self._ts_accum_data is not None:
            ref_lons, ref_lats, ref_data = (
                self._ts_accum_lons, self._ts_accum_lats, self._ts_accum_data)
            merc = self._accum_orig_merc  # pre-computed in _on_load_ts
            n = len(self._ts_grids)
            period_label = (
                f"{n}hr accum QPE"
                if n > 1
                else f"1hr QPE {self._ts_grids[0]['valid_time']}"
            )
        else:
            ref_lons, ref_lats, ref_data = self.lons, self.lats, self.data
            merc = self._orig_merc  # pre-computed in __init__
            period_label = f"1hr QPE {self._valid_time}"

        ari_mode = False
        if self._show_ari:
            ari_field = self._compute_display_ari_field(ref_data, ref_lons, ref_lats)
            if ari_field is not None:
                merc = regrid_to_mercator_lats(ref_lons, ref_lats, ari_field)
                ari_mode = True
                period_label = period_label.replace("QPE", "ARI")

        sel_poly = build_selected_watershed_polygon(self._selected_ws_geom_3857())
        overlay  = (
            make_tiles()
            * make_image(*merc, _already_merc=True, use_imperial=self._use_imperial,
                         long_duration=self._is_long_duration(), ari_mode=ari_mode)
            * self._coastline_overlay
            * self._all_ws_polys
            * sel_poly
            * self._gauge_points
            * self._sel_gauge_point()
            * make_marker(self._src_mx, self._src_my, "black")
        )
        return overlay.opts(
            hv.opts.Overlay(width=PLOT_W, height=PLOT_H,
                            title=f"Original - click to set SOURCE\n{period_label}",
                            legend_position='bottom_left')
        )

    def _trans_cb(self, x, y, selected_watershed, rotation_deg, fine_lon, fine_lat, data_version):
        """Right pane: records destination click, redraws transposed+rotated+fine-tuned overlay."""
        if x is not None and y is not None:
            self._dst_mx, self._dst_my = x, y
            dlon, dlat = to_ll(x, y)
            self._dst_lat_input.value = round(dlat, 4)
            self._dst_lon_input.value = round(dlon, 4)
            self._status.object = self._status_md()
            # Only rebuild ts when both points are set
            if self._src_mx is not None:
                self._ts_pane.object = self._build_ts_plot()

        total_d_lon = self._d_lon + fine_lon
        total_d_lat = self._d_lat + fine_lat
        piv_lon, piv_lat = self._pivot_ll(fine_lon, fine_lat)

        # Use cached accumulated transposed grid (avoids 24× rotate_data per call)
        t_lons, t_lats, t_data = self._get_trans_accum(
            total_d_lon, total_d_lat, rotation_deg, piv_lon, piv_lat)

        disp_lons, disp_lats, disp_data = t_lons, t_lats, t_data
        ari_mode = False
        if self._show_ari:
            ari_field = self._compute_display_ari_field(t_data, t_lons, t_lats)
            if ari_field is not None:
                disp_data = ari_field
                ari_mode = True

        if self._ts_grids:
            n = len(self._ts_grids)
            period_label = (
                f"{n}hr accum QPE"
                if n > 1
                else f"1hr QPE {self._ts_grids[0]['valid_time']}"
            )
        else:
            period_label = f"1hr QPE {self._valid_time}"
        if self._aep_enabled:
            period_label += " (AEP)"
        if ari_mode:
            period_label = period_label.replace("QPE", "ARI")

        sel_poly = build_selected_watershed_polygon(self._selected_ws_geom_3857())
        overlay  = (
            make_tiles()
            * make_image(disp_lons, disp_lats, disp_data, use_imperial=self._use_imperial,
                         long_duration=self._is_long_duration(), ari_mode=ari_mode)
            * self._coastline_overlay
            * self._all_ws_polys
            * sel_poly
            * self._gauge_points
            * self._sel_gauge_point()
            * make_marker(self._dst_mx, self._dst_my, "black")
        )
        return overlay.opts(
            hv.opts.Overlay(width=PLOT_W, height=PLOT_H,
                            title=f"Transposed - click to set DESTINATION\n{period_label}",
                            legend_position='bottom_left')
        )

    # ── Button callbacks ──────────────────────────────────────────────────

    def _on_transpose(self, _):
        if self._src_mx is None or self._dst_mx is None:
            self._status.object = (
                "⚠️ **Set both source and destination first.**\n\n"
                + self._status_md()
            )
            return

        src_lon, src_lat = to_ll(self._src_mx, self._src_my)
        dst_lon, dst_lat = to_ll(self._dst_mx, self._dst_my)

        self._d_lon = dst_lon - src_lon
        self._d_lat = dst_lat - src_lat

        self._recompute_trans_mean()
        self._recompute_aep_field()

        self.data_version += 1
        self._status.object = self._status_md()
        self._stats.object  = self._stats_md()
        # Update time series plots with new transposition
        self._clear_ws_mod_state()
        self._ts_pane.object = self._build_ts_plot()
        self._ws_ts_pane.object = self._build_ws_ts_plot()

    def _on_reset(self, _):
        self._src_mx = self._src_my = None
        self._dst_mx = self._dst_my = None
        self._d_lon  = self._d_lat  = 0.0
        self._trans_mean = None
        # Clear time series state and caches
        self._ts_grids              = []
        self._ts_grids_version     += 1
        self._ts_accum_lons         = None
        self._ts_accum_lats         = None
        self._ts_accum_data         = None
        self._accum_orig_merc       = None
        self._trans_accum_cache_key = None
        self._trans_accum_cache     = None
        self._rot_slider.value      = 0.0
        self._fine_lon_slider.value = 0.0
        self._fine_lat_slider.value = 0.0
        # Explicitly zero params — can't rely on throttled slider watchers firing
        self.rotation_deg = 0.0
        self.fine_lon     = 0.0
        self.fine_lat     = 0.0
        self._src_lat_input.value   = None
        self._src_lon_input.value   = None
        self._dst_lat_input.value   = None
        self._dst_lon_input.value   = None
        self._recompute_aep_field()
        # Increment data_version to guarantee both map streams fire on reset
        self.data_version += 1
        self._clear_ws_mod_state()
        self._ts_pane.object = self._empty_ts_plot()
        self._ws_ts_pane.object = self._empty_ws_ts_plot()
        self._ts_status.object = f"*Showing: {self._valid_time}*"
        self._status.object = self._status_md()
        self._stats.object  = self._stats_md()

    # ── Text helpers ──────────────────────────────────────────────────────

    def _qpe_scale(self): return 0.0393701 if self._use_imperial else 1.0
    def _qpe_unit(self):  return "in" if self._use_imperial else "mm"
    def _flow_scale(self): return 35.3147 if self._use_imperial else 1.0
    def _flow_unit(self):  return "ft³/s" if self._use_imperial else "m³/s"
    def _stage_scale(self): return 1.0 if self._use_imperial else 0.3048
    def _stage_unit(self):  return 'ft' if self._use_imperial else 'm'

    def _is_long_duration(self) -> bool:
        """True for periods > 6 hr, which use the stretched 10–300 mm color scale."""
        return int(self._period_select.value.split()[0]) > 6

    def _on_units_change(self, event):
        """Rebuild all displays when the SI/Imperial toggle changes."""
        self._use_imperial = event.new
        self._status.object = self._status_md()
        self._stats.object  = self._stats_md()
        if self._ts_grids:
            self._clear_ws_mod_state()
            self._ts_pane.object    = self._build_ts_plot()
            self._ws_ts_pane.object = self._build_ws_ts_plot()
        if self._last_sac_data is not None:
            self._rebuild_sac_plot()
        self._rebuild_spatial_dmaps()

    def _rebuild_spatial_dmaps(self):
        """
        Recreate Tap streams and DynamicMaps to force a fresh Bokeh figure
        (updates colorbar labels/ticker, title, hover tooltip).  Fresh streams
        initialised at the current click positions ensure the new Bokeh
        figures get clean tap callbacks registered — reusing the old stream
        objects left stale Bokeh linkage that broke click-to-select.
        """
        self._tap_orig  = streams.Tap(x=self._src_mx, y=self._src_my)
        self._tap_trans = streams.Tap(x=self._dst_mx, y=self._dst_my)
        self._orig_dmap  = hv.DynamicMap(
            self._orig_cb, streams=[self._tap_orig, self._orig_stream])
        self._trans_dmap = hv.DynamicMap(
            self._trans_cb, streams=[self._tap_trans, self._trans_stream])
        self._orig_pane.object  = self._orig_dmap
        self._trans_pane.object = self._trans_dmap

    def _rebuild_sac_plot(self):
        """Build/rebuild the SAC streamflow Bokeh figure from stored last-run data."""
        d = self._last_sac_data
        if d is None:
            return
        _COLORS = ['#30a2da', '#fc4f30', '#2ca02c']
        fs = self._flow_scale()
        fu = self._flow_unit()
        runs       = d['runs']
        times_plot = d['times_plot']
        n_event    = d['n_event']
        n_ext      = len(times_plot) - n_event
        title_ext  = f' (+{n_ext}hr zero-precip tail)' if n_ext > 0 else ''
        p = bk_figure(
            x_axis_type='datetime', height=280,
            sizing_mode='stretch_width',
            title=(f'SAC-SMA Simulated Streamflow — {d["ws_name"]}'
                   f' (Gauge {d["basin_id"]}){title_ext}'),
            x_axis_label='Time (UTC)',
            y_axis_label=f'Streamflow ({fu})',
        )
        p.grid.grid_line_alpha = 0.3
        p.grid.grid_line_color = 'gray'
        for color, (label, q_raw) in zip(_COLORS, runs.items()):
            q = q_raw * fs
            src = ColumnDataSource(data={
                'x': times_plot,
                'y': list(q),
                'dt': [t.strftime('%Y-%m-%d %H:%Mz') for t in times_plot],
            })
            peak = float(q.max())
            p.line('x', 'y', source=src, color=color, line_width=2,
                   legend_label=f'{label} [Peak: {peak:.2f} {fu}]')
            p.scatter('x', 'y', source=src, color=color, size=6,
                      line_color='white', line_width=1)
        hover = HoverTool(
            tooltips=[('Time', '@dt'), ('Flow', f'@y{{0.000}} {fu}')],
            mode='mouse')
        p.add_tools(hover)
        p.legend.click_policy = 'hide'
        p.add_layout(p.legend[0], 'right')
        boundary_ms = (times_plot[n_event - 1].timestamp() * 1000) if n_ext > 0 else None
        if n_ext > 0:
            p.add_layout(Span(
                location=boundary_ms, dimension='height',
                line_color='#888888', line_dash='dashed',
                line_width=1.5, line_alpha=0.8,
            ))
            _dummy_r = p.line(
                [times_plot[n_event - 1], times_plot[n_event - 1]],
                [0, 1],
                color='#888888', line_dash='dashed', line_width=1.5, line_alpha=0.8,
                legend_label='End of precip period',
            )
            _dummy_r.visible = False
        self._sac_pane.object = p

        # ── Stage height plot (shared x-axis with flow plot) ──────────────────
        rating = self._rating_curves.get(d['basin_id'])
        if rating is not None:
            ss = self._stage_scale()
            su = self._stage_unit()
            max_q_cfs   = float(rating['discharge_cfs'][-1])
            max_q_disp  = max_q_cfs / (1.0 if self._use_imperial else 1.0)  # cfs either way
            extrap_note = f' — extrapolated above {max_q_disp:.0f} ft³/s' # Note the max discharge in the rating curve
            p_stage = bk_figure(
                x_axis_type='datetime', height=200,
                sizing_mode='stretch_width',
                title=(f'Estimated Stage Height — {d["ws_name"]}'
                       f' (Gauge {d["basin_id"]}){title_ext}'),#{extrap_note}'),
                x_axis_label='Time (UTC)',
                y_axis_label=f'Stage ({su})',
                x_range=p.x_range,
            )
            p_stage.grid.grid_line_alpha = 0.3
            p_stage.grid.grid_line_color = 'gray'
            for color, (label, q_raw) in zip(_COLORS, runs.items()):
                stage_ft   = discharge_to_stage(q_raw, rating)
                stage_disp = stage_ft * ss
                src_s = ColumnDataSource(data={
                    'x':  times_plot,
                    'y':  list(stage_disp),
                    'dt': [t.strftime('%Y-%m-%d %H:%Mz') for t in times_plot],
                })
                peak_s = float(stage_disp.max())
                p_stage.line('x', 'y', source=src_s, color=color, line_width=2,
                             legend_label=f'{label} [Peak: {peak_s:.2f} {su}]')
                p_stage.scatter('x', 'y', source=src_s, color=color, size=6,
                                line_color='white', line_width=1)
            if boundary_ms is not None:
                p_stage.add_layout(Span(
                    location=boundary_ms, dimension='height',
                    line_color='#888888', line_dash='dashed',
                    line_width=1.5, line_alpha=0.8,
                ))
            hover_s = HoverTool(
                tooltips=[('Time', '@dt'), ('Stage', f'@y{{0.00}} {su}')],
                mode='mouse')
            p_stage.add_tools(hover_s)
            p_stage.legend.click_policy = 'hide'
            p_stage.add_layout(p_stage.legend[0], 'right')
            self._sac_stage_pane.object = p_stage
        else:
            self._sac_stage_pane.object = self._empty_stage_plot(
                msg=f'Rating curve not available for gauge {d["basin_id"]}')

        spinup_date = d.get('spinup_date')
        date_str = f' | Initial state: {spinup_date}' if spinup_date else ''
        ext_note = f' | +{n_ext}hr zero-precip tail' if n_ext > 0 else ''
        self._sac_status.object = (
            f'*✓ {len(runs)} scenario(s) computed. Peak flows: '
            + ', '.join(f'{lb}: {float(q_raw.max() * fs):.3f} {fu}'
                        for lb, q_raw in runs.items())
            + date_str + ext_note + '*'
        )

    def _status_md(self) -> str:
        if self._src_mx is not None:
            slon, slat = to_ll(self._src_mx, self._src_my)
            src_str = f"📍 **Source:** {slat:.4f}°N, {slon:.4f}°E"
        else:
            src_str = "📍 **Source:** *not set*"

        if self._dst_mx is not None:
            dlon, dlat = to_ll(self._dst_mx, self._dst_my)
            dst_str = f"🎯 **Dest:** {dlat:.4f}°N, {dlon:.4f}°E"
        else:
            dst_str = "🎯 **Dest:** *not set*"

        if self._d_lon != 0.0 or self._d_lat != 0.0:
            if self._use_imperial:
                ref_lat = 20.5  # approximate Hawaii latitude
                if self._src_my is not None:
                    _, ref_lat = to_ll(self._src_mx, self._src_my)
                lat_mi = self._d_lat * 69.0
                lon_mi = self._d_lon * 69.0 * float(np.cos(np.deg2rad(ref_lat)))
                dist_mi = float(np.sqrt(lat_mi**2 + lon_mi**2))
                delta_str = (f"*Displacement ≈ {dist_mi:.1f} mi "
                             f"({lat_mi:+.1f} N/S, {lon_mi:+.1f} E/W)*")
            else:
                delta_str = f"*Δlat = {self._d_lat:+.4f}°,  Δlon = {self._d_lon:+.4f}°*"
        else:
            delta_str = "*No transposition applied*"

        if self.rotation_deg != 0.0:
            rot_str = f"*Rotation: {self.rotation_deg:+.0f}° CW*"
        else:
            rot_str = "*No rotation applied*"

        fine_parts = []
        if self.fine_lon != 0.0:
            fine_parts.append(f"Δlon {self.fine_lon:+.3f}°")
        if self.fine_lat != 0.0:
            fine_parts.append(f"Δlat {self.fine_lat:+.3f}°")
        fine_str = f"*Fine-tune: {', '.join(fine_parts)}*" if fine_parts else "*No fine-tune applied*"

        return f"{src_str}\n\n{dst_str}\n\n{delta_str}\n\n{rot_str}\n\n{fine_str}"

    def _stats_md(self) -> str:
        if not self._has_watersheds or not self.selected_watershed:
            return "*(Select a watershed to see QPE statistics)*"

        # Use accumulated original data for stats when a period is loaded
        if self._ts_accum_data is not None:
            orig_mean = spatial_mean(self._ts_accum_data, self._ws_mask)
            n = len(self._ts_grids)
            period_label = f"{n}hr accum" if n > 1 else "1hr"
        else:
            orig_mean = spatial_mean(self.data, self._ws_mask)
            period_label = "1hr"
        trans_mean = self._trans_mean

        qs = self._qpe_scale()
        qu = self._qpe_unit()

        def fmt(v):
            return f"**{v * qs:.2f} {qu}**" if v is not None else "*(no data)*"

        ws_name = self.selected_watershed
        lines = [
            f"### {ws_name}",
            f"| | Mean QPE ({period_label} {qu}) |",
            f"|---|---|",
            f"| Original | {fmt(orig_mean)} |",
            f"| Transposed | {fmt(trans_mean)} |",
        ]
        if orig_mean is not None and trans_mean is not None:
            diff = (trans_mean - orig_mean) * qs
            sign = "+" if diff >= 0 else ""
            lines.append(f"| Difference | *{sign}{diff:.2f} {qu}* |")

        return "\n".join(lines)

    # ── Layout ────────────────────────────────────────────────────────────

    def layout(self) -> pn.viewable.Viewable:
        # ── Transposition controls (centre column) ────────────────────────
        ctrl = pn.Column(
            pn.pane.Markdown("## Controls"),
            pn.pane.Markdown(
                "1. Click **left** pane → set source\n"
                "2. Click **right** pane → set destination\n"
                "3. Press **Transpose**\n\n"
                "*Or type coordinates below and click Set.*"
            ),
            pn.Spacer(height=6),
            pn.pane.Markdown("---\n**Source point**"),
            pn.Row(self._src_lat_input, self._src_lon_input),
            self._btn_set_src,
            pn.pane.Markdown("**Destination point**"),
            pn.Row(self._dst_lat_input, self._dst_lon_input),
            self._btn_set_dst,
            pn.Spacer(height=6),
            self._status,
            pn.Spacer(height=10),
            pn.Row(self._btn_transpose, self._btn_reset),
            pn.Spacer(height=8),
            pn.pane.Markdown("---\n**Display units**"),
            self._units_toggle,
            pn.Spacer(height=15),
            pn.pane.Markdown(
                f"📂 `{self.nc_path}`",
                styles={"font-size": "11px", "color": "#888"},
            ),
            width=310,
            styles={
                "padding": "14px",
                "background": "#f5f5f5",
                "border-radius": "8px",
            },
        )

        # ── Fine-tune + watershed box (right column, beside Controls) ─────
        ws_section = pn.Column(
            pn.pane.Markdown("**Watershed**"),
            self._ws_select,
            self._stats,
        ) if self._has_watersheds else pn.pane.Markdown("")

        fine_ws_box = pn.Column(
            pn.pane.Markdown("**Rotation** *(° CW)*"),
            self._rot_slider,
            pn.Spacer(height=6),
            pn.pane.Markdown("**Fine-tune position** *(nudge grid)*"),
            self._fine_lon_slider,
            self._fine_lat_slider,
            pn.Spacer(height=8),
            ws_section,
            width=310,
            styles={
                "padding": "14px",
                "background": "#f5f5f5",
                "border-radius": "8px",
            },
        )

        # ── Time-period controls ──────────────────────────────────────────
        ts_controls = pn.Row(
            pn.Column(
                pn.pane.Markdown("**End date/time (UTC)**"),
                self._dt_picker,
                width=240,
            ),
            pn.Column(
                pn.pane.Markdown("**Period**"),
                pn.Row(self._period_select, self._btn_load_ts),
            ),
            pn.Column(
                pn.pane.Markdown("**Status**"),
                self._ts_status,
                sizing_mode="stretch_width",
            ),
            styles={
                "background": "#eef2f7",
                "border-radius": "8px",
                "padding": "10px 14px",
                "align-items": "flex-end",
            },
        )

        # ── Temporal modification controls (beneath watershed TS) ─────────
        mod_controls = pn.Column(
            pn.pane.Markdown("#### Temporal Pattern Modification"),
            pn.pane.Markdown(
                "Click a data point on the Watershed Areal Mean plot, then use "
                "the slider to adjust that hour. The total precipitation is conserved."
            ),
            pn.Row(
                pn.Column(
                    pn.pane.Markdown("**Redistribution Method**"),
                    self._ws_mod_method,
                    self._ws_mod_sigma,
                    self._ws_mod_window,
                    width=180,
                ),
                pn.Column(
                    self._ws_mod_info,
                    self._ws_mod_slider,
                    self._btn_reset_mod,
                    sizing_mode="stretch_width",
                ),
            ),
            sizing_mode="stretch_width",
            styles={
                "background": "#f0f4f8",
                "border-radius": "8px",
                "padding": "10px 14px",
                "margin-top": "6px",
            },
        ) if self._has_watersheds else pn.pane.Markdown("")

        # ── SAC-SMA section ──────────────────────────────────────────────────
        sac_controls = pn.Column(
            pn.pane.Markdown(
                '#### SAC-SMA Hydrological Model\n'
                'Runs a full spin-up from 2021 then simulates streamflow for the '
                f'selected watershed. Available for all {len(_AVAILABLE_BASIN_IDS)} '
                'calibrated Hawaii basins.'
            ),
            pn.Row(
                self._btn_run_sac,
                self._sac_status,
                align='center',
            ),
            self._sac_progress,
            self._sac_pane,
            self._sac_stage_pane,
            sizing_mode='stretch_width',
            styles={
                'background': '#f0f4f8',
                'border-radius': '8px',
                'padding': '10px 14px',
                'margin-top': '6px',
            },
        ) if _SAC_AVAILABLE else pn.pane.Markdown('')

        # ── Time series plots (below maps, same width as both maps) ──────
        ts_section = pn.Column(
            pn.pane.Markdown("### Point Time Series"),
            self._ts_pane,
            pn.pane.Markdown("### Watershed Areal Mean Time Series"),
            self._ws_ts_pane,
            mod_controls,
            sac_controls,
            width=2 * PLOT_W,
            styles={
                "background": "#fff",
                "border-radius": "8px",
                "padding": "8px 14px",
            },
        )

        # ── Nested layout: maps + ts on the left, panels on the right ─────
        maps_and_ts = pn.Column(
            pn.Row(self._orig_pane, self._trans_pane, margin=0),
            ts_section,
            margin=0,
        )

        # ── Transposition method controls — near the top of the page ──
        method_controls = pn.Row(
            pn.Column(
                pn.pane.Markdown("**Transposition Method**"),
                self._trans_method_toggle,
                width=360,
            ),
            pn.Column(
                pn.pane.Markdown("**Spatial Plot Display**"),
                self._ari_display_toggle,
                width=280,
            ),
            pn.Column(
                pn.pane.Markdown("**Status**"),
                self._aep_status,
                sizing_mode="stretch_width",
            ),
            styles={
                "background": "#fff3e0",
                "border-radius": "8px",
                "padding": "10px 14px",
                "align-items": "flex-start",
                "gap": "24px",
            },
        )

        return pn.template.BootstrapTemplate(
            title="MRMS QPE Transposition Dashboard",
            main=[
                pn.Column(
                    method_controls,
                    ts_controls,
                    pn.Row(maps_and_ts, ctrl, fine_ws_box, margin=0,
                           styles={"gap": "8px", "align-items": "flex-start"}),
                    margin=0,
                    styles={"display": "flex", "flex-direction": "column", "gap": "8px"},
                )
            ],
        )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="MRMS QPE Transposition Dashboard")
    parser.add_argument(
        "nc_path",
        nargs="?",
        default=DEFAULT_NC,
        help="Path to MRMS NetCDF file",
    )
    parser.add_argument(
        "shp_path",
        nargs="?",
        default=DEFAULT_PARQUET,
        help="Path to watershed parquet or shapefile (optional)",
    )
    known, _ = parser.parse_known_args()   # ignore panel serve's own flags
    return known


args = parse_args()
dashboard = MRMSTransposeDashboard(args.nc_path, shp_path=args.shp_path)
dashboard.layout().servable()
