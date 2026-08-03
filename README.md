# CIROH Flash Flood Scenario Tool

An interactive dashboard for exploring **MRMS (Multi-Radar/Multi-Sensor) QPE** precipitation grids over Hawaii and assessing the flood potential of alternative storm scenarios using the **SAC-SMA hydrological model**.

Developed by RTI International as part of the CIROH Flash Flood Forecasting project.

---

## Overview

Flash flood forecasting depends not only on observed precipitation but also on *what-if* scenarios: What if the same storm hit a different watershed? What if the temporal pattern of rainfall had peaked earlier or later?

This tool provides a browser-based dashboard that lets analysts:

1. **Load and visualize** MRMS 1-hour QPE grids for any time period (1, 3, 6, 12, 24, or 48 hours)
2. **Transpose** a storm spatially — shift the precipitation field to a new location and compare the original vs. transposed grids side-by-side
3. **Rotate and fine-tune** the transposed storm pattern
4. **Compute areal mean QPE** over calibrated USGS watershed boundaries for each scenario
5. **Modify the temporal pattern** of watershed precipitation (redistribute hourly totals while preserving the storm total)
6. **Run the SAC-SMA + Unit Hydrograph model** for each precipitation scenario and compare simulated streamflow hydrographs

---

## Repository Structure

```
code/
├── dashboard_mrms.py       # Main dashboard application (Panel + HoloViews)
├── sac_worker.py           # SAC-SMA + UHG worker (runs in hydro-model env)
├── config.py               # Shared configuration (paths, domain bounds)
├── download_mrms_hawaii.py # MRMS QPE data downloader for Hawaii
├── example_mrms_download.py
├── verify_availability.py  # Check MRMS data availability
├── verify_hawaii_http.py
├── requirements.txt        # Python dependencies
├── dashboard_test.ipynb    # Notebook launcher for the dashboard
└── mrms_processing.ipynb   # Exploratory MRMS data processing notebook
```

---

## Requirements

### Python environments

The project uses **two separate Python environments**:

| Environment | Purpose | Key packages |
|---|---|---|
| `dashboard` (default) | Dashboard, data processing | `panel`, `holoviews`, `bokeh`, `xarray`, `geopandas`, `pyproj`, `rasterio`, `scipy` |
| `hydro-model` | SAC-SMA model execution | `pysacsnow` |

The SAC-SMA model runs in a subprocess under the `hydro-model` environment so that the heavier geospatial stack and the hydrology stack remain isolated.

### Install dashboard dependencies

```bash
pip install -r requirements.txt
```

Or with conda:

```bash
conda create -n dashboard python=3.11
conda activate dashboard
pip install -r requirements.txt
```

---

## Configuration

Several **absolute paths are hardcoded** in `dashboard_mrms.py` and `sac_worker.py` and must be updated to match your local environment before running:

| Variable | File | Description |
|---|---|---|
| `_HYDRO_MODEL_PY` | `dashboard_mrms.py` | Path to `python.exe` in the `hydro-model` conda env |
| `HAWAII_RESULTS_DIR` | `dashboard_mrms.py`, `sac_worker.py` | Directory containing calibration outputs and configurations |
| `HYDRO_MODEL_CONFIG_DIR` | `dashboard_mrms.py`, `sac_worker.py` | Directory containing SAC/UHG YAML control files and forcing data |
| `DEFAULT_NC` | `dashboard_mrms.py` | Default MRMS NetCDF file to display on startup |
| `DEFAULT_COASTLINE_SHP` | `dashboard_mrms.py` | Path to Hawaii coastline shapefile |

### SAC-SMA calibration outputs

The hydrological model integration expects a directory structure like:

```
Hawaii results/
├── configurations/
│   ├── sac/
│   │   ├── sac_params_{basin_id}_cal.json    # Calibrated parameters
│   │   └── sac.control.{basin_id}.yaml       # SAC control template
│   └── unithg/
│       ├── uhg_params_{basin_id}.json
│       └── uhg.control.{basin_id}.yaml
├── output/
│   └── basin_{basin_id}/
│       └── best_params_{basin_id}_kge_q90.csv
├── usgs_basin_geometry_with_source_all.parquet   # Watershed geometries
└── usgs_lt500_min2014-2024.parquet               # USGS gauge locations
```

---

## Data

### MRMS QPE files

MRMS 1-hour QPE NetCDF files for Hawaii are expected in the format:

```
mrms_qpe_1hr_hawaii_YYYYMMDD_HHMM.nc
```

Use `download_mrms_hawaii.py` to fetch files for a target period:

```bash
python download_mrms_hawaii.py --start 2026-03-19 --end 2026-03-20 --output ./data/MRMS/
```

---

## Running the Dashboard

### Option 1 — Panel serve (recommended)

```bash
panel serve code/dashboard_mrms.py --show
```

With a specific MRMS file and watershed shapefile:

```bash
panel serve code/dashboard_mrms.py --show --args path/to/mrms_file.nc path/to/watersheds.parquet
```

### Option 2 — Jupyter notebook

Open `dashboard_test.ipynb` and run the first cell. The dashboard will launch in a browser tab at `http://localhost:5006`.

---

## Dashboard Workflow

```
┌─────────────────────────────────────────────────────────┐
│  1. Select end date/time and accumulation period         │
│     (1, 3, 6, 12, 24, or 48 hr), then click Load        │
│                                                          │
│  2. Click a point on the LEFT map  → SOURCE             │
│     Click a point on the RIGHT map → DESTINATION        │
│                                                          │
│  3. Optionally adjust rotation (° CW) and fine-tune     │
│     Δlon / Δlat sliders                                  │
│                                                          │
│  4. Click ▶ Transpose — storm shifts to new location    │
│                                                          │
│  5. Select a watershed from the dropdown to compute      │
│     areal mean QPE for both the original and transposed  │
│     grids                                                │
│                                                          │
│  6. Click a point on the Watershed Areal Mean chart to   │
│     select an hour, then use the slider to modify that   │
│     hour's rainfall (total is conserved)                 │
│                                                          │
│  7. Click ▶ Run SAC-SMA Model to simulate streamflow    │
│     for all active scenarios                             │
└─────────────────────────────────────────────────────────┘
```

### Map panes

| Pane | Content |
|---|---|
| **Original** | Observed MRMS QPE; click to set source point |
| **Transposed** | Spatially shifted/rotated QPE; click to set destination point |

Both panes overlay watershed boundaries, USGS gauge locations, and the selected watershed polygon.

### Units

Toggle between **SI** (mm, m³/s) and **Imperial** (in, ft³/s) using the units toggle in the Controls panel.

---

## SAC-SMA Hydrological Model

The SAC-SMA (Sacramento Soil Moisture Accounting) model coupled with a Unit Hydrograph is used to convert watershed areal mean precipitation into streamflow. The model is:

- Calibrated for each basin using historical USGS streamflow records
- Run in a subprocess under the `hydro-model` conda environment via `sac_worker.py`
- Initialized with a spin-up from 2021 using historical forcing data
- Extended by 12 zero-precipitation hours after the event period to capture the recession limb

Up to three scenarios are compared simultaneously:

| Scenario | Precipitation input |
|---|---|
| **Original** | Observed MRMS QPE over the selected watershed |
| **Transposed** | QPE from the transposed storm over the selected watershed |
| **Modified** | Temporally redistributed QPE (if temporal modifications have been made) |

---

## License

See [LICENSE](../LICENSE) for details.

---

## Acknowledgments

This work was supported by the Cooperative Institute for Research to Operations in Hydrology (CIROH).
