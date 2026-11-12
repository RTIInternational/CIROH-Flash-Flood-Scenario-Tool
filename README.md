# CIROH Flash Flood Scenario Tool

An interactive dashboard for exploring **MRMS (Multi-Radar/Multi-Sensor) QPE** precipitation grids across regional domains (e.g., Hawaii, Vermont, North Carolina) and assessing the flood potential of alternative storm scenarios using the **SAC-SMA hydrological model**.

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
7. **Quantify uncertainty with an ensemble** *(ensemble dashboard)* — run hundreds of model members that vary initial soil saturation, rainfall amount, soil parameters, and unit-hydrograph timing, and view the spread as percentile bands (see [Ensemble (Uncertainty) Mode](#ensemble-uncertainty-mode))

---

## Repository Structure

```
CIROH Flash Flooding/          ← project root
├── .env.example               # Template for environment configuration
│
├── regions/                   # Self-contained regional assets (GIS, forcing, parameters)
│   ├── template/              # Starter template for onboarding new states (e.g., VT, NC)
│   │   └── config.json
│   └── hawaii/
│       ├── config.json        # Hawaii domain profile (bounds, ref_lat, MRMS product, paths)
│       ├── GIS/               # Spatial boundary shapefiles, basins, and USGS gauge locations
│       │   ├── Coastline.shp
│       │   ├── usgs_basin_geometry_with_source_HI.shp
│       │   └── usgs_lt500_min2014-2024.parquet
│       └── hydro_model/       # SAC-SMA + Unit Hydrograph hydrologic assets
│           ├── forcing/       # Hourly forcing CSVs: forcing.sac.<basin>.csv
│           ├── parameters/    # Calibrated parameter JSONs (sac/ and unithg/)
│           │   ├── sac/       # Files that end in "hf.json" are for hydrofabric scale
│           │   └── unithg/
│           ├── control_templates/ # sac.control.template.yaml & uhg.control.template.yaml
│           └── calibration_results/ # outputs/ (best_params_*.csv) & diagnostics/
│
├── precip_data/               # Gridded precipitation datasets
│   └── MRMS/                  # 1-hour MRMS QPE NetCDF files
│
├── src/
│   ├── dashboard.py           # Unified dashboard application (deterministic + ensemble)
│   ├── sac_worker.py          # Unified SAC-SMA + UHG worker (deterministic + ensemble)
│   ├── ens_core.py            # Latin hypercube sampling and ensemble engine (numpy only)
│   ├── ens_ui.py              # Ensemble controls and plots (Panel / Bokeh)
│   ├── region_config.py       # Region profile and environment variable loader
│   ├── download_mrms.py       # Generalized multi-region MRMS QPE downloader
│   └── utils.py               # USGS NWIS rating curves and discharge-to-stage
│
├── notebooks/                 # Organized workflows and calibration
│   ├── exploration/           # Interactive dashboard launcher & data exploration
│   │   ├── dashboard_test.ipynb
│   │   ├── get_top_events.ipynb
│   │   ├── hydrofabric_test.ipynb
│   │   └── mrms_processing.ipynb
│   └── calibration/           # SAC-SMA calibration notebooks (01 to 06)
│       ├── 01_setup_config_files.ipynb
│       ├── 05_optimize_sac_all_basins.ipynb
│       └── 06_view_opt_results_all_basins.ipynb
│
├── tests/                     # Unit and integration test scripts
│   ├── test_worker_ens.py     # Worker regression and ensemble validation
│   ├── test_inplace.py        # In-place parameter mutation checks
│   └── test_nhru.py           # Multi-HRU execution test
│
└── requirements.txt           # Dashboard environment dependencies
```

> **\* Large file note:** `usgs_basin_geometry_with_source_all.parquet` (125 MB) exceeds
> GitHub's per-file limit and is git-ignored. If present in `regions/hawaii/GIS/`,
> the dashboard will load it; if absent, the tool automatically falls back to the shapefile in
> `regions/hawaii/GIS/usgs_basin_geometry_with_source_HI.shp` so the watershed overlay
> continues to work out-of-the-box.

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
conda create -n scenario_tool python=3.11
conda activate scenario_tool
pip install -r requirements.txt
```

---

## Configuration & Environment Setup

Paths and environment settings are dynamically managed and do not require editing Python source files.

### 1. Configure `.env`

Copy `.env.example` to `.env` in the repository root:

```bash
cp .env.example .env
```

Set the path to your `hydro-model` Python interpreter:

```ini
HYDRO_MODEL_PY=C:\path\to\envs\hydro-model\python.exe
ATLAS14_NC=C:\path\to\climatology\a14_hawaii.nc
ACTIVE_REGION=hawaii
```

If not set, the tool will attempt to auto-discover standard conda environment locations before falling back to `sys.executable`.

### 2. Region Profiles

Regional bounds, MRMS products, and model directories are stored as modular JSON profiles in `regions/<region_name>/config.json`. 

To run the tool for Hawaii or switch to a new region (e.g., Vermont or North Carolina), specify `--region <name>` when launching the dashboard, or set `ACTIVE_REGION=<name>` in `.env`.

### Standardized Regional Asset Structure

The hydrological model and dashboard dynamically load assets from a self-contained `regions/<state>/` directory:

```
regions/<state>/
├── config.json                     # Region bounds, ref_lat, MRMS product, and path mappings
├── GIS/                            # Boundaries (coastlines/state borders), basins, and gauges
│   ├── Coastline.shp               # (or state/county borders)
│   ├── usgs_basin_geometry_with_source_HI.shp
│   └── usgs_lt500_min2014-2024.parquet
└── hydro_model/                    # Hydrological modeling inputs and calibration
    ├── forcing/                    # Hourly forcing CSVs: forcing.sac.<basin>.csv
    ├── parameters/                 # Calibrated parameter JSONs (sac/ and unithg/)
    │   ├── sac/
    │   └── unithg/
    ├── control_templates/          # sac.control.template.yaml & uhg.control.template.yaml
    └── calibration_results/        # outputs/ (best_params_*.csv) & diagnostics/
```

---

## Data

### MRMS QPE files

MRMS 1-hour QPE NetCDF files are saved into `precip_data/MRMS/` using `download_mrms.py`:

```bash
# Download for Hawaii (default)
python src/download_mrms.py --region hawaii --start 2026-03-19 --end 2026-03-20

# Download for another configured region
python src/download_mrms.py --region template --start 2026-03-19 --end 2026-03-20
```

---

## Running the Dashboard

### Option 1 — Panel serve (recommended)

Launch the unified dashboard:

```bash
panel serve src/dashboard.py --show
```

To target a specific region profile:

```bash
panel serve src/dashboard.py --show --args --region hawaii
```

Or pass a specific MRMS file and watershed shapefile/parquet:

```bash
panel serve src/dashboard.py --show --args path/to/mrms_file.nc path/to/watersheds.parquet
```

### Option 2 — Jupyter notebook

Open `notebooks/exploration/dashboard_test.ipynb` and run the first cell. The dashboard will launch in a browser tab at `http://localhost:5006`.

---

## Onboarding a New Region (e.g., Vermont or North Carolina)

To adapt the tool to a new state or geographic domain:

1. **Create the State Folder Structure:**
   ```bash
   mkdir -p regions/<state>/GIS
   mkdir -p regions/<state>/hydro_model/{forcing,parameters/sac,parameters/unithg,control_templates,calibration_results}
   ```
2. **Add Watershed & Gauge Data:**
   - Place watershed boundary shapefiles or GeoParquet under `regions/<state>/GIS/`.
   - Place USGS gauge locations under `regions/<state>/GIS/`.
   - (Optional) Place regional boundary outlines (e.g., state borders or coastlines) under `regions/<state>/GIS/`.
3. **Add Forcing and Model Parameters:**
   - Drop hourly precipitation and PET CSVs (`forcing.sac.<basin_id>.csv`) into `regions/<state>/hydro_model/forcing/`.
   - Place calibrated parameter JSON files in `regions/<state>/hydro_model/parameters/sac/` and `parameters/unithg/`.
   - Put your SAC control YAML template in `regions/<state>/hydro_model/control_templates/sac.control.template.yaml`.
4. **Create a Region Profile:**
   - Copy `regions/template/config.json` to `regions/<state>/config.json`.
   - Update spatial bounds (`lat_min`, `lat_max`, `lon_min`, `lon_max`), reference latitude (`ref_lat`), center, and path mappings.
5. **Download Precipitation Climatology:**
   - Place the NOAA Atlas 14 / Atlas 15 NetCDF frequency grid for your domain and configure its path in `regions/<state>/config.json` or `.env`.
6. **Launch:**
   ```bash
   panel serve code/dashboard.py --show --args --region <state>
   ```
   - (Optional) Place regional boundary outlines (e.g., state borders or coastlines) under `data/GIS/`.
3. **Add Forcing and Model Parameters:**
   - Drop hourly precipitation and PET CSVs (`forcing.sac.<basin_id>.csv`) into `regions/<state>/forcing/`.
   - Place calibrated parameter JSON files in `regions/<state>/parameters/sac/` and `parameters/unithg/`.
   - Put your SAC control YAML template in `regions/<state>/templates/sac.control.template.yaml`.
4. **Create a Region Profile:**
   - Copy `configs/regions/conus_sample.json` to `configs/regions/<state>.json`.
   - Update spatial bounds (`lat_min`, `lat_max`, `lon_min`, `lon_max`), reference latitude (`ref_lat`), center, and path mappings.
5. **Download Precipitation Climatology:**
   - Place the NOAA Atlas 14 / Atlas 15 NetCDF frequency grid for your domain and configure its path in `configs/regions/<state>.json` or `.env`.
6. **Launch:**
   ```bash
   panel serve code/dashboard.py --show --args --region <state>
   ```

---

## Dashboard Workflow

```
┌─────────────────────────────────────────────────────────┐
│  1. Select end date/time and accumulation period         │
│     (1, 3, 6, 12, 24, or 48 hr), then click Load        │
│                                                          │
│  2. Choose a Transposition Method                        │
│     (Simple Translation or AEP Transposition)            │
│                                                          │
│  3. Click a point on the LEFT map  → SOURCE             │
│     Click a point on the RIGHT map → DESTINATION        │
│                                                          │
│  4. Optionally adjust rotation (° CW, Simple mode)      │
│     and fine-tune Δlon / Δlat sliders                    │
│                                                          │
│  5. Click ▶ Transpose — storm shifts to new location    │
│                                                          │
│  6. Select a watershed from the dropdown to compute      │
│     areal mean QPE for both the original and transposed  │
│     grids                                                │
│                                                          │
│  7. Click a point on the Watershed Areal Mean chart to   │
│     select an hour, then use the slider to modify that   │
│     hour's rainfall (total is conserved)                 │
│                                                          │
│  8. (Ensemble dashboard) Set the Ensemble settings:      │
│     soil saturation range, rainfall multiplier,          │
│     members. See Ensemble (Uncertainty) Mode             │
│                                                          │
│  9. Click ▶ Run SAC-SMA Model to simulate streamflow    │
│     for all active scenarios (plus the ensemble, if on)  │
└─────────────────────────────────────────────────────────┘
```

### Transposition Methods

The dashboard provides two transposition techniques to modify the spatial precipitation pattern:

1. **Simple Translation**
   - Directly translates the precipitation field by $(\Delta\text{lon}, \Delta\text{lat})$ without scaling or climatological adjustments.
   - Source and destination points may be placed anywhere across the grid (land or ocean).
   - Supports clockwise/counter-clockwise **rotation** (° CW) and fine-tune nudging.

2. **AEP Transposition (Atlas 14)** — *Annual Exceedance Probability*
   - Full-field climatological transposition where every wet grid cell is scaled independently based on its own local climatology rather than a single point.
   - Each cell's observed accumulation is converted to its equivalent local recurrence interval (ARI / AEP) via log-linear interpolation and extrapolation across Atlas 14 curves.
   - That recurrence interval is then evaluated against the Atlas 14 climatology at the cell's transposed target coordinates to calculate the adjusted precipitation amount ($P_{\text{transposed}} = f_{\text{dest}}(\text{AEP}_{\text{src}})$).
   - Wet cells lacking Atlas 14 land coverage at either the source or destination are blanked (NaN); dry cells remain zero. Rotation is disabled during AEP mode.

### Map panes

| Pane | Content |
|---|---|
| **Original** | Observed MRMS QPE; click to set source point |
| **Transposed** | Spatially shifted/rotated QPE; click to set destination point |

Both panes overlay watershed boundaries, USGS gauge locations, and the selected watershed polygon. Use the **Spatial Plot Display** toggle to switch between showing precipitation depth and the Atlas 14 **Recurrence Interval** (return period in years, from 1 to 200+ yr).

### Units

Toggle between **SI** (mm, m³/s) and **Imperial** (in, ft³/s) using the units toggle in the Controls panel.

---

## SAC-SMA Hydrological Model

The SAC-SMA (Sacramento Soil Moisture Accounting) model coupled with a Unit Hydrograph is used to convert watershed areal mean precipitation into streamflow. The model is:

- Calibrated for each basin using historical USGS streamflow records
- Run in a subprocess under the `hydro-model` conda environment via `sac_worker.py` (`sac_worker_ens.py` in the ensemble dashboard)
- Initialized with a spin-up from 2021 using historical forcing data
- Extended by 12 zero-precipitation hours after the event period to capture the recession limb

Up to three scenarios are compared simultaneously:

| Scenario | Precipitation input |
|---|---|
| **Original** | Observed MRMS QPE over the selected watershed |
| **Transposed** | QPE from the transposed storm over the selected watershed |
| **Modified** | Temporally redistributed QPE (if temporal modifications have been made) |

---

## Ensemble (Uncertainty) Mode

A single SAC-SMA run gives one hydrograph per scenario. Ensemble mode runs many **members** of each scenario — each with slightly different starting soil moisture, rainfall amount, and (optionally) soil parameters and unit-hydrograph timing — and shows how widely the simulated streamflow and stage can vary. It is available in `dashboard_mrms_ens.py` (served by `sac_worker_ens.py`).

> **How to read the bands.** The ranges are chosen by the user. The bands show the spread produced by those assumptions: a *what-if envelope*, not calibrated forecast probabilities.

### Ensemble settings

The **Ensemble (uncertainty) settings** panel sits above the **Run SAC-SMA Model** button. Each control has a one-line description in the dashboard.

| Control | Default | What it does |
|---|---|---|
| **Run ensemble** | on | Off = one deterministic run per scenario, no bands |
| **Initial soil saturation (%)** | 40–100 | Range of starting tension-water saturation; each member draws one value |
| **Precipitation multiplier** | 0.8–1.2 | Per-member factor applied to every hour of rainfall (1.0 = unchanged). Stands in for error in the rainfall amount, such as QPE bias |
| **Members** | 300 (50–1000) | Model runs per scenario |
| **Soil parameters ±20%** | on | Multiplies six calibrated parameters — `uztwm`, `uzfwm`, `lztwm`, `lzfpm`, `lzfsm`, `rexp` — by a per-member factor between 0.8 and 1.2 |
| **Unit-hydrograph timing ±1 h** | off | Shifts the unit hydrograph up to 1 h earlier or later per member, preserving its volume |

### Initial soil saturation

"Saturation" means **tension-water fill**: `(UZTWC + LZTWC) / (UZTWM + LZTWM)`. Free-water stores are not part of the percentage, so 100% means the tension-water stores are full, not that the soil can absorb no more.

The dashboard reports the **model spin-up state at event start** — the saturation the deterministic run begins from — and marks it with a dashed line on the saturation plot. If the sampled range does not include that value, every member is wetter or drier than the deterministic run, and the ensemble median will sit away from it.

Member initial states are scaled from the spin-up state:

- **Tension water** (`UZTWC`, `LZTWC`): both zones are scaled by one factor chosen so the combined saturation equals the member's sampled value. A zone that reaches capacity stays full and the other zone takes the remainder.
- **Free water** (`UZFWC`, `LZFPC`, `LZFSC`) and `ADIMC`: scaled by sampled saturation ÷ spin-up saturation.
- **Capacity:** every store is capped at the member's own capacity, so perturbed parameters cannot produce a state above capacity.
- A member whose saturation equals the spin-up saturation, with nothing else perturbed, starts from exactly the spin-up state.

### How members are generated

- **Latin hypercube sampling.** Each varied quantity's range is split into one slice per member, each member gets one random point per slice, and the slices are shuffled independently. This gives even coverage of every range; the quantities are independent and uniform within their ranges.
- **Fixed seed (1),** so a run is reproducible. The **same draws are used for every scenario** (Original, Transposed, Modified), so differences between scenarios are not sampling noise.
- **Parameter constraints.** The ordering `lzpk < lzsk < uzk` is enforced only where the calibrated parameter set already satisfies it (calibrated basin 16010000 has `lzsk > uzk`, so that pair is skipped there). `uzk` is capped at 0.99 and `pctim + adimp` is kept below 1.
- **Running a member.** The model is reset, the member's parameters and initial states are written in place, and the event is stepped through; the unit hydrograph then converts runoff to streamflow. An ensemble with nothing perturbed reproduces the deterministic run.

### Outputs

Shown below the deterministic hydrograph and stage plots:

- **Summary table** — member count, spin-up saturation, and the sampled ranges; for each scenario, the deterministic peak, the 5th / median / 95th percentiles of the member peak flows, and the 5–95% time-to-peak range (hours after the start of the period).
- **Ensemble streamflow plot** — 5–95% band, 25–75% band, median (solid), and the deterministic run (dashed) for each scenario. Click a legend entry to hide it.
- **Ensemble stage plot** — the same bands converted through the gauge rating curve (shown when a rating curve is available).
- **Peak flow vs. initial soil saturation** and **peak flow vs. precipitation multiplier** — one point per member. A plot appears only if its quantity was varied.

In test runs on basin 16010000, the dependence of peak flow on initial saturation weakened as the storm total grew, consistent with soil storage being small relative to a large storm. The scatter plots show this directly for your own storm.

### Run time

On the development machine the per-member loop took about 3 ms per member (500 members × 3 scenarios ran in about 6 s). The dashboard scales its subprocess timeout with the ensemble size: 180 s + 0.05 s × members × scenarios.

### Worker interface

`sac_worker_ens.py` reads JSON from stdin and writes JSON to stdout. Without an `ensemble` key it behaves exactly like the original worker. Every key in the `ensemble` block is optional except `n`:

```jsonc
"ensemble": {
  "n": 300, "seed": 1,
  "sat": [0.4, 1.0],                       // tension-water saturation range (omit = keep spin-up saturation)
  "ppt_mult": [0.8, 1.2],                  // precipitation multiplier range
  "mult": {"uztwm": [0.8, 1.2]},           // per-parameter multiplier ranges (allow-listed parameters only)
  "uh_shift": [-1, 1],                     // unit-hydrograph shift, hours
  "state_method": "scaled"                 // "scaled" (default) or "legacy" (earlier rule, comparison only)
}
```

The response adds `_ensemble` (and `_worker_version`):

| Key | Contents |
|---|---|
| `meta` | `n`, `seed`, `percentiles`, `sat_spinup`, `state_method`, `config`, `core`, `worker` |
| `members` | Per-member `saturation`, `saturation_achieved`, `ppt_mult`, `uh_shift_h`, and each perturbed parameter |
| `scenarios[label]` | `pct` (5/25/50/75/95 percentile hydrographs), `peak`, `tpeak_h`, `volume_m3` per member, and `traces` (first 25 members) |

If the ensemble fails, the deterministic hydrographs are still returned along with `_ensemble_error`.

### Developer notes

- **In-place parameter edits.** Members are run by editing the live model's parameter arrays, which is only safe if the edit matches a freshly built model. `test_inplace.py` checked this for 13 parameters (`uztwm`, `uzfwm`, `lztwm`, `lzfpm`, `lzfsm`, `adimp`, `uzk`, `lzpk`, `lzsk`, `zperc`, `rexp`, `pctim`, `pfree`) on basin 16010000 with pysacsnow 0.14.1; they are listed in `_INPLACE_SAFE` in `sac_worker_ens.py`, and any other parameter is refused. `pxadj` was not tested — use the precipitation multiplier instead.
- **Unit-hydrograph shifts** edit the private pysacsnow attribute `_uh_ords_padded` (`_UHG_INPLACE_ATTR`); editing `uh_ords` has no effect on the model. Re-run `test_inplace.py --uhg-only` after upgrading pysacsnow.
- **Adding a parameter to the ensemble:** run `test_inplace.py`, add it to `_INPLACE_SAFE`, then add it to `SOIL_PARAMS` in `ens_ui.py`.
- **Defaults are starting values.** The ±20% parameter range and the 0.8–1.2 rainfall multiplier are judgement-based defaults, not derived from calibration uncertainty. Change them in `ens_ui.py` (`SOIL_RANGE`, the slider defaults).
- **One HRU per member** (`nhru > 1`) was explored in `test_nhru.py` but not adopted; the per-member loop is fast enough.

### Limitations

- The ensemble perturbs basin-mean rainfall, initial state, parameters, and unit-hydrograph timing. It does not represent within-basin rainfall variability or uncertainty in the transposition itself.
- Quantities are sampled independently and uniformly; parameter correlations from calibration are not represented.
- The in-place parameter checks were run on one basin (16010000).

### Core files

| File | Role |
|---|---|
| `dashboard.py` | Unified dashboard application supporting deterministic and ensemble modes |
| `sac_worker.py` | Unified worker supporting deterministic and optional ensemble execution |
| `ens_core.py` | Latin hypercube sampling, saturation-to-state scaling, ensemble driver (numpy only) |
| `ens_ui.py` | Ensemble controls, summary table, band and scatter plots |
| `region_config.py` | Dynamic path resolution and region profile loader |
| `download_mrms.py` | Region-agnostic MRMS QPE data downloader |

### Testing

Run test scripts from the repository root in the `hydro-model` environment:

```bash
# Run worker ensemble and deterministic end-to-end regression test:
python tests/test_worker_ens.py 16010000 50

# Check in-place parameter compatibility:
python tests/test_inplace.py
```

| Script | Location | Verifies |
|---|---|---|
| `test_worker_ens.py` | `tests/` | End-to-end check of `sac_worker.py`, run as a subprocess like the dashboard does: bands are ordered, deterministic hydrographs are unchanged by the ensemble, and no-perturbation reproduces deterministic runs. |
| `test_inplace.py` | `tests/` | Checks which parameters and unit-hydrograph ordinates can be edited in place and match a freshly built model. |
| `test_nhru.py` | `tests/` | Exploration of running members as separate HRUs, plus per-member loop timing. |

---

## License

See [LICENSE](../LICENSE) for details.

---

## Acknowledgments

This work was supported by the Cooperative Institute for Research to Operations in Hydrology (CIROH).
