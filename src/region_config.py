"""
Region and environment configuration loader for CIROH Flash Flood Scenario Tool.

Handles:
- Dynamic project root resolution (no hardcoded absolute drive letters)
- .env file loading (standard dotenv or built-in lightweight fallback)
- Region profile discovery from configs/regions/<name>.json
- Automatic discovery of the hydro-model Python environment
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional

# Resolve project root (parent of src/ folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env(env_path: Optional[Path] = None) -> Dict[str, str]:
    """
    Load environment variables from .env if present.
    Uses python-dotenv if installed, otherwise uses a robust built-in parser.
    """
    if env_path is None:
        env_path = PROJECT_ROOT / ".env"

    loaded = {}
    if not env_path.exists():
        return loaded

    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
        return dict(os.environ)
    except ImportError:
        pass

    # Built-in lightweight .env parser
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k not in os.environ:
                    os.environ[k] = v
                loaded[k] = v
    except Exception as exc:
        print(f"  [warn] Could not read .env: {exc}")

    return loaded


# Load .env upon module import
load_env()


def find_hydro_model_python() -> Path:
    """
    Locate the python executable for the 'hydro-model' conda or virtual environment.
    Checks:
    1. HYDRO_MODEL_PY environment variable (from .env or shell)
    2. Domenic's known conda environment path
    3. Standard user .conda/envs/hydro-model locations
    4. Fallback to sys.executable
    """
    env_val = os.environ.get("HYDRO_MODEL_PY")
    if env_val:
        p = Path(env_val)
        if p.exists():
            return p

    # Standard locations
    home = Path.home()
    candidates = [
        home / ".conda" / "envs" / "hydro-model" / "python.exe",
        home / ".conda" / "envs" / "hydro-model" / "bin" / "python",
        Path("C:/ProgramData/miniforge3/envs/hydro-model/python.exe"),
        Path("C:/Users/domenicbrooks/.conda/envs/hydro-model/python.exe"),
    ]
    for c in candidates:
        if c.exists():
            return c

    # Fallback to current python
    return Path(sys.executable)


class RegionConfig:
    """Encapsulates regional settings, spatial bounds, and dataset paths."""

    def __init__(self, data: Dict[str, Any], project_root: Path = PROJECT_ROOT):
        self.raw = data
        self.project_root = project_root
        self.region_id = data.get("region_id", "hawaii")
        self.name = data.get("name", self.region_id)
        self.description = data.get("description", "")
        self.bounds = data.get("bounds", {"lat_min": 18.0, "lat_max": 23.0, "lon_min": -161.0, "lon_max": -154.0})
        self.ref_lat = float(data.get("ref_lat", 20.5))
        self.center = data.get("center", [self.ref_lat, (self.bounds["lon_min"] + self.bounds["lon_max"]) / 2])
        self.default_zoom = int(data.get("default_zoom", 8))
        self.mrms_meta = data.get("mrms", {})

        # Resolve paths
        p = data.get("paths", {})
        self.mrms_dir = self._resolve_path(p.get("mrms_dir"))
        self.default_nc = self._resolve_path(p.get("default_nc"))
        self.boundary_shp = self._resolve_path(p.get("boundary_shp"))
        self.watershed_parquet = self._resolve_path(p.get("watershed_parquet"))
        self.watershed_shp_fallback = self._resolve_path(p.get("watershed_shp_fallback"))
        self.gauge_parquet = self._resolve_path(p.get("gauge_parquet"))

        # Climatology (Atlas 14 / Atlas 15)
        env_a14 = os.environ.get("ATLAS14_NC")
        if env_a14:
            self.atlas14_nc = Path(env_a14)
        else:
            self.atlas14_nc = self._resolve_path(p.get("atlas14_nc"))

        # Hydrology model paths
        self.sac_params_dir = self._resolve_path(p.get("sac_params_dir"))
        self.uhg_params_dir = self._resolve_path(p.get("uhg_params_dir"))
        self.forcing_dir = self._resolve_path(p.get("forcing_dir"))
        self.sac_ctrl_template = self._resolve_path(p.get("sac_ctrl_template"))
        self.calibrated_basins_dir = self._resolve_path(p.get("calibrated_basins_dir"))

    def _resolve_path(self, val: Optional[str]) -> Optional[Path]:
        if not val:
            return None
        p = Path(val)
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()

    def get_available_basin_ids(self) -> list:
        """Find calibrated basin IDs from the calibration output directory."""
        if not self.calibrated_basins_dir or not self.calibrated_basins_dir.exists():
            return []
        try:
            return sorted([
                d.replace("basin_", "")
                for d in os.listdir(self.calibrated_basins_dir)
                if d.startswith("basin_") and (self.calibrated_basins_dir / d).is_dir()
            ])
        except Exception:
            return []


def load_region_config(region_name: Optional[str] = None) -> RegionConfig:
    """
    Load a region profile from regions/<region_name>/config.json.
    Defaults to ACTIVE_REGION in .env, or 'hawaii'.
    """
    if region_name is None:
        region_name = os.environ.get("ACTIVE_REGION", "hawaii")

    reg_dir = PROJECT_ROOT / "regions" / region_name
    json_path = reg_dir / "config.json"
    yaml_path = reg_dir / "config.yaml"

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif yaml_path.exists():
        try:
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except ImportError:
            raise ImportError(f"Found {yaml_path} but PyYAML is not installed. Install pyyaml or use JSON.")
    else:
        # Fallback to default hawaii if specified region file not found
        fallback_json = PROJECT_ROOT / "regions" / "hawaii" / "config.json"
        if fallback_json.exists():
            print(f"  [warn] Region '{region_name}' config not found at {json_path}. Falling back to 'hawaii'.")
            with open(fallback_json, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise FileNotFoundError(f"No region configuration found for '{region_name}' in {reg_dir}")

    return RegionConfig(data, project_root=PROJECT_ROOT)
