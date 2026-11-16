"""Build reusable Muskingum-Cunge network configs and route TCI runoff."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any
import warnings

import numpy as np
import pandas as pd


CONFIG_SCHEMA_VERSION = 3
DEFAULT_ROUTING_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configurations" / "routing"
DEFAULT_ROUTING_TIMESTEP_SECONDS = 300
DEFAULT_ROUTING_TAIL_HOURS = 36
MAX_SUBREACHES = 40
STAGE_RESOLUTION = 100
MIN_BED_SLOPE = 1e-5
MIN_BOTTOM_WIDTH_M = 1.0
DEFAULT_SIDE_SLOPE = 2.0
DEFAULT_MANNINGS_N = 0.06
DEFAULT_BANKFULL_DEPTH_M = 2.0

ATTRIBUTE_ALIASES = {
    "reach_id": ["link", "id", "ID"],
    "bankfull_depth_m": ["Y", "y", "bankfull_depth"],
    "mannings_n": ["n", "n_", "ManningN", "mannings_n"],
    "bottom_width_m": ["BtmWdth", "btmwdth", "bw", "BtmWdth_m"],
    "top_width_m": ["TopWdth", "topwdth", "tw"],
    "side_slope": ["ChSlp", "chslp", "side_slope"],
    "bed_slope": ["So", "so", "s0", "slope"],
}


@dataclass(frozen=True)
class RoutingResult:
    per_reach: pd.DataFrame
    hourly: pd.DataFrame
    outlet: pd.DataFrame
    rating_exceeded: tuple[str, ...]
    stability_adjusted_reaches: tuple[str, ...]


class _NetworkTrapezoidalReach:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from muskingumcunge.reach import TrapezoidalReach

        class IsolatedTrapezoidalReach(TrapezoidalReach):
            def __init__(self, *reach_args: Any, **reach_kwargs: Any) -> None:
                self.geometry = {}
                self.rating_curve = {}
                self.muskingum_params = {}
                super().__init__(*reach_args, **reach_kwargs)

        return IsolatedTrapezoidalReach(*args, **kwargs)


def extract_usgs_id(hydrofabric_path: str | Path) -> str:
    """Extract the USGS site number from a hydrofabric GeoPackage filename."""
    filename = Path(hydrofabric_path).stem
    match = re.search(r"usgs[-_](\d+)", filename, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(
            f"Hydrofabric filename must contain a USGS site ID, found: {filename}"
        )
    return match.group(1)


def _normalize_ids(values: pd.Series) -> pd.Series:
    if values.isna().any():
        raise ValueError("Hydrofabric identifiers cannot be null")
    return values.astype(str)


def _resolve_column(frame: pd.DataFrame, candidates: list[str], required: bool) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    if required:
        raise ValueError(
            f"Missing required field; expected one of {candidates}, "
            f"found {sorted(frame.columns)}"
        )
    return None


def _canonical_attributes(frame: pd.DataFrame, required: bool) -> pd.DataFrame:
    resolved = {
        canonical: column
        for canonical, candidates in ATTRIBUTE_ALIASES.items()
        if (column := _resolve_column(frame, candidates, required=False)) is not None
    }
    if "reach_id" not in resolved:
        if required:
            raise ValueError("Channel attributes need a link, id, or ID reach key")
        return pd.DataFrame(columns=["reach_id"])
    canonical = frame[list(resolved.values())].copy()
    canonical.columns = list(resolved)
    canonical["reach_id"] = _normalize_ids(canonical["reach_id"])
    return canonical.drop_duplicates("reach_id", keep="first")


def _positive_values(
    values: pd.Series, replacement: pd.Series | float
) -> tuple[pd.Series, int]:
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = ~np.isfinite(numeric) | (numeric <= 0)
    replacement_values = replacement if isinstance(replacement, pd.Series) else float(replacement)
    return numeric.where(~invalid, replacement_values), int(invalid.sum())


def _topological_order(network: pd.DataFrame) -> tuple[list[str], dict[str, list[str]]]:
    reach_ids = set(network["reach_id"])
    downstream_by_reach = network.set_index("reach_id")["downstream_reach_id"].to_dict()
    upstream_by_reach = {reach_id: [] for reach_id in reach_ids}
    indegree = {reach_id: 0 for reach_id in reach_ids}
    for reach_id, downstream_id in downstream_by_reach.items():
        if pd.notna(downstream_id):
            upstream_by_reach[downstream_id].append(reach_id)
            indegree[downstream_id] += 1

    ready = sorted(reach_id for reach_id, degree in indegree.items() if degree == 0)
    order = []
    while ready:
        reach_id = ready.pop(0)
        order.append(reach_id)
        downstream_id = downstream_by_reach[reach_id]
        if pd.notna(downstream_id):
            indegree[downstream_id] -= 1
            if indegree[downstream_id] == 0:
                ready.append(downstream_id)
                ready.sort()
    if len(order) != len(reach_ids):
        raise ValueError("The flowpath graph contains a cycle")
    return order, upstream_by_reach


def _build_reach(geometry: pd.Series, reach_length_m: float) -> Any:
    return _NetworkTrapezoidalReach(
        float(geometry["bottom_width_m"]),
        float(geometry["library_side_slope"]),
        float(geometry["mannings_n"]),
        float(geometry["bed_slope"]),
        float(reach_length_m),
        max_stage=float(geometry["max_stage_m"]),
        stage_resolution=STAGE_RESOLUTION,
    )


def build_routing_config(
    hydrofabric_path: str | Path,
    output_root: str | Path,
    region: str,
    routing_timestep_seconds: int = DEFAULT_ROUTING_TIMESTEP_SECONDS,
    overwrite: bool = False,
) -> Path:
    """Build a portable routing config once from a regional hydrofabric."""
    import fiona
    import geopandas as gpd

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", region):
        raise ValueError("region must contain only letters, digits, '.', '_' or '-'")
    if routing_timestep_seconds <= 0:
        raise ValueError("routing_timestep_seconds must be positive")

    source_path = Path(hydrofabric_path).expanduser().resolve()
    usgs_id = extract_usgs_id(source_path)
    destination = Path(output_root).expanduser().resolve() / usgs_id
    manifest_path = destination / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Routing config already exists: {destination}; pass overwrite=True")

    layers = set(fiona.listlayers(source_path))
    required_layers = {"flowpaths", "nexus", "divides", "flowpath-attributes"}
    missing_layers = sorted(required_layers - layers)
    if missing_layers:
        raise ValueError(f"Hydrofabric is missing required layers: {missing_layers}")

    flowpaths = gpd.read_file(source_path, layer="flowpaths")
    nexuses = gpd.read_file(source_path, layer="nexus")
    divides = gpd.read_file(source_path, layer="divides")
    attributes = _canonical_attributes(
        gpd.read_file(source_path, layer="flowpath-attributes"), required=True
    )
    attributes["attribute_source"] = "flowpath-attributes"
    if "flowpath-attributes-ml" in layers:
        fallback = _canonical_attributes(
            gpd.read_file(source_path, layer="flowpath-attributes-ml"), required=False
        )
        if not fallback.empty:
            primary_values = attributes.set_index("reach_id").drop(columns="attribute_source")
            fallback_values = fallback.set_index("reach_id")
            primary_ids = set(primary_values.index)
            value_columns = sorted(set(primary_values.columns) | set(fallback_values.columns))
            all_reach_ids = primary_values.index.union(fallback_values.index)
            primary_values = primary_values.reindex(index=all_reach_ids, columns=value_columns)
            fallback_values = fallback_values.reindex(index=all_reach_ids, columns=value_columns)
            fallback_contributed = (primary_values.isna() & fallback_values.notna()).any(axis=1)
            attributes = primary_values.combine_first(fallback_values)
            attributes["attribute_source"] = "flowpath-attributes"
            attributes.loc[~attributes.index.isin(primary_ids), "attribute_source"] = (
                "flowpath-attributes-ml"
            )
            attributes.loc[
                attributes.index.isin(primary_ids) & fallback_contributed,
                "attribute_source",
            ] = "flowpath-attributes+flowpath-attributes-ml"
            attributes = attributes.reset_index()

    required_flowpath_fields = ["id", "toid", "divide_id", "lengthkm"]
    if missing := sorted(set(required_flowpath_fields) - set(flowpaths.columns)):
        raise ValueError(f"flowpaths is missing columns: {missing}")
    if missing := sorted({"id", "toid"} - set(nexuses.columns)):
        raise ValueError(f"nexus is missing columns: {missing}")
    if missing := sorted({"divide_id", "areasqkm"} - set(divides.columns)):
        raise ValueError(f"divides is missing columns: {missing}")
    if flowpaths["id"].duplicated().any() or nexuses["id"].duplicated().any():
        raise ValueError("Flowpath and nexus ids must be unique")
    if divides["divide_id"].duplicated().any():
        raise ValueError("Divide ids must be unique")

    network = flowpaths[required_flowpath_fields].copy()
    network["reach_id"] = _normalize_ids(network.pop("id"))
    network["downstream_nexus_id"] = _normalize_ids(network.pop("toid"))
    network["divide_id"] = _normalize_ids(network["divide_id"])
    if network["divide_id"].duplicated().any():
        raise ValueError("Each routed reach must map to a unique divide_id")
    nexus_ids = _normalize_ids(nexuses["id"])
    nexus_downstream = pd.Series(
        nexuses["toid"].astype("string").to_numpy(), index=nexus_ids
    )
    missing_nexuses = sorted(set(network["downstream_nexus_id"]) - set(nexus_ids))
    if missing_nexuses:
        raise ValueError(f"Flowpaths refer to unknown downstream nexus ids: {missing_nexuses[:10]}")
    network["downstream_reach_id"] = network["downstream_nexus_id"].map(nexus_downstream)
    network.loc[~network["downstream_reach_id"].isin(network["reach_id"]), "downstream_reach_id"] = pd.NA
    network["downstream_reach_id"] = network["downstream_reach_id"].astype("string")

    areas = divides[["divide_id", "areasqkm"]].copy()
    areas["divide_id"] = _normalize_ids(areas["divide_id"])
    areas["area_km2"], area_repairs = _positive_values(areas["areasqkm"], np.nan)
    areas = areas.drop(columns="areasqkm")
    network = network.merge(areas, on="divide_id", how="left", validate="one_to_one")
    if network["area_km2"].isna().any():
        missing_divides = network.loc[network["area_km2"].isna(), "divide_id"].tolist()
        raise ValueError(f"Missing positive divide areas for ids: {missing_divides[:10]}")

    network = network.merge(attributes, on="reach_id", how="left", validate="one_to_one")
    network["attribute_source"] = network["attribute_source"].fillna("missing")
    repair_counts = {"area_missing_or_nonpositive": area_repairs}
    network["reach_length_m"], repair_counts["length_from_lengthkm"] = _positive_values(
        network["lengthkm"] * 1000.0, np.nan
    )
    network["bankfull_depth_m"], repair_counts["bankfull_depth_defaulted"] = _positive_values(
        network.get("bankfull_depth_m", pd.Series(np.nan, index=network.index)),
        DEFAULT_BANKFULL_DEPTH_M,
    )
    network["mannings_n"], repair_counts["mannings_n_defaulted"] = _positive_values(
        network.get("mannings_n", pd.Series(np.nan, index=network.index)),
        DEFAULT_MANNINGS_N,
    )
    network["bottom_width_m"], repair_counts["bottom_width_floored"] = _positive_values(
        network.get("bottom_width_m", pd.Series(np.nan, index=network.index)),
        MIN_BOTTOM_WIDTH_M,
    )
    top_width = pd.to_numeric(
        network.get("top_width_m", pd.Series(np.nan, index=network.index)), errors="coerce"
    )
    top_width = top_width.where(top_width >= network["bottom_width_m"], np.nan)
    network["top_width_m"], repair_counts["top_width_defaulted"] = _positive_values(
        top_width, network["bottom_width_m"]
    )
    network["bed_slope"], repair_counts["bed_slope_floored"] = _positive_values(
        network.get("bed_slope", pd.Series(np.nan, index=network.index)), MIN_BED_SLOPE
    )
    network["bed_slope"] = network["bed_slope"].clip(lower=MIN_BED_SLOPE)
    side_slope = pd.to_numeric(
        network.get("side_slope", pd.Series(np.nan, index=network.index)), errors="coerce"
    )
    derived_side_slope = (network["top_width_m"] - network["bottom_width_m"]) / (
        2.0 * network["bankfull_depth_m"]
    )
    side_slope = side_slope.where(np.isfinite(side_slope) & (side_slope > 0), derived_side_slope)
    network["side_slope"], repair_counts["side_slope_defaulted"] = _positive_values(
        side_slope, DEFAULT_SIDE_SLOPE
    )
    network["library_side_slope"] = 2.0 * network["side_slope"]
    network["max_stage_m"] = np.maximum(5.0 * network["bankfull_depth_m"], 5.0)
    geometry_columns = [
        "reach_length_m", "area_km2", "bankfull_depth_m", "mannings_n",
        "bottom_width_m", "top_width_m", "side_slope", "bed_slope",
    ]
    if not np.isfinite(network[geometry_columns].to_numpy(dtype=float)).all():
        raise ValueError("Reach geometry contains non-finite values after applying defaults")

    order, _ = _topological_order(network)
    order_by_reach = {reach_id: index for index, reach_id in enumerate(order)}
    network["topological_order"] = network["reach_id"].map(order_by_reach)
    terminal_reaches = network.loc[network["downstream_reach_id"].isna()]
    if terminal_reaches.empty:
        raise ValueError("The flowpath network has no terminal reaches")
    if terminal_reaches["downstream_nexus_id"].isna().any():
        missing_ids = terminal_reaches.loc[
            terminal_reaches["downstream_nexus_id"].isna(), "reach_id"
        ].tolist()
        raise ValueError(f"Terminal reaches are missing downstream nexus IDs: {missing_ids}")
    outlet_ids = terminal_reaches["reach_id"].tolist()
    outlet_reaches_by_nexus = {
        str(nexus_id): group["reach_id"].tolist()
        for nexus_id, group in terminal_reaches.groupby("downstream_nexus_id", sort=False)
    }

    network["subreach_count"] = 0
    network["bankfull_discharge_m3s"] = np.nan
    network["max_rating_discharge_m3s"] = np.nan
    network["full_reach_k_median_hours"] = np.nan
    curve_rows = []
    routing_timestep_hours = routing_timestep_seconds / 3600.0
    for row in network.itertuples(index=False):
        geometry = pd.Series(row._asdict())
        full_reach = _build_reach(geometry, row.reach_length_m)
        full_k_median = float(np.nanmedian(full_reach.muskingum_params["k"]))
        subreach_count = int(np.clip(round(full_k_median / routing_timestep_hours), 1, MAX_SUBREACHES))
        reach = _build_reach(geometry, row.reach_length_m / subreach_count)
        stages = np.asarray(reach.rating_curve["stage"], dtype=float)
        discharges = np.asarray(reach.rating_curve["discharge"], dtype=float)
        k_values = np.asarray(reach.muskingum_params["k"], dtype=float)
        x_values = np.asarray(reach.muskingum_params["x"], dtype=float)
        if not all(np.isfinite(values).all() for values in (stages, discharges, k_values, x_values)):
            raise ValueError(f"Non-finite rating or Muskingum parameters for reach {row.reach_id}")
        if np.any(k_values <= 0) or np.any(np.diff(discharges) < 0):
            raise ValueError(f"Invalid rating or Muskingum parameters for reach {row.reach_id}")
        curve_rows.extend(
            {
                "reach_id": row.reach_id,
                "stage_m": stage,
                "discharge_m3s": discharge,
                "k_hours": k_value,
                "x": x_value,
            }
            for stage, discharge, k_value, x_value in zip(
                stages, discharges, k_values, x_values, strict=True
            )
        )
        network.loc[network["reach_id"].eq(row.reach_id), "subreach_count"] = subreach_count
        network.loc[network["reach_id"].eq(row.reach_id), "bankfull_discharge_m3s"] = np.interp(
            row.bankfull_depth_m, stages, discharges
        )
        network.loc[network["reach_id"].eq(row.reach_id), "max_rating_discharge_m3s"] = discharges[-1]
        network.loc[network["reach_id"].eq(row.reach_id), "full_reach_k_median_hours"] = full_k_median

    destination.mkdir(parents=True, exist_ok=True)
    network.drop(columns=["downstream_nexus_id", "lengthkm"], errors="ignore").to_parquet(
        destination / "network.parquet", index=False
    )
    pd.DataFrame(curve_rows).to_parquet(destination / "rating_curves.parquet", index=False)
    manifest = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "usgs_id": usgs_id,
        "region": region,
        "source_hydrofabric": str(source_path),
        "source_size_bytes": source_path.stat().st_size,
        "source_modified_ns": source_path.stat().st_mtime_ns,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "routing_timestep_seconds": routing_timestep_seconds,
        "layers": sorted(required_layers | ({"flowpath-attributes-ml"} if "flowpath-attributes-ml" in layers else set())),
        "reach_count": len(network),
        "nexus_count": len(nexuses),
        "outlet_reach_ids": outlet_ids,
        "outlet_nexus_ids": list(outlet_reaches_by_nexus),
        "outlet_reaches_by_nexus": outlet_reaches_by_nexus,
        "units": {
            "area": "km2",
            "length": "m",
            "stage": "m",
            "discharge": "m3/s",
            "muskingum_k": "hours",
            "tci": "mm/model_timestep",
        },
        "network_file": "network.parquet",
        "rating_curves_file": "rating_curves.parquet",
        "geometry_repairs": repair_counts,
        "attribute_source_counts": {
            source: int(count)
            for source, count in network["attribute_source"].value_counts().items()
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination


def _load_routing_config(config_dir: str | Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    directory = Path(config_dir).expanduser().resolve()
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No routing manifest found in {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported routing config schema version: {manifest.get('schema_version')}"
        )
    network_path = directory / manifest["network_file"]
    curves_path = directory / manifest["rating_curves_file"]
    network = pd.read_parquet(network_path)
    curves = pd.read_parquet(curves_path)
    required_network_columns = {
        "reach_id", "divide_id", "downstream_reach_id", "area_km2",
        "topological_order", "subreach_count", "bankfull_discharge_m3s",
        "max_rating_discharge_m3s",
    }
    if missing := sorted(required_network_columns - set(network.columns)):
        raise ValueError(f"Routing network config is missing columns: {missing}")
    if missing := sorted({"reach_id", "stage_m", "discharge_m3s", "k_hours", "x"} - set(curves.columns)):
        raise ValueError(f"Routing curve config is missing columns: {missing}")
    network["reach_id"] = _normalize_ids(network["reach_id"])
    network["divide_id"] = _normalize_ids(network["divide_id"])
    network["downstream_reach_id"] = network["downstream_reach_id"].astype("string")
    curves["reach_id"] = _normalize_ids(curves["reach_id"])
    return manifest, network.sort_values("topological_order"), curves


def _muskingum_coefficients(
    curves: pd.DataFrame,
    reference_discharge: float,
    dt_hours: float,
    stability_adjusted_reaches: set[str] | None = None,
) -> tuple[float, float, float]:
    stage = np.interp(
        reference_discharge,
        curves["discharge_m3s"],
        curves["stage_m"],
    )
    k_value = np.interp(stage, curves["stage_m"], curves["k_hours"])
    x_value = np.interp(stage, curves["stage_m"], curves["x"])
    timestep_ratio = dt_hours / k_value
    maximum_stable_x = min(timestep_ratio / 2.0, 1.0 - timestep_ratio / 2.0)
    if maximum_stable_x < 0:
        reach_id = curves["reach_id"].iloc[0]
        raise ValueError(
            f"Routing timestep is too large for reach {reach_id}; rebuild the config "
            "with a shorter timestep"
        )
    bounded_x = min(max(x_value, 0.0), maximum_stable_x)
    if bounded_x != x_value and stability_adjusted_reaches is not None:
        stability_adjusted_reaches.add(str(curves["reach_id"].iloc[0]))
    denominator = (2.0 * (1.0 - bounded_x)) + timestep_ratio
    return (
        (timestep_ratio - (2.0 * bounded_x)) / denominator,
        (timestep_ratio + (2.0 * bounded_x)) / denominator,
        ((2.0 * (1.0 - bounded_x)) - timestep_ratio) / denominator,
    )


def _route_subreach(
    curves: pd.DataFrame,
    inflow: np.ndarray,
    dt_hours: float,
    parameter_mode: str,
    reference_discharge: float,
    stability_adjusted_reaches: set[str],
) -> np.ndarray:
    outflow = np.empty(len(inflow), dtype=float)
    if parameter_mode == "constant":
        coefficient_0, coefficient_1, coefficient_2 = _muskingum_coefficients(
            curves, reference_discharge, dt_hours, stability_adjusted_reaches
        )
    else:
        initial_reference = max(inflow[0], inflow[1] if len(inflow) > 1 else inflow[0])
        coefficient_0, coefficient_1, coefficient_2 = _muskingum_coefficients(
            curves, initial_reference, dt_hours, stability_adjusted_reaches
        )
    outflow[0] = max(0.0, coefficient_0 * inflow[0])
    for index in range(len(inflow) - 1):
        if parameter_mode == "variable":
            coefficient_0, coefficient_1, coefficient_2 = _muskingum_coefficients(
                curves,
                max(inflow[index], inflow[index + 1], outflow[index]),
                dt_hours,
                stability_adjusted_reaches,
            )
        outflow[index + 1] = max(
            0.0,
            coefficient_0 * inflow[index + 1]
            + coefficient_1 * inflow[index]
            + coefficient_2 * outflow[index],
        )
    return outflow


def route_muskingum_cunge(
    tci: pd.DataFrame,
    usgs_id: str,
    model_timestep_seconds: int,
    routing_config_root: str | Path | None = None,
    routing_timestep_seconds: int | None = None,
    routing_tail_hours: float = DEFAULT_ROUTING_TAIL_HOURS,
    parameter_mode: str = "constant",
    lateral_inflow_mode: str = "head",
    location_id_map: dict[str, str] | None = None,
    output_dir: str | Path | None = None,
) -> RoutingResult:
    """Route TCI by USGS site ID using its previously built network config."""
    usgs_id = str(usgs_id)
    if not usgs_id.isdigit():
        raise ValueError(f"usgs_id must contain only digits, found: {usgs_id}")
    config_root = (
        DEFAULT_ROUTING_CONFIG_ROOT
        if routing_config_root is None
        else Path(routing_config_root).expanduser().resolve()
    )
    routing_config_dir = config_root / usgs_id
    manifest, network, curves = _load_routing_config(routing_config_dir)
    if str(manifest.get("usgs_id")) != usgs_id:
        raise ValueError(
            f"Routing config at {routing_config_dir} is for USGS {manifest.get('usgs_id')}, "
            f"not {usgs_id}"
        )
    configured_timestep = int(manifest["routing_timestep_seconds"])
    if routing_timestep_seconds is not None and routing_timestep_seconds != configured_timestep:
        raise ValueError(
            "routing_timestep_seconds must match the precomputed routing config "
            f"({configured_timestep})"
        )
    if model_timestep_seconds <= 0 or model_timestep_seconds % configured_timestep:
        raise ValueError("model_timestep_seconds must be a positive multiple of the configured routing timestep")
    if routing_tail_hours < 0:
        raise ValueError("routing_tail_hours must be nonnegative")
    if parameter_mode not in {"constant", "variable"}:
        raise ValueError(f"Unknown parameter mode: {parameter_mode}")
    if lateral_inflow_mode not in {"head", "outlet"}:
        raise ValueError(f"Unknown lateral inflow mode: {lateral_inflow_mode}")
    forcing = tci.copy()
    if "value_time" not in forcing.columns and forcing.index.name == "value_time":
        forcing = forcing.reset_index()
    if not {"location_id", "value_time", "value"}.issubset(forcing.columns):
        raise ValueError(
            "TCI must contain location_id and value columns, with value_time "
            "as a column or named index"
        )
    if "variable_name" in forcing.columns:
        forcing = forcing.loc[forcing["variable_name"].eq("tci")].copy()
    if forcing.empty:
        raise ValueError("TCI input contains no tci records")
    if "measurement_unit" in forcing.columns:
        units = set(forcing["measurement_unit"].dropna().astype(str).str.lower())
        if units - {"mm", "mm/timestep"}:
            raise ValueError(f"TCI values must be in mm per timestep, found units: {sorted(units)}")
    forcing["location_id"] = forcing["location_id"].astype(str)
    divide_ids = set(network["divide_id"].astype(str))
    if location_id_map:
        normalized_id_map = {str(source): str(target) for source, target in location_id_map.items()}
    else:
        normalized_id_map = {}

    def normalize_location_id(location_id: str) -> str:
        if location_id in normalized_id_map:
            return normalized_id_map[location_id]
        if location_id in divide_ids:
            return location_id
        prefix = f"{usgs_id}-"
        if location_id.startswith(prefix) and location_id[len(prefix):] in divide_ids:
            return location_id[len(prefix):]
        return location_id

    forcing["location_id"] = forcing["location_id"].map(normalize_location_id)
    forcing["value_time"] = pd.to_datetime(forcing["value_time"], errors="raise")
    forcing["value"] = pd.to_numeric(forcing["value"], errors="raise")
    if not np.isfinite(forcing["value"]).all() or (forcing["value"] < 0).any():
        raise ValueError("TCI values must be finite and nonnegative")
    if forcing.duplicated(["value_time", "location_id"]).any():
        raise ValueError("TCI has duplicate location_id/value_time records")

    forcing_grid = forcing.pivot(index="value_time", columns="location_id", values="value").sort_index()
    expected_divides = set(network["divide_id"])
    supplied_divides = set(forcing_grid.columns)
    missing_divides = sorted(expected_divides - supplied_divides)
    extra_divides = sorted(supplied_divides - expected_divides)
    if missing_divides or extra_divides:
        raise ValueError(
            f"TCI divide IDs do not match routing config; missing={missing_divides[:10]}, "
            f"extra={extra_divides[:10]}"
        )
    time_deltas = forcing_grid.index.to_series().diff().dropna().dt.total_seconds()
    if not time_deltas.eq(model_timestep_seconds).all():
        raise ValueError("TCI timestamps must be regular and match model_timestep_seconds")

    area_by_divide = network.set_index("divide_id")["area_km2"]
    local_flow_by_divide = forcing_grid.mul(area_by_divide, axis="columns") * 1000.0
    local_flow_by_divide /= model_timestep_seconds
    first_zero_time = forcing_grid.index[-1] + pd.to_timedelta(model_timestep_seconds, unit="s")
    routing_step = pd.to_timedelta(configured_timestep, unit="s")
    routing_times = pd.date_range(
        forcing_grid.index[0],
        first_zero_time
        + pd.to_timedelta(routing_tail_hours, unit="h")
        - routing_step,
        freq=routing_step,
    )
    forcing_with_zero = pd.concat(
        [local_flow_by_divide, pd.DataFrame(0.0, index=[first_zero_time], columns=local_flow_by_divide.columns)]
    )
    local_substep_flow = forcing_with_zero.reindex(routing_times, method="ffill")
    if local_substep_flow.isna().any().any():
        raise ValueError("Could not align TCI forcing to the configured routing timestep")

    divide_by_reach = network.set_index("reach_id")["divide_id"]
    local_flow_by_reach = pd.DataFrame(
        {
            reach_id: local_substep_flow[divide_by_reach[reach_id]].to_numpy()
            for reach_id in network["reach_id"]
        },
        index=routing_times,
    )
    ordered_reaches = network.sort_values("topological_order")["reach_id"].tolist()
    upstream_by_reach = {reach_id: [] for reach_id in ordered_reaches}
    for reach in network.itertuples(index=False):
        if pd.notna(reach.downstream_reach_id):
            upstream_by_reach[reach.downstream_reach_id].append(reach.reach_id)
    network_by_id = network.set_index("reach_id")
    curves_by_reach = {
        reach_id: rows.sort_values("stage_m")
        for reach_id, rows in curves.groupby("reach_id", sort=False)
    }
    zero_flow = np.zeros(len(routing_times), dtype=float)
    inflow_by_reach: dict[str, np.ndarray] = {}
    outflow_by_reach: dict[str, np.ndarray] = {}
    rating_exceeded = []
    stability_adjusted_reaches: set[str] = set()

    for reach_id in ordered_reaches:
        upstream_inflow = sum(
            (outflow_by_reach[upstream_id] for upstream_id in upstream_by_reach[reach_id]),
            start=zero_flow,
        )
        local_inflow = local_flow_by_reach[reach_id].to_numpy()
        reach_inflow = (
            upstream_inflow + local_inflow
            if lateral_inflow_mode == "head"
            else upstream_inflow
        )
        reach_parameters = network_by_id.loc[reach_id]
        reference_discharge = float(reach_inflow.max())
        if reference_discharge <= 0:
            reference_discharge = float(reach_parameters["bankfull_discharge_m3s"])
        routed = reach_inflow
        for _ in range(int(reach_parameters["subreach_count"])):
            routed = _route_subreach(
                curves_by_reach[reach_id],
                routed,
                configured_timestep / 3600.0,
                parameter_mode,
                reference_discharge,
                stability_adjusted_reaches,
            )
        inflow_by_reach[reach_id] = reach_inflow
        outflow_by_reach[reach_id] = (
            routed
            if lateral_inflow_mode == "head"
            else routed + local_inflow
        )
        if reach_inflow.max() > float(reach_parameters["max_rating_discharge_m3s"]):
            rating_exceeded.append(reach_id)

    all_outflows = np.stack([outflow_by_reach[reach_id] for reach_id in ordered_reaches])
    if not np.isfinite(all_outflows).all() or (all_outflows < 0).any():
        raise ValueError("Muskingum-Cunge routing produced invalid discharge values")
    outflow_frame = pd.DataFrame(
        {reach_id: outflow_by_reach[reach_id] for reach_id in ordered_reaches},
        index=routing_times,
    )
    outflow_frame.index.name = "value_time"
    per_reach = (
        outflow_frame.stack()
        .rename("value")
        .rename_axis(["value_time", "location_id"])
        .reset_index()
    )
    per_reach["variable_name"] = "q_outflow"
    per_reach["measurement_unit"] = "m3/s"
    per_reach["configuration"] = f"MC_trapezoid_{lateral_inflow_mode}"
    hourly_frame = outflow_frame.resample("1h").mean()
    hourly = (
        hourly_frame.stack()
        .rename("value")
        .rename_axis(["value_time", "location_id"])
        .reset_index()
    )
    hourly["variable_name"] = "q_outflow"
    hourly["measurement_unit"] = "m3/s"
    hourly["configuration"] = f"MC_trapezoid_{lateral_inflow_mode}"
    outlet_groups = manifest.get("outlet_reaches_by_nexus", {})
    if not outlet_groups:
        raise ValueError("Routing config must define terminal reaches grouped by outlet nexus")
    outlet_frames = []
    for nexus_id, reach_ids in outlet_groups.items():
        missing_outlet_ids = sorted(set(reach_ids) - set(outflow_frame.columns))
        if missing_outlet_ids:
            raise ValueError(
                f"Routing config references unknown outlet reaches: {missing_outlet_ids}"
            )
        outlet_series = (
            outflow_frame[reach_ids]
            .sum(axis=1)
            .rename("value")
            .rename_axis("value_time")
            .reset_index()
        )
        outlet_series["location_id"] = nexus_id
        outlet_frames.append(outlet_series)
    outlet = pd.concat(outlet_frames, ignore_index=True)
    outlet["variable_name"] = "q_outflow"
    outlet["measurement_unit"] = "m3/s"
    outlet["configuration"] = f"MC_trapezoid_{lateral_inflow_mode}"
    result = RoutingResult(
        per_reach,
        hourly,
        outlet,
        tuple(rating_exceeded),
        tuple(sorted(stability_adjusted_reaches)),
    )
    if result.stability_adjusted_reaches:
        warnings.warn(
            f"Muskingum x was stability-bounded for {len(result.stability_adjusted_reaches)} reaches; "
            "inspect result.stability_adjusted_reaches",
            RuntimeWarning,
            stacklevel=2,
        )
    if output_dir is not None:
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        result.per_reach.to_parquet(destination / "mc_network_output.parquet", index=False)
        result.hourly.to_parquet(destination / "mc_network_output_hourly.parquet", index=False)
        result.outlet.to_parquet(destination / "mc_outlet_output.parquet", index=False)
    return result