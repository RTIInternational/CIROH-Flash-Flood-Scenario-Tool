"""
SAC-SMA + UHG worker script.

Runs under the 'hydro-model' conda environment (pysacsnow required).
Called as a subprocess by dashboard_mrms.py — reads JSON from stdin,
returns JSON to stdout.

Input JSON
----------
{
    "event_start": "2026-03-15T00:00:00",   # ISO-8601, naive UTC
    "times":       ["2026-03-15T00:00:00", ...],   # n hourly steps
    "scenarios": {
        "Original":   [ppt_0, ppt_1, ...],   # mm hr⁻¹
        "Transposed": [...],                  # optional
        "Modified":   [...]                   # optional
    }
}

Output JSON (success)
---------------------
{
    "Original":   [q_0, q_1, ...],   # m³ s⁻¹
    "Transposed": [...],
    "Modified":   [...]
}

Output JSON (error)
-------------------
{"error": "message"}
"""

import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pysacsnow as pss

# ── SAC-SMA / UHG paths ───────────────────────────────────────────────────────
HAWAII_RESULTS_DIR = Path(
    r"C:\Users\domenicbrooks\OneDrive - Research Triangle Institute"
    r"\CIROH Flash Flooding\Hawaii results"
)
HYDRO_MODEL_CONFIG_DIR = Path(
    r"C:\Users\domenicbrooks\OneDrive - Research Triangle Institute"
    r"\CIROH Flash Flooding\hydro-model-config\hawaii\hawaii"
)
# Template SAC control YAML — paths are overridden at runtime for each basin
_SAC_CTRL_TEMPLATE = HYDRO_MODEL_CONFIG_DIR / "configurations/sac/sac.control.16103000.yaml"
_SAC_STATE_VARS    = ("uztwc", "uzfwc", "lztwc", "lzfpc", "lzfsc", "adimc")


def _paths_for_basin(basin: str):
    """Return (sac_params, uhg_params, sac_forcing, uhg_ctrl) Paths for a basin."""
    return (
        HAWAII_RESULTS_DIR  / f"configurations/sac/sac_params_{basin}_cal.json",
        HAWAII_RESULTS_DIR     / f"configurations/unithg/uhg_params_{basin}.json",
        HYDRO_MODEL_CONFIG_DIR / f"data/forcing/forcing.sac.{basin}.csv",
        HAWAII_RESULTS_DIR     / f"configurations/unithg/uhg.control.{basin}.yaml",
    )


def _spinup(event_start_str, sac_params, sac_forcing):
    """Full batch spin-up from forcing start to just before event_start."""
    sac_ctrl = pss.Controller.load_from_yaml(_SAC_CTRL_TEMPLATE, verbose=False)
    sac_ctrl.config.model.parameters.path  = sac_params
    sac_ctrl.config.model.forcing[0].path  = sac_forcing
    sac_ctrl.config.simulation.output_path = Path(tempfile.mktemp(suffix='.csv'))

    sacsma = pss.SACSMA(control=sac_ctrl, calc_method='numpy')
    sacsma.run(finalize=False)

    event_np = np.datetime64(pd.Timestamp(event_start_str).tz_localize(None), 'ns')
    times    = sacsma.value_time_state
    idx      = max(0, int(np.searchsorted(times, event_np, side='left')) - 1)
    state_date = str(pd.Timestamp(times[idx]))
    states = {v: float(sacsma.buffer_states[v][idx, 0]) for v in _SAC_STATE_VARS}
    return states, state_date


def main():
    data = json.loads(sys.stdin.read())
    basin       = data.get('basin', '16103000')
    n_extend    = int(data.get('n_extend', 12))  # zero-precip hours after event
    event_start = data['event_start']
    times_iso   = data['times']
    scenarios   = data['scenarios']   # {label: [float, ...]}
    n           = len(times_iso)
    times_list  = [pd.Timestamp(t) for t in times_iso]

    # Extended time list: n event hours + n_extend zero-precip hours
    _dt = pd.Timedelta(hours=1)
    times_all = times_list + [times_list[-1] + _dt * (k + 1) for k in range(n_extend)]
    n_total   = n + n_extend

    sac_params, uhg_params, sac_forcing, uhg_ctrl_path = _paths_for_basin(basin)
    for p, desc in [
        (sac_params,    'SAC params'),
        (sac_forcing,   'SAC forcing'),
        (uhg_params,    'UHG params'),
        (uhg_ctrl_path, 'UHG control'),
    ]:
        if not p.exists():
            print(json.dumps({"error": f"{desc} not found: {p}"}))
            return

    # ── Spin-up ───────────────────────────────────────────────────────────
    try:
        spinup_state, spinup_date = _spinup(event_start, sac_params, sac_forcing)
    except Exception as exc:
        print(json.dumps({"error": f"spin-up failed: {exc}"}))
        return

    # ── PET from forcing CSV ──────────────────────────────────────────────
    try:
        forcing_df = pd.read_csv(sac_forcing, parse_dates=['time']).set_index('time')
        pet_arr = np.zeros(n_total, dtype=np.float64)
        for i, ts in enumerate(times_all):   # covers event + extension hours
            if ts in forcing_df.index:
                pet_arr[i] = float(forcing_df.loc[ts, 'hru_pet'])
            else:
                loc = forcing_df.index.get_indexer([ts], method='nearest')[0]
                if loc >= 0:
                    pet_arr[i] = float(forcing_df.iloc[loc]['hru_pet'])
    except Exception as exc:
        print(json.dumps({"error": f"PET load failed: {exc}"}))
        return

    # ── Build model instances ─────────────────────────────────────────────
    try:
        sac_ctrl = pss.Controller.load_from_yaml(_SAC_CTRL_TEMPLATE, verbose=False)
        sac_ctrl.config.model.parameters.path  = sac_params
        sac_ctrl.config.model.forcing[0].path  = sac_forcing
        sac_ctrl.config.simulation.output_path = Path(tempfile.mktemp(suffix='.csv'))

        uhg_ctrl = pss.Controller.load_from_yaml(uhg_ctrl_path, verbose=False)
        uhg_ctrl.config.model.parameters.path  = uhg_params
        uhg_ctrl.config.model.forcing[0].path  = sac_forcing  # placeholder
        uhg_ctrl.config.simulation.output_path = Path(tempfile.mktemp(suffix='.csv'))

        sacsma = pss.SACSMA(control=sac_ctrl, calc_method='numpy')
        uhg    = pss.UHG(control=uhg_ctrl,   calc_method='numpy')
    except Exception as exc:
        print(json.dumps({"error": f"model init failed: {exc}"}))
        return

    def _run_scenario(ppt_vals):
        sacsma.reset()
        uhg.reset()
        for var, val in spinup_state.items():
            getattr(sacsma, var)[:] = np.array([val])
        # Append n_extend zero-precip hours after the event period
        ppt = np.concatenate([
            np.array(ppt_vals, dtype=np.float64),
            np.zeros(n_extend, dtype=np.float64),
        ])
        for i in range(n_total):
            sac_ctrl.advance();  sacsma.advance()
            sacsma.hru_ppt[:]  = ppt[i]
            sacsma.hru_pet[:]  = pet_arr[i]
            sacsma.hru_aesc[:] = 0.0
            sacsma.calculate(); sacsma.output()
            uhg_ctrl.advance(); uhg.advance()
            uhg.hru_tci[:] = sacsma.tci[:]
            uhg.calculate(); uhg.output()
        return np.sum(uhg.buffer_states['q'][:n_total], axis=1).tolist()

    # ── Run scenarios ─────────────────────────────────────────────────────
    try:
        results = {
            '_spinup_date': spinup_date,
            '_times_all':   [t.isoformat() for t in times_all],
            '_n_event':     n,
        }
        for label, ppt_vals in scenarios.items():
            results[label] = _run_scenario(ppt_vals)
        print(json.dumps(results))
    except Exception as exc:
        print(json.dumps({"error": f"scenario run failed: {exc}"}))


if __name__ == '__main__':
    main()
