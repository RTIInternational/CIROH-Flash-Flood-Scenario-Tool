"""
Spotpy setup class for calibrating SAC-SMA parameters against observed daily
streamflow at ALEC2, with SNOW-17 held fixed at the per-zone calibrated values.

Parameter strategy
------------------
SAC-SMA parameters are calibrated as single global scalars applied uniformly
to all HRUs.

Model chain
-----------
  SAC-SMA (calibrated) → UHG (fixed)

Objective function options (so far)
------------------

obj_func = 'nkge1_qall'  --> NKGE' --> normalized, modification 1 of KGE (single prime) 
    Minimises 1 − NKGE', where NKGE' = KGE' / (2 − KGE') is the normalised
    modified Kling-Gupta Efficiency (Farahani et al., 2025).

obj_func = 'kge_q90'  --> KGE (original) of the 90th+ percentile flows
    Minimises -1 * KGE_q90

obj_func = 'kge_q99'  --> KGE (original) of the 99th+ percentile flows
    Minimises -1 * KGE_q99

obj_func = xxxxx --> where 'xxxxx' must be an available spotpy objective function
    Minimises -1 * score

Run from the scripts/ directory or pass absolute paths to __init__.
"""
from __future__ import annotations

import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import spotpy
import pysacsnow as pss
from pysacsnow.base.data_models import ConfigFile
from spotpy.parameter import Uniform

from spotpy.objectivefunctions import kge
from spotpy.objectivefunctions import rmse

#from spot_setup_snow17_alec2 import get_modified_KGE


# SAC-SMA parameter bounds
# less sensitive parameters (for humid basins) commented out
_SAC_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    #'uztwm': (25.0,   125.0),
    'uzfwm': (10.0,    75.0),
    #'lztwm': (75.0,   300.0),
    'lzfpm': (20.0,   100.0),
    'lzfsm': (5.0,     80.0),
    #'uzk':   ( 0.2,     0.5),
    'lzpk':  ( 0.001,  0.015),
    'lzsk':  ( 0.03,   0.2),
    #'pfree': ( 0.05,   0.5),
    'adimp': ( 0.0,    0.2),
    'zperc': (20.0,   300.0),
    'rexp':  ( 1.4,    3.5),
    #'efc':   ( 0.1,    0.3),
    #'rserv': ( 0.2,    0.4),
}

def get_modified_KGE(obs: np.ndarray, sim: np.ndarray) -> float:
    """Calculate modified Kling-Gupta Efficiency (KGE'; Kling et al., 2012)."""
    ind = (~np.isnan(obs)) & (~np.isnan(sim))
    obs = obs[ind]
    sim = sim[ind]
    try:
        sd_sim = np.std(sim, ddof=1)
        sd_obs = np.std(obs, ddof=1)
        m_sim  = np.mean(sim)
        m_obs  = np.mean(obs)
        r      = np.corrcoef(sim, obs)[0, 1]
        relvar = (float(sd_sim) / float(m_sim)) / (float(sd_obs) / float(m_obs))
        bias   = float(m_sim) / float(m_obs)
        kge    = 1.0 - np.sqrt((r - 1)**2 + (relvar - 1)**2 + (bias - 1)**2)
    except Exception:
        kge = np.nan
    return kge

class spot_setup:
    """Spotpy setup for SAC-SMA calibration against hourly streamflow .

    SAC-SMA parameters are calibrated as single global values applied uniformly to all
    HRUs. The full model chain SAC-SMA → UHG is executed
    each evaluation; the objective is defined by 'obj_func' and 
    calculated on hourly streamflow.
    """

    _CALIBRATED_PARAMS = list(_SAC_PARAM_BOUNDS.keys())

    # ------------------------------------------------------------------
    def __init__(
        self,
        basin: str = '16103000',
        obs_path: str = '../data/observations/QIN_16103000.csv',
        sac_control_yaml: str = '../configurations/sac/sac.control.16103000.yaml',
        uhg_control_yaml: str = '../configurations/unithg/uhg.control.16103000.yaml',
        start_date: str = '2021-04-20',
        end_date: str = '2026-04-12',
        warmup_hrs: int = 4380,  # default 6 months
        obj_func=None
    ):
        self.basin = basin
        self.obj_func = obj_func
        self.warmup_hrs = warmup_hrs

        # --- Build model controllers ------------------------------------
        def _make_controller(yaml_path: str) -> pss.Controller:
            with Path(yaml_path).open('r') as fh:
                cfg = yaml.safe_load(fh)
            config = ConfigFile(**cfg)
            config.simulation.start_date = np.datetime64(start_date)
            config.simulation.end_date = np.datetime64(end_date)
            return pss.Controller(config=config, verbose=False)

        sac_ctrl     = _make_controller(sac_control_yaml)
        uhg_ctrl     = _make_controller(uhg_control_yaml)

        self._sacsma  = pss.SACSMA(control=sac_ctrl,      calc_method='numba')
        self._uhg     = pss.UHG(control=uhg_ctrl,         calc_method='numba')

        self._hru_ids = self._sacsma.params.spatial_ids
        self._n_hru   = len(self._hru_ids)

        # --- Build spotpy parameter list (one per param × zone) ---------
        # Names: {param}_{zone_id}, e.g. 'uztwm_ALEC2HLF'
        self._spotpy_params: list[Uniform] = [
            Uniform(f'{param}_{zone}', low=lo, high=hi)
            for param, (lo, hi) in _SAC_PARAM_BOUNDS.items()
            for zone in self._hru_ids
        ]

        # --- Load & align observed hourly streamflow ---------------------
        obs_df = pd.read_csv(obs_path, parse_dates=['time'], index_col='time')
        obs_df = obs_df.loc[start_date:end_date]
        obs_df = obs_df.iloc[warmup_hrs:]  # skip first 6 months (6*30*24 hours) as model warm-up
        
        # Precompute the intersection of obs dates with the expected simulation
        # daily output range so that simulation() and evaluation() always return
        # arrays of the same fixed length without NaN padding.
        sim_hourly_dates = pd.date_range(start=obs_df.index[0], end=end_date, freq='h')
        common_dates = obs_df.index.intersection(sim_hourly_dates)
        
        self._obs_dates = common_dates
        self.obs = obs_df.loc[common_dates, 'QIN_cms'].to_numpy()
        
        print(f"Basin              : {self.basin}")
        print(f"HRU IDs            : {self._hru_ids}")
        print(f"Calb period        : {self._obs_dates[0].date()} → {self._obs_dates[-1].date()}")
        print(f"Calb hours         : {len(self._obs_dates)}")
        print(f"Warm up period     : {warmup_hrs}")
        print(f"Objective function : {self.obj_func}")
        print(f"SAC params  : {len(self._spotpy_params)}  "
              f"({len(self._CALIBRATED_PARAMS)} params x {len(self._hru_ids)} zones)")

    # ------------------------------------------------------------------
    def parameters(self):
        """Expose SAC parameter list to Spotpy."""
        return spotpy.parameter.generate(self._spotpy_params)

    # ------------------------------------------------------------------
    def _apply_sac_params(self, vector) -> None:
        """Write each zone's calibrated value directly to the SAC model.

        Parameters are named '{param}_{zone}' matching the spotpy param list.
        If vector is None the model retains its current (default) values.
        """
        if vector is None:
            return
        for param in self._CALIBRATED_PARAMS:
            per_hru = np.array(
                [float(getattr(vector, f'{param}_{zone}')) for zone in self._hru_ids],
                dtype=float,
            )
            setattr(self._sacsma, param, per_hru)

    # ------------------------------------------------------------------
    def simulation(self, vector) -> np.ndarray:
        """Apply SAC params, run full model chain, return hourly simulated flow
        aligned to the observed date index.
        """
        self._apply_sac_params(vector)

        # Reset all models before each run
        self._sacsma.reset()
        self._uhg.reset()

        # --- SAC-SMA ---------------------------------------------------
        self._sacsma.run(
            finalize=False,
        )
        keys = ['tci']
        uhg_input = {k: self._sacsma.buffer_states[k] for k in keys}

        # --- UHG -------------------------------------------------------
        self._uhg.run(
            finalize=False,
            input_overrides={'hru_tci': uhg_input['tci']},
        )
        total_flow = np.sum(self._uhg.buffer_states['q'], axis=1)
        flow_df = pd.DataFrame(
            {'flow_sim': total_flow},
            index=self._uhg.value_time_state,
        )

        # --- Align to observed dates by matching dates -------------------

        return flow_df.loc[self._obs_dates, 'flow_sim'].to_numpy()

    # ------------------------------------------------------------------
    def evaluation(self) -> np.ndarray:
        return self.obs

    # ------------------------------------------------------------------
    def objectivefunction(self, simulation, evaluation) -> float:
        """Return 1 − NKGE' (Spotpy minimises this → maximises NKGE').

        Only paired finite values are used so NaN-padded edges after reindex
        do not contaminate the score.
        """
        sim = np.asarray(simulation, dtype=float)
        obs = np.asarray(evaluation, dtype=float)
        mask = np.isfinite(sim) & np.isfinite(obs)
        if mask.sum() < 2:
            return np.nan

        simulation_filtered = sim[mask]
        evaluation_filtered = obs[mask]
            
        if not self.obj_func or self.obj_func == 'nkge1_qall':
            score = get_modified_KGE(evaluation_filtered, simulation_filtered)
            if np.isnan(score):
                return np.nan
            nkge = score / (2.0 - score)
            obj = 1.0 - nkge
            
        elif self.obj_func == 'kge_q90':

            q90 = np.nanpercentile(evaluation_filtered, 90)
            q90_mask = evaluation_filtered > q90

            evaluation_q90 = evaluation_filtered[q90_mask]
            simulation_q90 = simulation_filtered[q90_mask]

            score = kge(evaluation_q90, simulation_q90)
            obj = -1*score            

        elif self.obj_func == 'kge_q99':

            q99 = np.nanpercentile(evaluation_filtered, 99)
            q99_mask = evaluation_filtered > q99

            evaluation_q99 = evaluation_filtered[q99_mask]
            simulation_q99 = simulation_filtered[q99_mask]

            score = kge(evaluation_q99, simulation_q99)
            obj = -1*score  
            
        else:
            # Spotpy minimizes the objective function, so for objective functions where fitness improves with increasing result, we need to multiply by -1
            score = self.obj_func(evaluation_filtered, simulation_filtered)
            obj = -1*score
        
        return obj
