"""
MPI-parallel SAC-SMA calibration

SAC-SMA parameters are calibrated as global scalars 
(one value applied uniformly to all HRUs). 
The full model chain SAC-SMA → UHG is run each evaluation, 
and the objective is defined by obj_func
and calculated on hourly observed streamflow.

Usage
-----
Single-process (testing):
    cd scripts/
    python run_sceua_sac.py --reps 100 --basin 16103000 --obj kge_q90

MPI parallel:
    cd scripts/
    poetry run mpirun -n 8 python run_sceua_sac.py --reps 10000 --basin 16103000

    # Recommended -- avoid numba cache corruption under MPI:
    NUMBA_CACHE=0 poetry run mpirun -n 8 python run_sceua_sac.py --reps 10000 --basin 16103000

Output
------
A CSV results file and a best-params summary are written to --output_dir
(default: ../data/output/sceua_{basin}/).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import spotpy

sys.path.insert(0, str(Path(__file__).parent))
from spot_setup_sac import spot_setup  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Calibrate SAC-SMA against observed daily streamflow using Spotpy SCEUA (MPI-capable)'
    )
    p.add_argument('--basin',              default='16103000',
                   help='Basin name -- must match YAML/JSON filenames (default: 16103000)')
    
    p.add_argument('--reps',               type=int,   default=10000,
                   help='Total number of model evaluations (default: 10000)')
    
    p.add_argument('--ngs',                type=int,   default=50,
                   help='Number of complexes for SCEUA (default: 50)')
    
    p.add_argument('--kstop',              type=int,   default=15,
                   help='Max shuffling loops without pcento improvement (default: 15)')
    
    p.add_argument('--pcento',             type=float, default=0.01,
                   help='Convergence threshold %% change (default: 0.01)')
    
    p.add_argument('--obj_func',           default='kge_q90',
                   help='Objective function (default: kge_q90)')
    
    p.add_argument('--start_date',         default='2021-04-20',
                   help='Model start date (default: 2021-04-20)')
    
    p.add_argument('--end_date',           default='2026-04-12',
                   help='Model end date (default: 2026-04-12)')
    
    p.add_argument('--warmup_hrs',       default=4380,
                   help='Model warm up period (default: 4380 hours [6 months])')
    
    p.add_argument('--obs_path',           default='../data/observations/QIN_16103000.csv',
                   help='Path to observed daily streamflow CSV (default: ../data/observations/QIN_16103000.csv)')
    
    p.add_argument('--sac_control_yaml',
                   default='../configurations/sac/sac.control.16103000.yaml',
                   help='SAC-SMA control YAML')
    
    p.add_argument('--uhg_control_yaml',
                   default='../configurations/unithg/uhg.control.16103000.yaml',
                   help='UHG control YAML')
    
    p.add_argument('--output_dir',         default='../data/output/',
                   help='Directory for results CSV and best-params file')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir, f"basin_{args.basin}")
    output_dir.mkdir(parents=True, exist_ok=True)

    db_name = str(output_dir / f'sceua_params_{args.basin}_{args.obj_func}')

    # ------------------------------------------------------------------
    # Build setup
    # ------------------------------------------------------------------
    setup = spot_setup(
        basin                = args.basin,
        obs_path             = args.obs_path,
        sac_control_yaml     = args.sac_control_yaml,
        uhg_control_yaml     = args.uhg_control_yaml,
        start_date           = args.start_date,
        end_date             = args.end_date,
        warmup_hrs           = args.warmup_hrs,
        obj_func             = args.obj_func
    )

    n_params = len(setup._spotpy_params)
    print(f'Basin        : {args.basin}')
    print(f'HRU IDs      : {setup._hru_ids}')
    print(f'Free params  : {n_params}  '
          f'({len(setup._CALIBRATED_PARAMS)} params x {len(setup._hru_ids)} zones)')

    # ------------------------------------------------------------------
    # Run SCEUA optimisation
    # parallel='mpi'  ->  mpirun -n N python run_optimization_sac_alec2.py
    # ------------------------------------------------------------------
    sampler = spotpy.algorithms.sceua(
        setup,
        dbname   = db_name,
        dbformat = 'csv',
        # parallel = 'mpi',
        save_sim = False,
    )

    sampler.sample(
        repetitions = args.reps,
        ngs         = args.ngs,
        kstop       = args.kstop,
        peps        = args.pcento,
    )

    # ------------------------------------------------------------------
    # Post-process: only rank-0 writes the summary
    # ------------------------------------------------------------------
    try:
        from mpi4py import MPI
        is_rank0 = MPI.COMM_WORLD.Get_rank() == 0
    except ImportError:
        is_rank0 = True   # single-process run

    if is_rank0:
        results_path = Path(f'{db_name}.csv')
        if results_path.exists():
            print(f'SCEUA calibration complete  ({args.basin})')

if __name__ == '__main__':
    main()
