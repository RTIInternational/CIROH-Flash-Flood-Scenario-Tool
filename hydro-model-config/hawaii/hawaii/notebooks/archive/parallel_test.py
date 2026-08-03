from spot_setup_pss import spot_setup
from pathlib import Path
import spotpy
import numpy as np
import matplotlib.pyplot as plt
import time
from mpi4py import MPI

basin_name = "ANBC2"
start_date = np.datetime64("2000-10-01 00:00:00")
end_date   = np.datetime64("2020-09-30 18:00:00")

spy_setup = spot_setup(
                basin_name=basin_name,
                config_dir=Path("../configurations"),
                start_date=start_date,
                end_date=end_date
            )


def run_algorithm(algo_name, algo_class, n_samples, parallel='mpi'):
    """Generic function to run any SPOTPY algorithm"""
    print(f"\n{'='*50}")
    print(f"  Running: {algo_name}")
    print(f"{'='*50}")

    db_name = f'results_{algo_name.lower()}'

    sampler = algo_class(
        spy_setup,
        dbname=db_name,
        dbformat='csv',
        parallel=parallel,
        save_sim=True,
    )
    
    if algo_name == "SCE-UA":
        sampler.sample(n_samples, ngs=30, kstop=15, pcento=0.01)
    else:
        sampler.sample(n_samples)
    print(f"  {algo_name} done! Results saved to {db_name}.csv")

    
    
    return db_name


if __name__ == "__main__":
    # Define the number of samples for each algorithm
    n_samples = 100

    # List of algorithms to run
    algorithms = [
        # ("SCE-UA", spotpy.algorithms.sceua),
        ("DDS", spotpy.algorithms.dds),
        # ("DE", spotpy.algorithms.de),
        # ("MC", spotpy.algorithms.mc)
    ]

    # Run each algorithm in parallel
    for algo_name, algo_class in algorithms:
        run_algorithm(algo_name, algo_class, n_samples)
    


