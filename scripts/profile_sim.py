import cProfile
import io
import os
import pstats
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulate_fsrs import run_simulation
from src.simulation_config import SimulationConfig


def profile_sim() -> None:
    config = SimulationConfig(
        n_days=365, new_limit=50, review_limit=1000, verbose=False, seed=42
    )

    pr = cProfile.Profile()
    pr.enable()
    run_simulation(config)
    pr.disable()

    s = io.StringIO()
    sortby = "cumulative"
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(30)
    print(s.getvalue())


if __name__ == "__main__":
    profile_sim()
