import os
import sys
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulate_fsrs import run_simulation
from src.simulation_config import SimulationConfig


def run_bench(
    n_days: int, new_limit: int, review_limit: int, label: str
) -> tuple[float, float]:
    config = SimulationConfig(
        n_days=n_days,
        new_limit=new_limit,
        review_limit=review_limit,
        compute_final_params=False,
        return_logs=False,
        verbose=False,
        seed=42,
    )

    print(f"Running benchmark: {label}...")
    start = time.perf_counter()
    fitted, gt, metrics = run_simulation(config)
    end = time.perf_counter()

    duration = end - start
    rev_count = metrics["review_count"]
    card_count = metrics["card_count"]
    ips = rev_count / duration if duration > 0 else 0

    print(f"  Duration: {duration:.2f}s")
    print(f"  Reviews:  {rev_count}")
    print(f"  Cards:    {card_count}")
    print(f"  Rev/sec:  {ips:.2f}")
    print("-" * 30)
    return duration, ips


if __name__ == "__main__":
    # Small benchmark
    run_bench(180, 20, 200, "Small Deck (1k cards-ish, 180 days)")

    # Large benchmark
    run_bench(365, 50, 1000, "Large Deck (10k cards-ish, 365 days)")
