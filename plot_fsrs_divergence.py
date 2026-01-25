import argparse
import traceback
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from simulate_fsrs import run_simulation


def calculate_retrievability(
    t: np.ndarray[Any, Any],
    stability: float,
    parameters: list[float] | tuple[float, ...],
) -> np.ndarray[Any, Any]:
    # Ensure parameters is a list/tuple of floats, not tensors
    # run_simulation returns list of floats

    decay = -parameters[20]
    # Bounds for decay param (index 20) are [0.1, 0.8] in FSRS, so non-zero.
    factor = 0.9 ** (1 / decay) - 1

    # Formula: R = (1 + factor * t / S) ^ decay
    res: np.ndarray[Any, Any] = (1 + factor * t / stability) ** decay
    return res


def plot_forgetting_curves(results: list[dict[str, Any]]) -> None:
    """
    results: list of dicts with keys 'label', 'params'
    """
    plt.figure(figsize=(12, 8))

    # Time range (days)
    t = np.linspace(0, 100, 200)

    for res in results:
        label = res["label"]
        params = res["params"]

        # Scenario: First review was 'Good'
        # Stability = parameters[2] (for Good=3)
        stability = params[2]

        r_values = calculate_retrievability(t, stability, params)
        plt.plot(t, r_values, label=label)

    plt.xlabel("Days since review")
    plt.ylabel("Probability of Recall (Retrievability)")
    plt.title("Forgetting Curve Divergence (Scenario: First Review = Good)")
    plt.legend()
    plt.grid(True)
    plt.savefig("forgetting_curve_divergence.png")
    print("Plot saved to forgetting_curve_divergence.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simulations and plot divergence")
    parser.add_argument(
        "--days", type=int, nargs="+", default=[365], help="List of day limits"
    )
    parser.add_argument(
        "--reviews", type=int, nargs="+", default=[200], help="List of review limits"
    )
    parser.add_argument(
        "--retentions",
        type=str,
        nargs="+",
        default=["0.9"],
        help="List of target retentions (floats or schedules)",
    )
    parser.add_argument(
        "--burn-ins", type=int, nargs="+", default=[0], help="List of burn-in periods"
    )

    args = parser.parse_args()

    results = []
    ground_truth_captured = False

    # Iterate over all combinations
    for burn_in in args.burn_ins:
        for days in args.days:
            for reviews in args.reviews:
                for retention in args.retentions:
                    print(
                        f"Running config: Days={days}, Reviews={reviews}, "
                        f"Retention={retention}, Burn-in={burn_in}"
                    )
                    try:
                        fitted, gt, _metrics = run_simulation(
                            n_days=days,
                            review_limit=reviews,
                            retention=retention,
                            burn_in_days=burn_in,
                            verbose=False,
                        )

                        if fitted is None:
                            print("Optimization failed. Skipping.")
                            continue

                        if not ground_truth_captured:
                            results.append({"label": "Ground Truth", "params": gt})
                            ground_truth_captured = True

                        label = (
                            f"Fit (D={days}, R={reviews}, "
                            f"Ret={retention}, BI={burn_in})"
                        )
                        results.append({"label": label, "params": fitted})
                        mse = sum(
                            (f - g) ** 2 for f, g in zip(fitted, gt, strict=False)
                        ) / len(fitted)
                        print(f"Completed config. MSE: {mse:.6f}")

                    except Exception as e:
                        print(f"Simulation failed for config: {e}")
                        traceback.print_exc()

    if len(results) > 0:
        plot_forgetting_curves(results)
    else:
        print("No results to plot.")


if __name__ == "__main__":
    main()
