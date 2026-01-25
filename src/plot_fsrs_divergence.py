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


def calculate_metrics(
    gt_r: np.ndarray[Any, Any], fitted_r: np.ndarray[Any, Any]
) -> tuple[float, float]:
    """
    Calculate RMSE and Mean KL Divergence between ground truth and fitted curves.
    KL divergence is calculated between Bernoulli distributions at each time point.
    """
    # RMSE
    rmse = float(np.sqrt(np.mean((gt_r - fitted_r) ** 2)))

    # KL Divergence between Bernoulli distributions at each t
    # p = ground truth, q = fitted
    p = np.clip(gt_r, 1e-10, 1 - 1e-10)
    q = np.clip(fitted_r, 1e-10, 1 - 1e-10)

    kl = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    mean_kl = float(np.mean(kl))

    return rmse, mean_kl


def plot_forgetting_curves(results: list[dict[str, Any]]) -> None:
    """
    results: list of dicts with keys 'label', 'r_values', 'rmse', 'kl'
    """
    plt.figure(figsize=(12, 8))

    # Time range (days)
    t = np.linspace(0, 100, 200)

    for res in results:
        label = res["label"]
        r_values = res["r_values"]

        if label != "Ground Truth":
            rmse = res["rmse"]
            kl = res["kl"]
            label = f"{label} (avg RMSE: {rmse:.4f}, avg KL: {kl:.4f})"

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
    parser.add_argument(
        "--repeats", type=int, default=1, help="Number of repeats per config"
    )

    args = parser.parse_args()

    results = []
    t_eval = np.linspace(0, 100, 200)

    # Capture Ground Truth once
    _, gt_params, _ = run_simulation(n_days=1, verbose=False)
    gt_r = calculate_retrievability(t_eval, gt_params[2], gt_params)
    results.append({"label": "Ground Truth", "r_values": gt_r})

    # Iterate over all combinations
    for burn_in in args.burn_ins:
        for days in args.days:
            for reviews in args.reviews:
                for retention in args.retentions:
                    print(
                        f"Running config: Days={days}, Reviews={reviews}, "
                        f"Retention={retention}, Burn-in={burn_in}, "
                        f"Repeats={args.repeats}"
                    )

                    all_fit_r = []
                    all_rmse = []
                    all_kl = []

                    for i in range(args.repeats):
                        try:
                            # Use different seed for each repeat
                            fitted, gt, _metrics = run_simulation(
                                n_days=days,
                                review_limit=reviews,
                                retention=retention,
                                burn_in_days=burn_in,
                                verbose=False,
                                seed=42 + i,
                            )

                            if fitted is None:
                                print(f"Repeat {i} optimization failed. Skipping.")
                                continue

                            fit_r = calculate_retrievability(t_eval, fitted[2], fitted)
                            rmse, kl = calculate_metrics(gt_r, fit_r)

                            all_fit_r.append(fit_r)
                            all_rmse.append(rmse)
                            all_kl.append(kl)

                        except Exception as e:
                            print(f"Repeat {i} failed for config: {e}")
                            traceback.print_exc()

                    if not all_fit_r:
                        print("All repeats failed for this config. Skipping.")
                        continue

                    # Average the curves
                    avg_fit_r = np.mean(all_fit_r, axis=0)
                    # Average the metrics
                    avg_rmse = float(np.mean(all_rmse))
                    avg_kl = float(np.mean(all_kl))

                    label = (
                        f"Fit (D={days}, R={reviews}, Ret={retention}, BI={burn_in})"
                    )
                    results.append(
                        {
                            "label": label,
                            "r_values": avg_fit_r,
                            "rmse": avg_rmse,
                            "kl": avg_kl,
                        }
                    )
                    print(
                        f"Completed config. avg RMSE: {avg_rmse:.6f}, "
                        f"avg KL: {avg_kl:.6f}"
                    )

    if len(results) > 1:
        plot_forgetting_curves(results)
    else:
        print("No results to plot.")


if __name__ == "__main__":
    main()
