import argparse
import concurrent.futures
import itertools
import multiprocessing
from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm

from simulate_fsrs import run_simulation


def calculate_retrievability(
    t: np.ndarray[Any, Any],
    stability: float,
    parameters: list[float] | tuple[float, ...],
) -> np.ndarray[Any, Any]:
    decay = -parameters[20]
    factor = 0.9 ** (1 / decay) - 1
    res: np.ndarray[Any, Any] = (1 + factor * t / stability) ** decay
    return res


def calculate_metrics(
    gt_r: np.ndarray[Any, Any], fitted_r: np.ndarray[Any, Any]
) -> tuple[float, float]:
    """
    Calculate RMSE and Mean KL Divergence between ground truth and fitted curves.
    """
    # RMSE
    rmse = float(np.sqrt(np.mean((gt_r - fitted_r) ** 2)))

    # KL Divergence between Bernoulli distributions at each t
    p = np.clip(gt_r, 1e-10, 1 - 1e-10)
    q = np.clip(fitted_r, 1e-10, 1 - 1e-10)

    # KL(P || Q) for Bernoulli is sum of KL for each outcome
    kl = entropy(p, q) + entropy(1 - p, 1 - q)
    mean_kl = float(np.mean(kl))

    return rmse, mean_kl


def plot_forgetting_curves(results: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(12, 8))
    t = np.linspace(0, 100, 200)

    for res in results:
        label = res["label"]
        r_values = res["r_values"]
        r_std = res.get("r_std")

        if label != "Ground Truth":
            rmse = res["rmse"]
            kl = res["kl"]
            label = f"{label} (avg RMSE: {rmse:.4f}, avg KL: {kl:.4f})"

        (line,) = plt.plot(t, r_values, label=label)

        if r_std is not None and np.any(r_std > 0):
            plt.fill_between(
                t,
                r_values - r_std,
                r_values + r_std,
                color=line.get_color(),
                alpha=0.2,
            )

    plt.xlabel("Days since review")
    plt.ylabel("Probability of Recall (Retrievability)")
    plt.title("Forgetting Curve Divergence (Scenario: First Review = Good)")
    plt.legend()
    plt.grid(True)
    plt.savefig("forgetting_curve_divergence.png")
    tqdm.write("Plot saved to forgetting_curve_divergence.png")


def run_single_task(task: dict[str, Any]) -> dict[str, Any]:
    """Worker function for multiprocessing."""
    try:
        fitted, gt, _metrics = run_simulation(
            n_days=task["days"],
            review_limit=task["reviews"],
            retention=task["retention"],
            burn_in_days=task["burn_in"],
            verbose=False,  # Disable inner logs in parallel
            seed=task["seed"],
        )
        return {
            "config_key": task["config_key"],
            "fitted": fitted,
            "gt": gt,
            "success": fitted is not None,
        }
    except Exception as e:
        return {"config_key": task["config_key"], "success": False, "error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simulations and plot divergence")
    parser.add_argument("--days", type=int, nargs="+", default=[365])
    parser.add_argument("--reviews", type=int, nargs="+", default=[200])
    parser.add_argument("--retentions", type=str, nargs="+", default=["0.9"])
    parser.add_argument("--burn-ins", type=int, nargs="+", default=[0])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=multiprocessing.cpu_count())

    args = parser.parse_args()

    # Flatten configurations into individual tasks
    tasks = []
    for burn_in, days, reviews, retention in itertools.product(
        args.burn_ins, args.days, args.reviews, args.retentions
    ):
        config_key = (burn_in, days, reviews, retention)
        for i in range(args.repeats):
            tasks.append(
                {
                    "burn_in": burn_in,
                    "days": days,
                    "reviews": reviews,
                    "retention": retention,
                    "seed": 42 + i,
                    "config_key": config_key,
                }
            )

    t_eval = np.linspace(0, 100, 200)
    _, gt_params, _ = run_simulation(n_days=1, verbose=False)
    gt_r = calculate_retrievability(t_eval, gt_params[2], gt_params)

    # Map to store list of fit_r for each config
    aggregated_results: dict[tuple[Any, ...], list[np.ndarray[Any, Any]]] = defaultdict(
        list
    )
    aggregated_metrics: dict[tuple[Any, ...], list[tuple[float, float]]] = defaultdict(
        list
    )

    tqdm.write(f"Starting {len(tasks)} simulations using {args.concurrency} workers...")

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = {executor.submit(run_single_task, task): task for task in tasks}

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(tasks),
            desc="Simulating",
        ):
            res = future.result()
            if res["success"]:
                fitted = res["fitted"]
                fit_r = calculate_retrievability(t_eval, fitted[2], fitted)
                rmse, kl = calculate_metrics(gt_r, fit_r)

                key = res["config_key"]
                aggregated_results[key].append(fit_r)
                aggregated_metrics[key].append((rmse, kl))
            elif "error" in res:
                tqdm.write(f"Task failed: {res['error']}")

    # Process and average results
    final_results = [{"label": "Ground Truth", "r_values": gt_r}]

    for key, fit_rs in aggregated_results.items():
        burn_in, days, reviews, retention = key
        metrics = aggregated_metrics[key]

        avg_fit_r = np.mean(fit_rs, axis=0)
        std_fit_r = np.std(fit_rs, axis=0)
        avg_rmse = float(np.mean([m[0] for m in metrics]))
        avg_kl = float(np.mean([m[1] for m in metrics]))

        label = f"Fit (D={days}, R={reviews}, Ret={retention}, BI={burn_in})"
        final_results.append(
            {
                "label": label,
                "r_values": avg_fit_r,
                "r_std": std_fit_r,
                "rmse": avg_rmse,
                "kl": avg_kl,
            }
        )

    if len(final_results) > 1:
        plot_forgetting_curves(final_results)
    else:
        tqdm.write("No results to plot.")


if __name__ == "__main__":
    main()
