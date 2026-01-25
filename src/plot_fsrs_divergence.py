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

from simulate_fsrs import parse_parameters, run_simulation


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
    gt_params: list[float] | tuple[float, ...],
    fit_params: list[float] | tuple[float, ...],
    stabilities: list[tuple[float, float]],
) -> tuple[float, float]:
    """
    Calculate RMSE and Mean KL Divergence between ground truth and fitted curves,
    averaged across all cards in the simulation.
    """
    if not stabilities:
        return 0.0, 0.0

    t_eval = np.linspace(0, 100, 200)

    # Pre-calculate nature curve parameters
    decay_nat = -gt_params[20]
    factor_nat = 0.9 ** (1 / decay_nat) - 1

    # Pre-calculate algorithm curve parameters
    decay_alg = -fit_params[20]
    factor_alg = 0.9 ** (1 / decay_alg) - 1

    all_rmse = []
    all_kl = []

    for s_nat, s_alg in stabilities:
        # Avoid division by zero
        s_nat = max(s_nat, 0.001)
        s_alg = max(s_alg, 0.001)

        # Nature curve for this card
        r_nat = (1 + factor_nat * t_eval / s_nat) ** decay_nat
        # Fitted curve for this card
        r_alg = (1 + factor_alg * t_eval / s_alg) ** decay_alg

        # RMSE for this card
        rmse = np.sqrt(np.mean((r_nat - r_alg) ** 2))
        all_rmse.append(rmse)

        # KL for this card
        p = np.clip(r_nat, 1e-10, 1 - 1e-10)
        q = np.clip(r_alg, 1e-10, 1 - 1e-10)
        kl = entropy(p, q) + entropy(1 - p, 1 - q)
        all_kl.append(np.mean(kl))

    return float(np.mean(all_rmse)), float(np.mean(all_kl))


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
        fitted, gt, metrics = run_simulation(
            n_days=task["days"],
            review_limit=task["reviews"],
            retention=task["retention"],
            burn_in_days=task["burn_in"],
            verbose=False,  # Disable inner logs in parallel
            seed=task["seed"],
            ground_truth=task.get("ground_truth"),
        )
        return {
            "config_key": task["config_key"],
            "fitted": fitted,
            "gt": gt,
            "stabilities": metrics.get("stabilities", []),
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
    parser.add_argument("--ground-truth", type=str, help="Comma-separated parameters")

    args = parser.parse_args()

    gt_params_input = parse_parameters(args.ground_truth) if args.ground_truth else None

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
                    "ground_truth": gt_params_input,
                }
            )

    t_eval = np.linspace(0, 100, 200)
    _, gt_params, _ = run_simulation(
        n_days=1, verbose=False, ground_truth=gt_params_input
    )
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
                gt_params_res = res["gt"]
                stabilities = res["stabilities"]

                # Metrics for this repeat: averaged across all cards
                rmse, kl = calculate_metrics(gt_params_res, fitted, stabilities)

                # Still calculate Scenario Curve for plotting
                fit_r = calculate_retrievability(t_eval, fitted[2], fitted)

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
