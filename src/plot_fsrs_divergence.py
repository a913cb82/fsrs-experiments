import argparse
import concurrent.futures
import itertools
import multiprocessing
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm

try:
    from simulate_fsrs import (
        DEFAULT_PARAMETERS,
        Card,
        ReviewLog,
        RustOptimizer,
        Scheduler,
        load_anki_history,
        parse_parameters,
        run_simulation,
    )
except ImportError:
    try:
        from .simulate_fsrs import (
            DEFAULT_PARAMETERS,
            Card,
            ReviewLog,
            RustOptimizer,
            Scheduler,
            load_anki_history,
            parse_parameters,
            run_simulation,
        )
    except ImportError:
        # Fallback for direct execution when src is not in sys.path
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from simulate_fsrs import (
            DEFAULT_PARAMETERS,
            Card,
            ReviewLog,
            RustOptimizer,
            Scheduler,
            load_anki_history,
            parse_parameters,
            run_simulation,
        )


def calculate_population_retrievability(
    t: np.ndarray[Any, Any],
    stabilities: np.ndarray[Any, Any],
    parameters: list[float] | tuple[float, ...],
) -> np.ndarray[Any, Any]:
    """
    Calculate the aggregate (average) forgetting curve for a population.
    Returns a 1D array of retrievability values for each time point in t.
    """
    if len(stabilities) == 0:
        res_ones: np.ndarray[Any, Any] = np.ones_like(t)
        return res_ones

    decay = -parameters[20]
    factor = 0.9 ** (1 / decay) - 1

    s_safe = np.maximum(stabilities, 0.001)

    # Broadcast: (T, 1) and (1, N) -> (T, N)
    r_matrix = (1 + factor * t[:, np.newaxis] / s_safe[np.newaxis, :]) ** decay

    # Average across the population
    res_mean: np.ndarray[Any, Any] = np.mean(r_matrix, axis=1)
    return res_mean


def calculate_metrics(
    gt_params: list[float] | tuple[float, ...],
    fit_params: list[float] | tuple[float, ...],
    stabilities: list[tuple[float, float]],
) -> tuple[float, float]:
    """
    Calculate RMSE and Mean KL Divergence between ground truth and fitted curves,
    averaged across all individual cards in the simulation.
    """
    if not stabilities:
        return 0.0, 0.0

    t_eval = np.linspace(0, 100, 200)

    # Pre-calculate constants
    decay_nat = -gt_params[20]
    factor_nat = 0.9 ** (1 / decay_nat) - 1
    decay_alg = -fit_params[20]
    factor_alg = 0.9 ** (1 / decay_alg) - 1

    all_rmse = []
    all_kl = []

    for s_nat, s_alg in stabilities:
        s_nat = max(s_nat, 0.001)
        s_alg = max(s_alg, 0.001)

        r_nat = (1 + factor_nat * t_eval / s_nat) ** decay_nat
        r_alg = (1 + factor_alg * t_eval / s_alg) ** decay_alg

        # Per-card RMSE
        rmse = np.sqrt(np.mean((r_nat - r_alg) ** 2))
        all_rmse.append(rmse)

        # Per-card KL
        p = np.clip(r_nat, 1e-10, 1 - 1e-10)
        q = np.clip(r_alg, 1e-10, 1 - 1e-10)
        kl = entropy(p, q) + entropy(1 - p, 1 - q)
        all_kl.append(np.mean(kl))

    return float(np.mean(all_rmse)), float(np.mean(all_kl))


def plot_forgetting_curves(results: list[dict[str, Any]]) -> None:
    """
    results: list of dicts with keys 'label', 'r_fit_avg', 'r_fit_std',
    'r_nat_avg', 'rmse', 'kl'
    """
    plt.figure(figsize=(12, 8))
    t = np.linspace(0, 100, 200)

    for res in results:
        label = res["label"]
        r_fit = res["r_fit_avg"]
        r_fit_std = res.get("r_fit_std")
        r_nat = res["r_nat_avg"]

        rmse = res["rmse"]
        kl = res["kl"]

        # Plot Model (Solid)
        algo_label = f"{label} (avg RMSE: {rmse:.4f}, KL: {kl:.4f})"
        (line,) = plt.plot(t, r_fit, label=algo_label, linestyle="-")

        # Plot Nature (Dashed, same color)
        plt.plot(t, r_nat, linestyle="--", color=line.get_color(), alpha=0.6)

        # Shading for repeat variability
        if r_fit_std is not None and np.any(r_fit_std > 0):
            plt.fill_between(
                t,
                r_fit - r_fit_std,
                r_fit + r_fit_std,
                color=line.get_color(),
                alpha=0.15,
            )

    # Explain styles
    plt.plot([], [], color="gray", linestyle="-", label="Model (Predicted)")
    plt.plot([], [], color="gray", linestyle="--", label="Nature (Actual)")

    plt.xlabel("Days since end of simulation")
    plt.ylabel("Aggregate Expected Retention")
    plt.title("Aggregate Forgetting Curve Divergence")
    plt.legend(fontsize="small", ncol=1)
    plt.grid(True, alpha=0.3)
    plt.savefig("forgetting_curve_divergence.png")
    tqdm.write("Plot saved to forgetting_curve_divergence.png")


def run_single_task(task: dict[str, Any]) -> dict[str, Any]:
    try:
        fitted, gt, metrics = run_simulation(
            n_days=task["days"],
            review_limit=task["reviews"],
            retention=task["retention"],
            burn_in_days=task["burn_in"],
            verbose=False,
            seed=task["seed"],
            ground_truth=task.get("ground_truth"),
            initial_params=task.get("initial_params"),
            seeded_data=task.get("seeded_data"),
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
    parser.add_argument("--seed-history", type=str, help="Path to collection.anki2")
    parser.add_argument("--deck-config", type=str, help="Anki deck options preset name")
    parser.add_argument("--deck-name", type=str, help="Anki deck name")

    args = parser.parse_args()

    gt_params_input = (
        parse_parameters(args.ground_truth) if args.ground_truth else DEFAULT_PARAMETERS
    )

    # 1. Load history ONCE and pre-calculate initial states
    initial_params: tuple[float, ...] | None = None
    seeded_data: tuple[dict[int, list[ReviewLog]], datetime] | None = None
    initial_card_states: tuple[dict[int, Card], dict[int, Card]] | None = None

    if args.seed_history:
        tqdm.write(f"Loading Anki history from {args.seed_history}...")
        logs, last_rev = load_anki_history(
            args.seed_history, args.deck_config, args.deck_name
        )
        if logs:
            seeded_data = (logs, last_rev)
            flat_logs = [log for card_logs in logs.values() for log in card_logs]

            # Pre-fit initial parameters
            if len(flat_logs) >= 512:
                tqdm.write("Pre-fitting initial parameters from history...")
                initial_params = tuple(
                    RustOptimizer(flat_logs).compute_optimal_parameters()
                )
                tqdm.write(f"Initial params fitted: {initial_params}")
            else:
                tqdm.write("Not enough logs for pre-fitting. Using defaults.")

            # Pre-calculate card states (History Replay)
            tqdm.write("Pre-calculating initial card states (replaying history)...")
            nat_sch = Scheduler(parameters=gt_params_input)
            alg_sch = Scheduler(parameters=initial_params or DEFAULT_PARAMETERS)

            true_cards: dict[int, Card] = {}
            sys_cards: dict[int, Card] = {}
            for cid, card_logs in logs.items():
                true_cards[cid] = nat_sch.reschedule_card(Card(card_id=cid), card_logs)
                sys_cards[cid] = alg_sch.reschedule_card(Card(card_id=cid), card_logs)
            initial_card_states = (true_cards, sys_cards)

    # Combine pre-calculated data
    seeded_payload = None
    if seeded_data and initial_card_states:
        seeded_payload = {
            "logs": seeded_data[0],
            "last_rev": seeded_data[1],
            "true_cards": initial_card_states[0],
            "sys_cards": initial_card_states[1],
        }

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
                    "initial_params": initial_params,
                    "seeded_data": seeded_payload,
                }
            )

    t_eval = np.linspace(0, 100, 200)
    _, gt_params, baseline_metrics = run_simulation(
        n_days=1,
        verbose=False,
        ground_truth=gt_params_input,
        initial_params=initial_params,
        seeded_data=seeded_payload,
    )
    s_nat_base = np.array([s[0] for s in baseline_metrics.get("stabilities", [])])
    gt_r_base = calculate_population_retrievability(t_eval, s_nat_base, gt_params_input)

    # Map to store population results for averaging
    aggregated_r_fit = defaultdict(list)
    aggregated_r_nat = defaultdict(list)
    aggregated_metrics = defaultdict(list)

    tqdm.write(f"Starting {len(tasks)} simulations...")

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
                key = res["config_key"]

                # Metrics for this repeat: averaged across all cards
                rmse, kl = calculate_metrics(gt_params_res, fitted, stabilities)

                # Calculate population curves for this run
                s_nat_list = np.array([s[0] for s in stabilities])
                s_alg_list = np.array([s[1] for s in stabilities])

                r_nat_agg = calculate_population_retrievability(
                    t_eval, s_nat_list, gt_params_res
                )
                r_fit_agg = calculate_population_retrievability(
                    t_eval, s_alg_list, fitted
                )

                aggregated_r_fit[key].append(r_fit_agg)
                aggregated_r_nat[key].append(r_nat_agg)
                aggregated_metrics[key].append((rmse, kl))
            elif "error" in res:
                tqdm.write(f"Task failed: {res['error']}")

    final_results = [
        {
            "label": "Ground Truth",
            "r_fit_avg": gt_r_base,
            "r_nat_avg": gt_r_base,
            "rmse": 0.0,
            "kl": 0.0,
        }
    ]
    for key in aggregated_r_fit:
        burn_in, days, reviews, retention = key

        # Average across repeats
        avg_fit_r = np.mean(aggregated_r_fit[key], axis=0)
        std_fit_r = np.std(aggregated_r_fit[key], axis=0)
        avg_nat_r = np.mean(aggregated_r_nat[key], axis=0)

        avg_rmse = float(np.mean([m[0] for m in aggregated_metrics[key]]))
        avg_kl = float(np.mean([m[1] for m in aggregated_metrics[key]]))

        label = f"Ret={retention}, BI={burn_in}"
        final_results.append(
            {
                "label": label,
                "r_fit_avg": avg_fit_r,
                "r_fit_std": std_fit_r,
                "r_nat_avg": avg_nat_r,
                "rmse": avg_rmse,
                "kl": avg_kl,
            }
        )

    if final_results:
        plot_forgetting_curves(final_results)
    else:
        tqdm.write("No results to plot.")


if __name__ == "__main__":
    main()
