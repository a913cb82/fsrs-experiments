import random
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, cast

import numpy as np
from tqdm import tqdm

import fsrs_engine
from anki_utils import (
    START_DATE,
    infer_review_weights,
    load_anki_history,
)
from deck import Deck
from optimizer import RustOptimizer
from simulation_config import (
    SeededData,
    SimulationConfig,
)
from utils import (
    get_retention_for_day,
    parse_retention_schedule,
)

# Ensure we can import fsrs if it's not in path for some reason
try:
    from fsrs import Card, Rating, ReviewLog, Scheduler
    from fsrs.scheduler import DEFAULT_PARAMETERS
except ImportError:
    import sys

    sys.exit("Error: fsrs package not found. Please install it.")


def _batch_process_reviews_numpy(
    deck_sys: Deck,
    deck_true: Deck,
    indices: np.ndarray[Any, Any],
    nature_params: tuple[float, ...],
    algo_params: tuple[float, ...],
    current_day_idx: int,
    desired_retention: float,
    rating_estimator: Any,
    time_estimator: Any,
    card_logs_acc: dict[str, list[np.ndarray[Any, Any]]],
    start_time_accumulated: float,
    time_limit: float | None,
) -> tuple[int, float]:
    if len(indices) == 0:
        return 0, start_time_accumulated

    ratings_vals = rating_estimator(
        deck_true.stabilities[indices],
        deck_true.difficulties[indices],
        current_day_idx,
        deck_true.last_reviews[indices],
        nature_params,
    )

    times_vals = time_estimator(
        deck_true.stabilities[indices],
        deck_true.difficulties[indices],
        current_day_idx,
        deck_true.last_reviews[indices],
        nature_params,
        ratings_vals,
    )

    cumulative_times = np.cumsum(times_vals)
    if time_limit is not None:
        reviews_done = int(
            np.searchsorted(
                cumulative_times + start_time_accumulated, time_limit, side="right"
            )
        )
    else:
        reviews_done = len(indices)

    if reviews_done == 0:
        return 0, start_time_accumulated

    actual_indices = indices[:reviews_done]
    actual_ratings = ratings_vals[:reviews_done]
    actual_times = times_vals[:reviews_done]
    new_accumulated = float(start_time_accumulated + cumulative_times[reviews_done - 1])

    forget_mask = actual_ratings == 1
    recall_mask = ~forget_mask

    if np.any(recall_mask):
        idx_recall = actual_indices[recall_mask]
        rat_recall = actual_ratings[recall_mask]

        # Ground Truth
        ret_true = fsrs_engine.predict_retrievability(
            deck_true.stabilities[idx_recall],
            current_day_idx - deck_true.last_reviews[idx_recall],
            nature_params,
        )
        new_s_true, new_d_true = fsrs_engine.update_state_recall(
            deck_true.stabilities[idx_recall],
            deck_true.difficulties[idx_recall],
            rat_recall,
            ret_true,
            nature_params,
        )
        deck_true.stabilities[idx_recall] = new_s_true
        deck_true.difficulties[idx_recall] = new_d_true

        # Algo
        ret_sys = fsrs_engine.predict_retrievability(
            deck_sys.stabilities[idx_recall],
            current_day_idx - deck_sys.last_reviews[idx_recall],
            algo_params,
        )
        new_s_sys, new_d_sys = fsrs_engine.update_state_recall(
            deck_sys.stabilities[idx_recall],
            deck_sys.difficulties[idx_recall],
            rat_recall,
            ret_sys,
            algo_params,
        )
        deck_sys.stabilities[idx_recall] = new_s_sys
        deck_sys.difficulties[idx_recall] = new_d_sys

        intervals = fsrs_engine.next_interval(new_s_sys, desired_retention, algo_params)
        deck_sys.dues[idx_recall] = current_day_idx + intervals
        deck_sys.last_reviews[idx_recall] = current_day_idx
        deck_true.last_reviews[idx_recall] = current_day_idx

    if np.any(forget_mask):
        idx_forget = actual_indices[forget_mask]

        # Ground Truth
        ret_true = fsrs_engine.predict_retrievability(
            deck_true.stabilities[idx_forget],
            current_day_idx - deck_true.last_reviews[idx_forget],
            nature_params,
        )
        new_s_true, new_d_true = fsrs_engine.update_state_forget(
            deck_true.stabilities[idx_forget],
            deck_true.difficulties[idx_forget],
            ret_true,
            nature_params,
        )
        deck_true.stabilities[idx_forget] = new_s_true
        deck_true.difficulties[idx_forget] = new_d_true

        # Algo
        ret_sys = fsrs_engine.predict_retrievability(
            deck_sys.stabilities[idx_forget],
            current_day_idx - deck_sys.last_reviews[idx_forget],
            algo_params,
        )
        new_s_sys, new_d_sys = fsrs_engine.update_state_forget(
            deck_sys.stabilities[idx_forget],
            deck_sys.difficulties[idx_forget],
            ret_sys,
            algo_params,
        )
        deck_sys.stabilities[idx_forget] = new_s_sys
        deck_sys.difficulties[idx_forget] = new_d_sys

        deck_sys.dues[idx_forget] = current_day_idx + 1
        deck_sys.last_reviews[idx_forget] = current_day_idx
        deck_true.last_reviews[idx_forget] = current_day_idx

    # Logging - Append NumPy arrays to avoid Python scalar overhead
    card_logs_acc["card_id"].append(deck_sys.card_ids[actual_indices])
    card_logs_acc["rating"].append(actual_ratings)
    card_logs_acc["day"].append(np.full(reviews_done, current_day_idx, dtype=np.int32))
    card_logs_acc["duration"].append(actual_times)

    return reviews_done, new_accumulated


def simulate_day_numpy(
    nature_params: tuple[float, ...],
    algo_params: tuple[float, ...],
    deck_true: Deck,
    deck_sys: Deck,
    card_logs_acc: dict[str, list[np.ndarray[Any, Any]]],
    current_day_idx: int,
    config: SimulationConfig,
    desired_retention: float,
    rating_estimator: Any,
    time_estimator: Any,
    next_card_id: int,
) -> int:
    due_mask = deck_sys.current_dues <= current_day_idx
    due_indices = np.where(due_mask)[0]

    if len(due_indices) > 0:
        np.random.shuffle(due_indices)
        if config.review_limit is not None:
            due_indices = due_indices[: config.review_limit]

        reviews_done, time_accumulated = _batch_process_reviews_numpy(
            deck_sys,
            deck_true,
            due_indices,
            nature_params,
            algo_params,
            current_day_idx,
            desired_retention,
            rating_estimator,
            time_estimator,
            card_logs_acc,
            0.0,
            config.time_limit,
        )
    else:
        reviews_done = 0
        time_accumulated = 0.0

    new_done = 0
    new_batch_size = 50
    while True:
        if config.time_limit is not None and time_accumulated >= config.time_limit:
            break

        remaining_total = (
            (config.review_limit - reviews_done)
            if config.review_limit is not None
            else 999999
        )
        remaining_new = (
            (config.new_limit - new_done) if config.new_limit is not None else 999999
        )

        if remaining_total <= 0 or remaining_new <= 0:
            break

        batch_size = min(new_batch_size, remaining_total, remaining_new)
        new_ids = np.arange(next_card_id, next_card_id + batch_size, dtype=np.int64)
        next_card_id += int(batch_size)

        init_ratings = rating_estimator(
            np.zeros(batch_size),
            np.zeros(batch_size),
            current_day_idx,
            np.full(batch_size, -1),
            nature_params,
        )

        s_true = fsrs_engine.init_stability(init_ratings, nature_params)
        d_true = fsrs_engine.init_difficulty(init_ratings, nature_params)
        s_sys = fsrs_engine.init_stability(init_ratings, algo_params)
        d_sys = fsrs_engine.init_difficulty(init_ratings, algo_params)

        intervals = fsrs_engine.next_interval(s_sys, desired_retention, algo_params)
        dues = current_day_idx + intervals
        dues[init_ratings == 1] = current_day_idx + 1

        times = time_estimator(
            np.zeros(batch_size),  # stability 0
            np.zeros(batch_size),  # difficulty 0
            current_day_idx,
            np.full(batch_size, -1),  # last review -1
            nature_params,
            init_ratings,
        )

        # Check time limit within the new cards batch
        cumulative_times = np.cumsum(times)
        if config.time_limit is not None:
            batch_completed = int(
                np.searchsorted(
                    cumulative_times + time_accumulated,
                    config.time_limit,
                    side="right",
                )
            )
        else:
            batch_completed = int(batch_size)

        if batch_completed == 0:
            break

        # Slice to actual completed
        init_ratings = init_ratings[:batch_completed]
        new_ids = cast(np.ndarray[Any, Any], new_ids[:batch_completed])
        times = times[:batch_completed]

        s_true = s_true[:batch_completed]
        d_true = d_true[:batch_completed]
        s_sys = s_sys[:batch_completed]
        d_sys = d_sys[:batch_completed]
        dues = dues[:batch_completed]

        deck_true.add_cards(
            new_ids,
            s_true,
            d_true,
            np.full(batch_completed, current_day_idx),
            np.full(batch_completed, current_day_idx),
        )
        deck_sys.add_cards(
            new_ids, s_sys, d_sys, dues, np.full(batch_completed, current_day_idx)
        )

        # Log these new cards
        card_logs_acc["card_id"].append(new_ids)
        card_logs_acc["rating"].append(init_ratings)
        card_logs_acc["day"].append(
            np.full(batch_completed, current_day_idx, dtype=np.int32)
        )
        card_logs_acc["duration"].append(times)

        time_accumulated += float(np.sum(times))
        reviews_done += batch_completed
        new_done += batch_completed

        if batch_completed < batch_size:
            break

    return next_card_id


def _get_estimators_numpy(
    config: SimulationConfig,
    card_logs: dict[int, list[ReviewLog]],
    nature_params: tuple[float, ...],
) -> tuple[Any, Any]:
    weights = infer_review_weights(card_logs)
    flat_logs = [log for logs_list in card_logs.values() for log in logs_list]
    duration_data = defaultdict(list)
    for log in flat_logs:
        if log.review_duration is not None:
            duration_data[int(log.rating)].append(log.review_duration)
    avg_durations = np.zeros(5)
    for r in [1, 2, 3, 4]:
        durs = duration_data[r]
        avg_durations[r] = float(np.mean(durs)) if durs else 0.0

    # Handle Custom Rating Estimator
    if config.rating_estimator:
        rating_est = config.rating_estimator
    else:

        def rating_est(
            stabilities: np.ndarray[Any, Any],
            difficulties: np.ndarray[Any, Any],
            date_days: int | np.ndarray[Any, Any],
            last_reviews: np.ndarray[Any, Any],
            params: tuple[float, ...],
        ) -> np.ndarray[Any, Any]:
            count = len(stabilities)
            results = np.zeros(count, dtype=np.int8)
            new_mask = last_reviews == -1
            review_mask = ~new_mask
            if np.any(new_mask):
                results[new_mask] = np.random.choice(
                    [1, 2, 3, 4], size=np.sum(new_mask), p=weights.first
                )
            if np.any(review_mask):
                n_rev = np.sum(review_mask)
                rets = fsrs_engine.predict_retrievability(
                    stabilities[review_mask],
                    date_days - last_reviews[review_mask],
                    params,
                )
                success = np.random.random(n_rev) < rets
                results_rev = np.ones(n_rev, dtype=np.int8)
                n_success = np.sum(success)
                if n_success > 0:
                    results_rev[success] = np.random.choice(
                        [2, 3, 4], size=n_success, p=weights.success
                    )
                results[review_mask] = results_rev
            return results

    # Handle Custom Time Estimator
    if config.time_estimator:
        time_est = config.time_estimator
    else:

        def time_est(
            stabilities: np.ndarray[Any, Any],
            difficulties: np.ndarray[Any, Any],
            day_idx: int | np.ndarray[Any, Any],
            last_reviews: np.ndarray[Any, Any],
            params: tuple[float, ...],
            ratings: np.ndarray[Any, Any],
        ) -> np.ndarray[Any, Any]:
            return cast(np.ndarray[Any, Any], avg_durations[ratings])

    return rating_est, time_est

    return rating_est, time_est


def run_simulation(
    config: SimulationConfig,
    ground_truth: tuple[float, ...] | None = None,
    initial_params: tuple[float, ...] | None = None,
    seeded_data: SeededData | None = None,
    seed_history: str | None = None,
    deck_config: str | None = None,
    deck_name: str | None = None,
    tqdm_pos: int = 0,
) -> tuple[list[float] | None, tuple[float, ...], dict[str, Any]]:
    parsed_schedule = parse_retention_schedule(config.retention)
    np.random.seed(config.seed)
    random.seed(config.seed)

    gt_params = ground_truth if ground_truth is not None else DEFAULT_PARAMETERS
    algo_params = initial_params or DEFAULT_PARAMETERS

    import fsrs.optimizer

    original_tqdm = fsrs.optimizer.tqdm

    def patched_tqdm(*args: Any, **kwargs: Any) -> Any:
        if "position" not in kwargs:
            kwargs["position"] = tqdm_pos + 1
        kwargs["leave"] = False
        return original_tqdm(*args, **kwargs)

    fsrs.optimizer.tqdm = patched_tqdm

    try:
        init_true, init_sys, initial_card_logs, _ = _load_initial_state(
            Scheduler(parameters=gt_params),
            Scheduler(parameters=algo_params),
            seeded_data,
            seed_history,
            deck_config,
            deck_name,
        )
        deck_true = Deck.from_cards(init_true)
        deck_sys = Deck.from_cards(init_sys)

        rating_est, time_est = _get_estimators_numpy(
            config, initial_card_logs, gt_params
        )
        card_logs_acc: dict[str, list[np.ndarray[Any, Any]]] = {
            "card_id": [],
            "rating": [],
            "day": [],
            "duration": [],
        }
        next_card_id = (
            int(np.max(deck_sys.current_card_ids)) + 1
            if len(deck_sys) > 0
            else int(datetime.now().timestamp() * 1000)
        )

        # Pre-construct items for initial logs
        initial_logs_flat = [log for logs in initial_card_logs.values() for log in logs]
        opt_init = RustOptimizer(review_logs=initial_logs_flat)
        seeded_items = opt_init.get_items()

        phases = [
            (
                "P1",
                config.burn_in_days
                if 0 < config.burn_in_days < config.n_days
                else config.n_days,
                0,
            )
        ]
        if 0 < config.burn_in_days < config.n_days:
            phases.append(("P2", config.n_days, config.burn_in_days))

        for phase_name, limit, start_day in phases:
            pbar = tqdm(
                range(start_day, limit),
                desc=f"Simulating ({phase_name})",
                position=tqdm_pos,
                leave=False,
                disable=not config.verbose,
            )
            for day in pbar:
                ret = get_retention_for_day(day, parsed_schedule)
                next_card_id = simulate_day_numpy(
                    gt_params,
                    algo_params,
                    deck_true,
                    deck_sys,
                    card_logs_acc,
                    day,
                    config,
                    ret,
                    rating_est,
                    time_est,
                    next_card_id,
                )

            if phase_name == "P1" and 0 < config.burn_in_days < config.n_days:
                # Re-fit algo scheduler
                if card_logs_acc["card_id"]:
                    all_card_ids = np.concatenate(card_logs_acc["card_id"])
                    all_ratings = np.concatenate(card_logs_acc["rating"])
                    all_days = np.concatenate(card_logs_acc["day"])

                    optimizer = RustOptimizer(pre_constructed_items=seeded_items)
                    new_params = optimizer.compute_optimal_parameters_from_arrays(
                        all_card_ids, all_ratings, all_days, verbose=config.verbose
                    )
                    if new_params:
                        algo_params = tuple(new_params)

        # Optimization: Use optimized optimizer call
        total_simulated_reviews = 0
        all_card_ids = None
        if card_logs_acc["card_id"]:
            all_card_ids = np.concatenate(card_logs_acc["card_id"])
            total_simulated_reviews = len(all_card_ids)

        if config.compute_final_params:
            if all_card_ids is not None:
                all_ratings = np.concatenate(card_logs_acc["rating"])
                all_days = np.concatenate(card_logs_acc["day"])

                optimizer = RustOptimizer(pre_constructed_items=seeded_items)
                fitted_params = optimizer.compute_optimal_parameters_from_arrays(
                    all_card_ids, all_ratings, all_days, verbose=config.verbose
                )
            else:
                optimizer = RustOptimizer(pre_constructed_items=seeded_items)
                fitted_params = optimizer.compute_optimal_parameters(
                    verbose=config.verbose
                )
        else:
            fitted_params = None

        # Final metrics
        all_logs_final = []
        if config.return_logs:
            if all_card_ids is not None:
                all_ratings = np.concatenate(card_logs_acc["rating"])
                all_days = np.concatenate(card_logs_acc["day"])
                all_durations = np.concatenate(card_logs_acc["duration"])

                # This loop is still slow, but we've already optimized the hot loop
                for i in range(len(all_card_ids)):
                    all_logs_final.append(
                        ReviewLog(
                            card_id=int(all_card_ids[i]),
                            rating=Rating(int(all_ratings[i])),
                            review_datetime=START_DATE
                            + timedelta(days=int(all_days[i])),
                            review_duration=int(all_durations[i]),
                        )
                    )

        # total_logs is used by metrics["logs"] and weights calculation
        # If we didn't return logs, total_logs will just be the initial logs
        initial_logs_flat = [log for logs in initial_card_logs.values() for log in logs]
        total_logs = initial_logs_flat + all_logs_final
        metrics = {
            "review_count": len(initial_logs_flat) + total_simulated_reviews,
            "card_count": len(deck_true),
            "stabilities": (
                deck_true.current_stabilities,
                deck_sys.current_stabilities,
            ),
            "total_retention": np.sum(
                fsrs_engine.predict_retrievability(
                    deck_true.current_stabilities,
                    config.n_days - deck_true.current_last_reviews,
                    gt_params,
                )
            ),
            "logs": total_logs,
        }
        return fitted_params, gt_params, metrics
    finally:
        fsrs.optimizer.tqdm = original_tqdm


def _load_initial_state(
    nature_scheduler: Scheduler,
    algo_scheduler: Scheduler,
    seeded_data: SeededData | None,
    seed_history: str | None,
    deck_config: str | None,
    deck_name: str | None,
) -> tuple[list[Card], list[Card], dict[int, list[ReviewLog]], datetime]:
    true_cards, sys_cards, card_logs = [], [], defaultdict(list)
    if seeded_data:
        current_date = seeded_data.last_rev + timedelta(days=1)
        true_cards, sys_cards, card_logs = (
            list(deepcopy(seeded_data.true_cards).values()),
            list(deepcopy(seeded_data.sys_cards).values()),
            deepcopy(seeded_data.logs),
        )
    elif seed_history:
        logs, last_rev = load_anki_history(seed_history, deck_config, deck_name)
        current_date = last_rev + timedelta(days=1)
        for cid, logs_list in logs.items():
            card = Card(card_id=cid)
            true_cards.append(nature_scheduler.reschedule_card(card, logs_list))
            sys_cards.append(algo_scheduler.reschedule_card(card, logs_list))
            card_logs[cid].extend(logs_list)
    else:
        current_date = START_DATE
    return true_cards, sys_cards, card_logs, current_date
