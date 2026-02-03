import random
from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from tqdm import tqdm

from anki_utils import (
    START_DATE,
    infer_review_weights,
    load_anki_history,
)
from optimizer import RustOptimizer
from simulation_config import (
    RatingEstimator,
    SeededData,
    SimulationConfig,
    TimeEstimator,
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


def _batch_process_reviews(
    sys_cards_batch: Sequence[Card],
    true_cards_batch: Sequence[Card],
    nature_scheduler: Scheduler,
    algo_scheduler: Scheduler,
    current_date: datetime,
    rating_estimator: RatingEstimator,
    time_estimator: TimeEstimator,
    card_logs: dict[int, list[ReviewLog]],
    start_time_accumulated: float,
    time_limit: float | None,
) -> tuple[int, float, list[Card], list[Card]]:
    """
    Process a batch of reviews.
    Returns:
        reviews_done: Number of reviews actually completed (due to time limit)
        new_time_accumulated: Updated time
        updated_sys_cards: List of updated system cards (length = reviews_done)
        updated_true_cards: List of updated true cards (length = reviews_done)
    """
    if not sys_cards_batch:
        return 0, start_time_accumulated, [], []

    # 1. Predict Ratings (Batch)
    ratings_vals = rating_estimator(true_cards_batch, current_date, nature_scheduler)

    # 2. Predict Times (Batch)
    times_vals = time_estimator(
        true_cards_batch, current_date, ratings_vals, nature_scheduler
    )

    current_accumulated = start_time_accumulated
    reviews_done = 0
    updated_sys_cards = []
    updated_true_cards = []

    for i in range(len(sys_cards_batch)):
        sys_card = sys_cards_batch[i]
        true_card = true_cards_batch[i]
        r_val = ratings_vals[i]
        t_val = times_vals[i]

        # Check time limit
        if time_limit is not None and current_accumulated + t_val > time_limit:
            break

        current_accumulated += t_val
        reviews_done += 1

        rating = Rating(r_val)
        updated_true_card, _ = nature_scheduler.review_card(
            true_card, rating, current_date
        )
        updated_sys_card, log = algo_scheduler.review_card(
            sys_card, rating, current_date
        )

        card_logs[updated_sys_card.card_id].append(log)
        updated_sys_cards.append(updated_sys_card)
        updated_true_cards.append(updated_true_card)

    return reviews_done, current_accumulated, updated_sys_cards, updated_true_cards


def simulate_day(
    nature_scheduler: Scheduler,
    algo_scheduler: Scheduler,
    true_cards: list[Card],
    sys_cards: list[Card],
    card_logs: dict[int, list[ReviewLog]],
    current_date: datetime,
    config: SimulationConfig,
    rating_estimator: RatingEstimator,
    time_estimator: TimeEstimator,
    next_card_id: int,
) -> int:
    # Use NumPy for fast due-check across the deck
    dues = np.array([c.due for c in sys_cards])
    due_mask = dues <= current_date
    due_indices = np.where(due_mask)[0].tolist()

    # Sort indices to keep deterministic if needed
    due_indices = sorted(due_indices)
    random.shuffle(due_indices)

    # 1. Due Cards Processing

    # Select candidates based on review limit
    candidate_indices = due_indices
    if config.review_limit is not None:
        candidate_indices = due_indices[: config.review_limit]

    if candidate_indices:
        batch_sys = [sys_cards[i] for i in candidate_indices]
        batch_true = [true_cards[i] for i in candidate_indices]

        reviews_done, time_accumulated, updated_sys, updated_true = (
            _batch_process_reviews(
                batch_sys,
                batch_true,
                nature_scheduler,
                algo_scheduler,
                current_date,
                rating_estimator,
                time_estimator,
                card_logs,
                start_time_accumulated=0.0,
                time_limit=config.time_limit,
            )
        )

        # Update the cards in the main lists
        for k, idx in enumerate(candidate_indices[:reviews_done]):
            sys_cards[idx] = updated_sys[k]
            true_cards[idx] = updated_true[k]
    else:
        reviews_done = 0
        time_accumulated = 0.0

    # 2. New Cards Processing
    # Process new cards in chunks to respect time limits and avoid over-allocation
    new_done = 0
    new_batch_size = 50

    while True:
        # Check global stopping conditions
        if config.time_limit is not None and time_accumulated >= config.time_limit:
            break

        remaining_total_reviews = (
            (config.review_limit - reviews_done)
            if config.review_limit is not None
            else 999999
        )
        remaining_new_cards = (
            (config.new_limit - new_done) if config.new_limit is not None else 999999
        )

        # We stop if we hit review limit or new card limit
        if remaining_total_reviews <= 0 or remaining_new_cards <= 0:
            break

        # Determine batch size for this iteration
        batch_size = min(new_batch_size, remaining_total_reviews, remaining_new_cards)

        # Generate IDs and Cards
        new_ids = list(range(next_card_id, next_card_id + batch_size))
        next_card_id += batch_size

        new_batch_sys = [Card(card_id=uid) for uid in new_ids]
        new_batch_true = [Card(card_id=uid) for uid in new_ids]

        # Process this batch
        batch_completed, time_accumulated, updated_sys, updated_true = (
            _batch_process_reviews(
                new_batch_sys,
                new_batch_true,
                nature_scheduler,
                algo_scheduler,
                current_date,
                rating_estimator,
                time_estimator,
                card_logs,
                start_time_accumulated=time_accumulated,
                time_limit=config.time_limit,
            )
        )

        # Update counters
        reviews_done += batch_completed
        new_done += batch_completed

        # Add processed cards to deck
        true_cards.extend(updated_true)
        sys_cards.extend(updated_sys)

        # If batch is incomplete, we hit the time limit
        if batch_completed < batch_size:
            break

    return next_card_id


def _load_initial_state(
    nature_scheduler: Scheduler,
    algo_scheduler: Scheduler,
    seeded_data: SeededData | None,
    seed_history: str | None,
    deck_config: str | None,
    deck_name: str | None,
) -> tuple[list[Card], list[Card], dict[int, list[ReviewLog]], datetime]:
    true_cards: list[Card] = []
    sys_cards: list[Card] = []
    card_logs: dict[int, list[ReviewLog]] = defaultdict(list)

    if seeded_data:
        current_date = seeded_data.last_rev + timedelta(days=1)
        true_cards = list(deepcopy(seeded_data.true_cards).values())
        sys_cards = list(deepcopy(seeded_data.sys_cards).values())
        card_logs = deepcopy(seeded_data.logs)
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


def _initialize_estimators(
    config: SimulationConfig,
    card_logs: dict[int, list[ReviewLog]],
) -> tuple[RatingEstimator, TimeEstimator]:
    # Rating Estimator
    rating_estimator = config.rating_estimator
    if rating_estimator is None:
        weights = infer_review_weights(card_logs)

        def default_rating_est(
            true_cards: Sequence[Card], date: datetime, nature_scheduler: Scheduler
        ) -> Sequence[int]:
            results = []
            for true_card in true_cards:
                if true_card.stability is not None and true_card.stability > 0:
                    retrievability = nature_scheduler.get_card_retrievability(
                        true_card, date
                    )
                    if random.random() < retrievability:
                        results.append(
                            int(
                                random.choices(
                                    [Rating.Hard, Rating.Good, Rating.Easy],
                                    weights=weights.success,
                                )[0]
                            )
                        )
                    else:
                        results.append(int(Rating.Again))
                else:
                    results.append(
                        int(
                            random.choices(
                                [Rating.Again, Rating.Hard, Rating.Good, Rating.Easy],
                                weights=weights.first,
                            )[0]
                        )
                    )
            return results

        rating_estimator = default_rating_est

    # Time Estimator
    time_estimator = config.time_estimator
    if time_estimator is None:
        flat_logs = [log for logs_list in card_logs.values() for log in logs_list]
        duration_data = defaultdict(list)
        for log in flat_logs:
            if log.review_duration is not None:
                duration_data[int(log.rating)].append(log.review_duration)

        avg_durations = {}
        for r in [Rating.Again, Rating.Hard, Rating.Good, Rating.Easy]:
            durs = duration_data[int(r)]
            avg_durations[int(r)] = float(np.mean(durs)) if durs else 0.0

        def default_time_est(
            true_cards: Sequence[Card],
            date: datetime,
            ratings: Sequence[int],
            nature_scheduler: Scheduler,
        ) -> Sequence[float]:
            return [avg_durations.get(r, 0.0) for r in ratings]

        time_estimator = default_time_est

    return rating_estimator, time_estimator


def _calculate_final_metrics(
    true_cards: list[Card],
    sys_cards: list[Card],
    card_logs: dict[int, list[ReviewLog]],
    nature_scheduler: Scheduler,
    current_date: datetime,
    verbose: bool,
) -> tuple[list[float] | None, dict[str, Any]]:
    all_logs_final = [log for logs in card_logs.values() for log in logs]
    fitted_params = None
    try:
        optimizer = RustOptimizer(all_logs_final)
        fitted_params = optimizer.compute_optimal_parameters(verbose=verbose)
    except Exception:
        pass

    stabilities = []
    total_retention = 0.0
    if fitted_params:
        final_algo_scheduler = Scheduler(parameters=tuple(fitted_params))
        for i in range(len(true_cards)):
            card_id = true_cards[i].card_id
            s_nat = true_cards[i].stability or 0.0
            r_nat = nature_scheduler.get_card_retrievability(
                true_cards[i], current_date
            )
            total_retention += r_nat
            rescheduled = final_algo_scheduler.reschedule_card(
                Card(card_id=card_id), card_logs[card_id]
            )
            s_alg = rescheduled.stability or 0.0
            stabilities.append((s_nat, s_alg))

    metrics = {
        "review_count": len(all_logs_final),
        "card_count": len(true_cards),
        "stabilities": stabilities,
        "total_retention": total_retention,
        "logs": all_logs_final,
    }
    return fitted_params, metrics


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
    initial_retention = get_retention_for_day(0, parsed_schedule)

    random.seed(config.seed)
    ground_truth_params = (
        ground_truth if ground_truth is not None else DEFAULT_PARAMETERS
    )

    import fsrs.optimizer

    original_optimizer_tqdm = fsrs.optimizer.tqdm

    def patched_tqdm(*args: Any, **kwargs: Any) -> Any:
        if "position" not in kwargs:
            kwargs["position"] = tqdm_pos + 1
        if "leave" not in kwargs:
            kwargs["leave"] = False
        return original_optimizer_tqdm(*args, **kwargs)

    fsrs.optimizer.tqdm = patched_tqdm

    try:
        nature_scheduler = Scheduler(
            parameters=ground_truth_params, desired_retention=initial_retention
        )
        algo_scheduler = Scheduler(
            parameters=initial_params or DEFAULT_PARAMETERS,
            desired_retention=initial_retention,
        )

        true_cards, sys_cards, card_logs, current_date = _load_initial_state(
            nature_scheduler,
            algo_scheduler,
            seeded_data,
            seed_history,
            deck_config,
            deck_name,
        )

        rating_estimator, time_estimator = _initialize_estimators(config, card_logs)

        # Initialize next_card_id to avoid sleep in Card() constructor
        if sys_cards:
            max_id = max(c.card_id for c in sys_cards)
            next_card_id = max_id + 1
        else:
            next_card_id = int(datetime.now().timestamp() * 1000)

        # Simulation Phases
        if 0 < config.burn_in_days < config.n_days:
            phases = [
                ("P1", config.burn_in_days, 0),
                ("P2", config.n_days, config.burn_in_days),
            ]
        else:
            phases = [("P1", config.n_days, 0)]

        for phase_name, limit, start_day in phases:
            pbar = tqdm(
                range(start_day, limit),
                desc=f"Simulating ({phase_name})",
                position=tqdm_pos,
                leave=False,
                disable=not config.verbose,
            )

            for day in pbar:
                daily_retention = get_retention_for_day(day, parsed_schedule)
                algo_scheduler.desired_retention = daily_retention
                next_card_id = simulate_day(
                    nature_scheduler,
                    algo_scheduler,
                    true_cards,
                    sys_cards,
                    card_logs,
                    current_date,
                    config,
                    rating_estimator,
                    time_estimator,
                    next_card_id,
                )
                current_date += timedelta(days=1)

            # Re-fit algo scheduler after burn-in
            if phase_name == "P1" and 0 < config.burn_in_days < config.n_days:
                all_logs = [log for logs in card_logs.values() for log in logs]
                if len(all_logs) >= 512:
                    optimizer = RustOptimizer(all_logs)
                    fitted_bi = optimizer.compute_optimal_parameters(
                        verbose=config.verbose
                    )
                    algo_scheduler = Scheduler(
                        parameters=tuple(fitted_bi),
                        desired_retention=algo_scheduler.desired_retention,
                    )
                    for i, card in enumerate(sys_cards):
                        sys_cards[i] = algo_scheduler.reschedule_card(
                            Card(card_id=card.card_id), card_logs[card.card_id]
                        )

        fitted_params, metrics = _calculate_final_metrics(
            true_cards,
            sys_cards,
            card_logs,
            nature_scheduler,
            current_date,
            config.verbose,
        )

        return fitted_params, ground_truth_params, metrics
    finally:
        fsrs.optimizer.tqdm = original_optimizer_tqdm
