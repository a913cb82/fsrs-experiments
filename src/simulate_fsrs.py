import random
from collections import defaultdict
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

# Use the Rust-powered optimizer if available
try:
    import fsrs_rs_python

    HAS_RUST_OPTIMIZER = True
except ImportError:
    HAS_RUST_OPTIMIZER = False


class RustOptimizer:
    """A wrapper for fsrs-rs-python to match our existing Optimizer interface."""

    def __init__(
        self,
        review_logs: list[ReviewLog],
        pre_constructed_items: list[Any] | None = None,
    ):
        self.review_logs = review_logs
        self.pre_constructed_items = pre_constructed_items or []

    def compute_optimal_parameters(self, verbose: bool = False) -> list[float]:
        if not HAS_RUST_OPTIMIZER:
            raise ImportError("fsrs-rs-python not installed.")

        items_map: dict[int, list[ReviewLog]] = defaultdict(list)
        for log in self.review_logs:
            items_map[log.card_id].append(log)

        rust_items = list(self.pre_constructed_items)
        for card_id in items_map:
            # Sort logs for this card by date
            logs = sorted(items_map[card_id], key=lambda x: x.review_datetime)

            # We need at least 2 reviews to create a training sample
            if len(logs) < 2:
                continue

            # Convert logs to FSRSReview objects
            all_reviews = []
            last_date = None
            for log in logs:
                if last_date is None:
                    delta_t = 0
                else:
                    delta_t = (log.review_datetime - last_date).days

                all_reviews.append(
                    fsrs_rs_python.FSRSReview(int(log.rating), int(delta_t))
                )
                last_date = log.review_datetime

            # Create one FSRSItem for every review from the 2nd one onwards
            for i in range(2, len(all_reviews) + 1):
                item_history = all_reviews[:i]
                rust_items.append(fsrs_rs_python.FSRSItem(item_history))

        # Filter items: Rust optimizer requires at least one review with delta_t > 0
        filtered_items = [
            item for item in rust_items if item.long_term_review_cnt() > 0
        ]

        if not filtered_items:
            return list(fsrs_rs_python.DEFAULT_PARAMETERS)

        # Run the Rust optimizer
        fsrs_core = fsrs_rs_python.FSRS(fsrs_rs_python.DEFAULT_PARAMETERS)
        optimized_w = fsrs_core.compute_parameters(filtered_items)
        return list(optimized_w)


def _process_review(
    card: Card,
    true_card: Card,
    nature_scheduler: Scheduler,
    algo_scheduler: Scheduler,
    current_date: datetime,
    rating_estimator: RatingEstimator,
    time_estimator: TimeEstimator,
) -> tuple[Card, Card, ReviewLog, float]:
    rating_val = rating_estimator(
        true_card,
        current_date,
        nature_scheduler,
    )
    rating = Rating(rating_val)

    review_time = time_estimator(true_card, current_date, int(rating), nature_scheduler)

    updated_true_card, _ = nature_scheduler.review_card(true_card, rating, current_date)
    updated_card, log = algo_scheduler.review_card(card, rating, current_date)

    return updated_card, updated_true_card, log, review_time


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
) -> None:
    # Use NumPy for fast due-check across the deck
    dues = np.array([c.due for c in sys_cards])
    due_mask = dues <= current_date
    due_indices = np.where(due_mask)[0].tolist()

    # Sort indices to keep deterministic if needed (NumPy where is stable)
    due_indices = sorted(due_indices)
    random.shuffle(due_indices)
    reviews_done = 0
    time_accumulated = 0.0

    # Due cards
    for idx in due_indices:
        if config.time_limit is not None and time_accumulated >= config.time_limit:
            break
        if config.review_limit is not None and reviews_done >= config.review_limit:
            break

        sys_card, true_card, log, review_time = _process_review(
            sys_cards[idx],
            true_cards[idx],
            nature_scheduler,
            algo_scheduler,
            current_date,
            rating_estimator,
            time_estimator,
        )

        if (
            config.time_limit is not None
            and time_accumulated + review_time > config.time_limit
        ):
            break

        time_accumulated += review_time
        sys_cards[idx] = sys_card
        true_cards[idx] = true_card
        card_logs[sys_card.card_id].append(log)
        reviews_done += 1

    # New cards
    new_done = 0
    while True:
        if config.time_limit is not None and time_accumulated >= config.time_limit:
            break
        if config.review_limit is not None and reviews_done >= config.review_limit:
            break
        if config.new_limit is not None and new_done >= config.new_limit:
            break

        # For new cards, we determine rating first to see if it's Again/Success
        rating_val = rating_estimator(
            Card(),
            current_date,
            nature_scheduler,
        )
        rating = Rating(rating_val)

        review_time = time_estimator(
            Card(), current_date, int(rating), nature_scheduler
        )

        if (
            config.time_limit is not None
            and time_accumulated + review_time > config.time_limit
        ):
            break

        time_accumulated += review_time
        updated_true_card, _ = nature_scheduler.review_card(
            Card(), rating, current_date
        )
        updated_sys_card, log = algo_scheduler.review_card(Card(), rating, current_date)

        true_cards.append(updated_true_card)
        sys_cards.append(updated_sys_card)
        card_logs[updated_sys_card.card_id].append(log)
        reviews_done += 1
        new_done += 1


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

        true_cards: list[Card] = []
        sys_cards: list[Card] = []
        card_logs: dict[int, list[ReviewLog]] = defaultdict(list)
        s_data = seeded_data

        if s_data:
            current_date = s_data.last_rev + timedelta(days=1)
            true_cards = list(deepcopy(s_data.true_cards).values())
            sys_cards = list(deepcopy(s_data.sys_cards).values())
            card_logs = deepcopy(s_data.logs)
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

        # Initialize estimators
        rating_estimator = config.rating_estimator
        if rating_estimator is None:
            weights = infer_review_weights(card_logs)

            def default_rating_est(
                true_card: Card, date: datetime, nature_scheduler: Scheduler
            ) -> int:
                if true_card.stability is not None and true_card.stability > 0:
                    retrievability = nature_scheduler.get_card_retrievability(
                        true_card, date
                    )
                    if random.random() < retrievability:
                        return int(
                            random.choices(
                                [Rating.Hard, Rating.Good, Rating.Easy],
                                weights=weights.success,
                            )[0]
                        )
                    else:
                        return int(Rating.Again)
                else:
                    return int(
                        random.choices(
                            [Rating.Again, Rating.Hard, Rating.Good, Rating.Easy],
                            weights=weights.first,
                        )[0]
                    )

            rating_estimator = default_rating_est

        time_estimator = config.time_estimator
        if time_estimator is None:
            # Calculate average time per grade from history
            flat_logs = [log for logs_list in card_logs.values() for log in logs_list]
            duration_data = defaultdict(list)
            for log in flat_logs:
                if log.review_duration is not None:
                    duration_data[int(log.rating)].append(log.review_duration)

            # Compute averages, fallback to 0.0
            avg_durations = {}
            for r in [Rating.Again, Rating.Hard, Rating.Good, Rating.Easy]:
                durs = duration_data[int(r)]
                avg_durations[int(r)] = float(np.mean(durs)) if durs else 0.0

            def default_time_est(
                true_card: Card,
                date: datetime,
                rating: int,
                nature_scheduler: Scheduler,
            ) -> float:
                return avg_durations.get(rating, 0.0)

            time_estimator = default_time_est

        limit_phase_1 = config.n_days
        if 0 < config.burn_in_days < config.n_days:
            limit_phase_1 = config.burn_in_days
        pbar_p1 = tqdm(
            range(limit_phase_1),
            desc="Simulating (P1)",
            position=tqdm_pos,
            leave=False,
            disable=not config.verbose,
        )

        for day in pbar_p1:
            daily_retention = get_retention_for_day(day, parsed_schedule)
            algo_scheduler.desired_retention = daily_retention
            simulate_day(
                nature_scheduler,
                algo_scheduler,
                true_cards,
                sys_cards,
                card_logs,
                current_date,
                config,
                rating_estimator,
                time_estimator,
            )
            current_date += timedelta(days=1)

        if 0 < config.burn_in_days < config.n_days:
            all_logs = [log for logs in card_logs.values() for log in logs]
            if len(all_logs) >= 512:
                optimizer = RustOptimizer(all_logs)
                fitted_bi = optimizer.compute_optimal_parameters(verbose=config.verbose)
                algo_scheduler = Scheduler(
                    parameters=tuple(fitted_bi),
                    desired_retention=algo_scheduler.desired_retention,
                )
                for i, card in enumerate(sys_cards):
                    sys_cards[i] = algo_scheduler.reschedule_card(
                        Card(card_id=card.card_id), card_logs[card.card_id]
                    )

            pbar_p2 = tqdm(
                range(config.burn_in_days, config.n_days),
                desc="Simulating (P2)",
                position=tqdm_pos,
                leave=False,
                disable=not config.verbose,
            )
            for day in pbar_p2:
                daily_retention = get_retention_for_day(day, parsed_schedule)
                algo_scheduler.desired_retention = daily_retention
                simulate_day(
                    nature_scheduler,
                    algo_scheduler,
                    true_cards,
                    sys_cards,
                    card_logs,
                    current_date,
                    config,
                    rating_estimator,
                    time_estimator,
                )
                current_date += timedelta(days=1)

        all_logs_final = [log for logs in card_logs.values() for log in logs]
        fitted_params = None
        try:
            optimizer = RustOptimizer(all_logs_final)
            fitted_params = optimizer.compute_optimal_parameters(verbose=config.verbose)
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

        return (
            fitted_params,
            ground_truth_params,
            {
                "review_count": len(all_logs_final),
                "card_count": len(true_cards),
                "stabilities": stabilities,
                "total_retention": total_retention,
                "logs": all_logs_final,
            },
        )
    finally:
        fsrs.optimizer.tqdm = original_optimizer_tqdm
