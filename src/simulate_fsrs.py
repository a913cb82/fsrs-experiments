import argparse
import math
import random
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any

from tqdm import tqdm

# Ensure we can import fsrs if it's not in path for some reason
try:
    from fsrs import Card, Rating, ReviewLog, Scheduler
    from fsrs.scheduler import DEFAULT_PARAMETERS
except ImportError:
    sys.exit("Error: fsrs package not found. Please install it.")

# Use the Rust-powered optimizer if available
try:
    import fsrs_rs_python

    HAS_RUST_OPTIMIZER = True
except ImportError:
    HAS_RUST_OPTIMIZER = False

# Constants for simulation
START_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)

# Probabilities given recall (Success)
PROB_HARD = 0.15
PROB_GOOD = 0.75
PROB_EASY = 0.10

# Weights for new cards (first review ratings)
PROB_FIRST_AGAIN = 0.2
PROB_FIRST_HARD = 0.2
PROB_FIRST_GOOD = 0.5
PROB_FIRST_EASY = 0.1


class RustOptimizer:
    """A wrapper for fsrs-rs-python to match our existing Optimizer interface."""

    def __init__(self, review_logs: list[ReviewLog]):
        self.review_logs = review_logs

    def compute_optimal_parameters(self, verbose: bool = False) -> list[float]:
        if not HAS_RUST_OPTIMIZER:
            raise ImportError("fsrs-rs-python not installed.")

        # Convert ReviewLog objects to Rust-compatible FSRSItem objects.
        # CRITICAL: FSRS-rs expects one FSRSItem per review (after the first).
        # Each item contains the history UP TO that review.
        items_map = defaultdict(list)
        for log in self.review_logs:
            items_map[log.card_id].append(log)

        rust_items = []
        for card_id in items_map:
            # Sort logs for this card by date
            logs = sorted(items_map[card_id], key=lambda x: x.review_datetime)

            # We need at least 2 reviews to create a training sample (1st is initial)
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
            # Each item contains the history of reviews up to that point.
            for i in range(2, len(all_reviews) + 1):
                item_history = all_reviews[:i]
                rust_items.append(fsrs_rs_python.FSRSItem(item_history))

        if not rust_items:
            return list(fsrs_rs_python.DEFAULT_PARAMETERS)

        # Run the Rust optimizer
        fsrs_core = fsrs_rs_python.FSRS(fsrs_rs_python.DEFAULT_PARAMETERS)
        optimized_w = fsrs_core.compute_parameters(rust_items)
        return list(optimized_w)


def simulate_day(
    nature_scheduler: Scheduler,
    algo_scheduler: Scheduler,
    true_cards: dict[int, Card],
    sys_cards: dict[int, Card],
    card_logs: dict[int, list[ReviewLog]],
    current_date: datetime,
    review_limit: int,
) -> None:
    # Identify due cards based on SYSTEM scheduler
    due_cards = []
    for card in sys_cards.values():
        if card.due <= current_date:
            due_cards.append(card)

    # Sort due cards or shuffle
    random.shuffle(due_cards)

    reviews_done = 0

    # Review due cards
    for sys_card in due_cards:
        if reviews_done >= review_limit:
            break

        card_id = sys_card.card_id
        true_card = true_cards[card_id]

        # Calculate retrievability using NATURE (Ground Truth)
        retrievability = nature_scheduler.get_card_retrievability(
            true_card, current_date
        )

        # Determine outcome
        if random.random() < retrievability:
            # Success
            rating = random.choices(
                [Rating.Hard, Rating.Good, Rating.Easy],
                weights=[PROB_HARD, PROB_GOOD, PROB_EASY],
            )[0]
        else:
            # Fail
            rating = Rating.Again

        # Update NATURE state (updates P(Recall) for future)
        updated_true_card, _ = nature_scheduler.review_card(
            true_card, rating, current_date
        )
        true_cards[card_id] = updated_true_card

        # Update SYSTEM state (calculates interval using Algo Params)
        updated_sys_card, log = algo_scheduler.review_card(
            sys_card, rating, current_date
        )
        sys_cards[card_id] = updated_sys_card

        card_logs[card_id].append(log)
        reviews_done += 1

    # Review new cards
    while reviews_done < review_limit:
        base_card = Card()
        true_card = deepcopy(base_card)
        sys_card = deepcopy(base_card)

        # Determine first rating
        rating = random.choices(
            [Rating.Again, Rating.Hard, Rating.Good, Rating.Easy],
            weights=[
                PROB_FIRST_AGAIN,
                PROB_FIRST_HARD,
                PROB_FIRST_GOOD,
                PROB_FIRST_EASY,
            ],
        )[0]

        # Update both
        updated_true_card, _ = nature_scheduler.review_card(
            true_card, rating, current_date
        )
        updated_sys_card, log = algo_scheduler.review_card(
            sys_card, rating, current_date
        )

        # Add to maps
        true_cards[updated_true_card.card_id] = updated_true_card
        sys_cards[updated_sys_card.card_id] = updated_sys_card
        card_logs[updated_sys_card.card_id].append(log)

        reviews_done += 1


def parse_retention_schedule(
    schedule_str: str,
) -> list[tuple[int, float]]:
    try:
        if ":" not in schedule_str:
            return [(1, float(schedule_str))]

        segments = []
        parts = schedule_str.split(",")
        for part in parts:
            d, r = part.split(":")
            segments.append((int(d), float(r)))
        return segments
    except ValueError:
        tqdm.write(
            f"Invalid retention format: {schedule_str}. "
            "Expected float (e.g. '0.9') or schedule (e.g. '5:0.7,1:0.9')"
        )
        return [(1, 0.9)]


def get_retention_for_day(
    day: int, schedule_segments: list[tuple[int, float]]
) -> float:
    total_duration = sum(d for d, r in schedule_segments)
    day_in_cycle = day % total_duration

    current_pos = 0
    for duration, retention in schedule_segments:
        if current_pos <= day_in_cycle < current_pos + duration:
            return retention
        current_pos += duration
    return schedule_segments[-1][1]


def run_simulation(
    n_days: int = 365,
    burn_in_days: int = 0,
    review_limit: int = 200,
    retention: str = "0.9",
    verbose: bool = True,
    seed: int = 42,
    tqdm_pos: int = 0,
) -> tuple[list[float] | None, tuple[float, ...], dict[str, Any]]:
    parsed_schedule = parse_retention_schedule(retention)
    initial_retention = get_retention_for_day(0, parsed_schedule)

    if verbose and tqdm_pos == 0:
        burn_in_info = f", Burn-in: {burn_in_days} days" if burn_in_days > 0 else ""
        tqdm.write(
            f"Starting simulation: {n_days} days{burn_in_info}, "
            f"{review_limit} reviews/day, Retention: {retention}, Seed: {seed}"
        )

    # Set seed for reproducibility
    random.seed(seed)

    ground_truth_params = DEFAULT_PARAMETERS

    nature_scheduler = Scheduler(
        parameters=ground_truth_params, desired_retention=initial_retention
    )

    algo_scheduler = Scheduler(
        parameters=ground_truth_params, desired_retention=initial_retention
    )

    true_cards: dict[int, Card] = {}
    sys_cards: dict[int, Card] = {}
    card_logs: dict[int, list[ReviewLog]] = defaultdict(list)

    current_date = START_DATE
    limit_phase_1 = burn_in_days if burn_in_days > 0 else n_days

    pbar_p1 = (
        tqdm(
            range(limit_phase_1),
            desc="Simulating (P1)",
            position=tqdm_pos,
            leave=False,
            disable=not verbose,
        )
        if verbose
        else range(limit_phase_1)
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
            review_limit,
        )
        current_date += timedelta(days=1)

    if burn_in_days > 0 and burn_in_days < n_days:
        all_logs = [log for logs in card_logs.values() for log in logs]

        if len(all_logs) >= 512:
            try:
                # Use Rust optimizer
                optimizer = RustOptimizer(all_logs)
                fitted_bi = optimizer.compute_optimal_parameters(verbose=verbose)

                algo_scheduler = Scheduler(
                    parameters=tuple(fitted_bi),
                    desired_retention=algo_scheduler.desired_retention,
                )

                for cid in sys_cards:
                    sys_cards[cid] = algo_scheduler.reschedule_card(
                        sys_cards[cid], card_logs[cid]
                    )

            except Exception as e:
                if verbose:
                    tqdm.write(f"Burn-in optimization failed: {e}")

        pbar_p2 = (
            tqdm(
                range(burn_in_days, n_days),
                desc="Simulating (P2)",
                position=tqdm_pos,
                leave=False,
                disable=not verbose,
            )
            if verbose
            else range(burn_in_days, n_days)
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
                review_limit,
            )
            current_date += timedelta(days=1)

    all_logs = [log for logs in card_logs.values() for log in logs]

    total_retrievability = 0.0
    end_date = current_date
    log_prob_sum = 0.0

    for card in true_cards.values():
        r = nature_scheduler.get_card_retrievability(card, end_date)
        total_retrievability += r
        if r > 0:
            log_prob_sum += math.log(r)
        else:
            log_prob_sum = -float("inf")

    fitted_params: list[float] | None = None
    try:
        optimizer = RustOptimizer(all_logs)
        fitted_params = optimizer.compute_optimal_parameters(verbose=verbose)

    except Exception as e:
        if verbose:
            tqdm.write(f"Error during final optimization: {e}")

    return (
        fitted_params,
        ground_truth_params,
        {
            "total_retention": total_retrievability,
            "log_prob_sum": log_prob_sum,
            "review_count": len(all_logs),
            "card_count": len(true_cards),
        },
    )


def run_simulation_cli() -> None:
    parser = argparse.ArgumentParser(description="Run FSRS Simulation")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--reviews", type=int, default=200)
    parser.add_argument("--retention", type=str, default="0.9")
    parser.add_argument("--burn-in", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_simulation(
        n_days=args.days,
        review_limit=args.reviews,
        retention=args.retention,
        burn_in_days=args.burn_in,
        seed=args.seed,
    )


if __name__ == "__main__":
    run_simulation_cli()
