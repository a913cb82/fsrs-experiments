import argparse
import math
import os
import random
import sqlite3
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
        items_map = defaultdict(list)
        for log in self.review_logs:
            items_map[log.card_id].append(log)

        rust_items = []
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

        if not rust_items:
            return list(fsrs_rs_python.DEFAULT_PARAMETERS)

        # Run the Rust optimizer
        fsrs_core = fsrs_rs_python.FSRS(fsrs_rs_python.DEFAULT_PARAMETERS)
        optimized_w = fsrs_core.compute_parameters(rust_items)
        return list(optimized_w)


def load_anki_history(
    path: str,
) -> tuple[dict[int, list[ReviewLog]], datetime]:
    """
    Extracts FSRS-compatible review logs from an Anki collection.anki2 file.
    """
    if not os.path.exists(path):
        tqdm.write(f"Error: Anki database not found at {path}")
        return {}, START_DATE

    conn = sqlite3.connect(path)
    cur = conn.cursor()

    try:
        cur.execute(
            "SELECT cid, ease, id, type FROM revlog "
            "WHERE ease BETWEEN 1 AND 4 ORDER BY id ASC"
        )
        rows = cur.fetchall()
    except sqlite3.OperationalError as e:
        tqdm.write(f"Error reading Anki database: {e}")
        return {}, START_DATE
    finally:
        conn.close()

    card_logs: dict[int, list[ReviewLog]] = defaultdict(list)
    last_review_time = START_DATE

    for cid, ease, rev_id, _rev_type in rows:
        dt = datetime.fromtimestamp(rev_id / 1000.0, tz=timezone.utc)
        log = ReviewLog(
            card_id=cid,
            rating=Rating(ease),
            review_datetime=dt,
            review_duration=None,
        )
        card_logs[cid].append(log)
        if dt > last_review_time:
            last_review_time = dt

    total_revs = sum(len(logs) for logs in card_logs.values())
    tqdm.write(f"Loaded {total_revs} reviews for {len(card_logs)} cards from Anki.")
    return card_logs, last_review_time


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

        # Update NATURE state
        updated_true_card, _ = nature_scheduler.review_card(
            true_card, rating, current_date
        )
        true_cards[card_id] = updated_true_card

        # Update SYSTEM state
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


def parse_parameters(params_str: str) -> tuple[float, ...]:
    """Parses a comma-separated string of parameters into a tuple of floats."""
    try:
        parts = [float(p.strip()) for p in params_str.split(",")]
        if len(parts) != 21:
            tqdm.write(
                f"Warning: Expected 21 parameters, got {len(parts)}. Using defaults."
            )
            return tuple(DEFAULT_PARAMETERS)
        return tuple(parts)
    except ValueError:
        tqdm.write(f"Invalid parameter format: {params_str}. Using defaults.")
        return tuple(DEFAULT_PARAMETERS)


def run_simulation(
    n_days: int = 365,
    burn_in_days: int = 0,
    review_limit: int = 200,
    retention: str = "0.9",
    verbose: bool = True,
    seed: int = 42,
    tqdm_pos: int = 0,
    ground_truth: tuple[float, ...] | None = None,
    seed_history: str | None = None,
) -> tuple[list[float] | None, tuple[float, ...], dict[str, Any]]:
    parsed_schedule = parse_retention_schedule(retention)
    initial_retention = get_retention_for_day(0, parsed_schedule)

    if verbose and tqdm_pos == 0:
        burn_in_info = f", Burn-in: {burn_in_days} days" if burn_in_days > 0 else ""
        seed_info = f", Seeded from: {seed_history}" if seed_history else ""
        tqdm.write(
            f"Starting simulation: {n_days} days{burn_in_info}{seed_info}, "
            f"{review_limit} reviews/day, Retention: {retention}, Seed: {seed}"
        )

    # Set seed for reproducibility
    random.seed(seed)

    ground_truth_params = (
        ground_truth if ground_truth is not None else DEFAULT_PARAMETERS
    )

    # Monkeypatch Optimizer's tqdm to respect position
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
        # Nature Scheduler: Always Ground Truth
        nature_scheduler = Scheduler(
            parameters=ground_truth_params, desired_retention=initial_retention
        )

        # Algo Scheduler: Initially Ground Truth (or defaults)
        algo_scheduler = Scheduler(
            parameters=DEFAULT_PARAMETERS, desired_retention=initial_retention
        )

        # Two maps to track divergent states
        true_cards: dict[int, Card] = {}  # ID -> Card (Nature State)
        sys_cards: dict[int, Card] = {}  # ID -> Card (Algo State)

        card_logs: dict[int, list[ReviewLog]] = defaultdict(list)

        current_date = START_DATE

        # Seeding Phase
        if seed_history:
            seeded_logs, last_rev = load_anki_history(seed_history)
            current_date = last_rev + timedelta(days=1)

            if seeded_logs:
                if verbose:
                    tqdm.write("Initializing card states from seeded history...")
                for cid, logs in seeded_logs.items():
                    # Reconstruct NATURE state using Ground Truth params
                    true_cards[cid] = nature_scheduler.reschedule_card(
                        Card(card_id=cid), logs
                    )
                    # Reconstruct ALGO state using default params
                    sys_cards[cid] = algo_scheduler.reschedule_card(
                        Card(card_id=cid), logs
                    )
                    card_logs[cid].extend(logs)

        # PHASE 1: Burn-in
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

        # PHASE 2: After burn-in
        if burn_in_days > 0 and burn_in_days < n_days:
            # Flatten logs for optimizer
            all_logs = [log for logs in card_logs.values() for log in logs]

            if len(all_logs) >= 512:
                try:
                    optimizer = RustOptimizer(all_logs)
                    fitted_bi = optimizer.compute_optimal_parameters(verbose=verbose)

                    # Update Algo Scheduler with fitted params
                    algo_scheduler = Scheduler(
                        parameters=tuple(fitted_bi),
                        desired_retention=algo_scheduler.desired_retention,
                    )

                    # Reschedule existing cards
                    for cid in sys_cards:
                        sys_cards[cid] = algo_scheduler.reschedule_card(
                            sys_cards[cid], card_logs[cid]
                        )

                except Exception as e:
                    if verbose:
                        tqdm.write(f"Burn-in optimization failed: {e}")

            # Resume simulation
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

        # End of simulation
        all_logs = [log for logs in card_logs.values() for log in logs]

        # Calculate memorized over time using TRUE cards (Nature)
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

        # Fit FSRS parameters
        fitted_params: list[float] | None = None
        try:
            optimizer = RustOptimizer(all_logs)
            fitted_params = optimizer.compute_optimal_parameters(verbose=verbose)

        except Exception as e:
            if verbose:
                tqdm.write(f"Error during final optimization: {e}")

        # Reconstruct final stabilities for all cards
        stabilities: list[tuple[float, float]] = []
        if fitted_params:
            final_algo_scheduler = Scheduler(parameters=tuple(fitted_params))
            for cid in true_cards:
                s_nat = true_cards[cid].stability or 0.0
                rescheduled = final_algo_scheduler.reschedule_card(
                    Card(card_id=cid), card_logs[cid]
                )
                s_alg = rescheduled.stability or 0.0
                stabilities.append((s_nat, s_alg))

        return (
            fitted_params,
            ground_truth_params,
            {
                "total_retention": total_retrievability,
                "log_prob_sum": log_prob_sum,
                "review_count": len(all_logs),
                "card_count": len(true_cards),
                "stabilities": stabilities,
            },
        )
    finally:
        # Restore original tqdm
        fsrs.optimizer.tqdm = original_optimizer_tqdm


def run_simulation_cli() -> None:
    parser = argparse.ArgumentParser(description="Run FSRS Simulation")
    parser.add_argument(
        "--days", type=int, default=365, help="Number of days to simulate"
    )
    parser.add_argument("--reviews", type=int, default=200, help="Daily review limit")
    parser.add_argument(
        "--retention", type=str, default="0.9", help="Retention (float or schedule)"
    )
    parser.add_argument("--burn-in", type=int, default=0, help="Burn-in days")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--ground-truth", type=str, help="Comma-separated FSRS-6 parameters"
    )
    parser.add_argument("--seed-history", type=str, help="Path to collection.anki2")

    args = parser.parse_args()

    gt_params = parse_parameters(args.ground_truth) if args.ground_truth else None

    run_simulation(
        n_days=args.days,
        review_limit=args.reviews,
        retention=args.retention,
        burn_in_days=args.burn_in,
        seed=args.seed,
        ground_truth=gt_params,
        seed_history=args.seed_history,
    )


if __name__ == "__main__":
    run_simulation_cli()
