import argparse
import math
import random
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any

from tqdm import tqdm

# Ensure we can import fsrs if it's not in path for some reason,
try:
    from fsrs import Card, Optimizer, Rating, ReviewLog, Scheduler
    from fsrs.scheduler import DEFAULT_PARAMETERS
except ImportError:
    # Fallback to local import if installed in venv but not picked up?
    # Should be fine as we verified install.
    sys.exit("Error: fsrs package not found. Please install it.")

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
        # Create a new card
        # We need to ensure both schedulers see the same new card
        # Card() uses time.sleep(0.001) to ensure unique IDs based on time
        base_card = Card()

        # Deepcopy to ensure independent state evolution
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
    """
    Parses a schedule string like "5:0.7,1:0.9" or a constant float like "0.9"
    into a list of (days, retention).
    """
    try:
        if ":" not in schedule_str:
            # Constant retention
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
        return [(1, 0.9)]  # Default fallback


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
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass

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
        ground_truth_params = DEFAULT_PARAMETERS

        # Nature Scheduler: Always Ground Truth
        nature_scheduler = Scheduler(
            parameters=ground_truth_params, desired_retention=initial_retention
        )

        # Algo Scheduler: Initially Ground Truth, changes after burn-in
        algo_scheduler = Scheduler(
            parameters=ground_truth_params, desired_retention=initial_retention
        )

        # Two maps to track divergent states
        true_cards: dict[int, Card] = {}  # ID -> Card (Nature State)
        sys_cards: dict[int, Card] = {}  # ID -> Card (Algo State)

        card_logs: dict[int, list[ReviewLog]] = defaultdict(list)

        current_date = START_DATE

        # PHASE 1: Burn-in (or full run if burn_in_days=0)
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
                    optimizer = Optimizer(all_logs)
                    fitted_params = optimizer.compute_optimal_parameters(
                        verbose=verbose
                    )

                    # Update Algo Scheduler with fitted params
                    algo_scheduler = Scheduler(
                        parameters=fitted_params,
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
        fitted_params = None
        try:
            optimizer = Optimizer(all_logs)
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
