import argparse
import json
import os
import random
import sqlite3
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
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

# Probabilities given recall (Success) - Defaults
DEFAULT_PROB_HARD = 0.15
DEFAULT_PROB_GOOD = 0.75
DEFAULT_PROB_EASY = 0.10

# Weights for new cards (first review ratings) - Defaults
DEFAULT_PROB_FIRST_AGAIN = 0.2
DEFAULT_PROB_FIRST_HARD = 0.2
DEFAULT_PROB_FIRST_GOOD = 0.5
DEFAULT_PROB_FIRST_EASY = 0.1

__all__ = [
    "run_simulation",
    "run_simulation_cli",
    "load_anki_history",
    "parse_parameters",
    "RustOptimizer",
    "ReviewLog",
    "Rating",
    "Card",
    "Scheduler",
    "DEFAULT_PARAMETERS",
    "infer_review_weights",
]


def infer_review_weights(
    card_logs: dict[int, list[ReviewLog]],
) -> dict[str, list[float]]:
    """
    Infers rating probabilities from real review history.
    Returns weights for first reviews and weights for subsequent successful reviews.
    """
    first_ratings = [0, 0, 0, 0]  # Again, Hard, Good, Easy
    success_ratings = [0, 0, 0]  # Hard, Good, Easy

    for logs in card_logs.values():
        if not logs:
            continue
        # Sort logs by time
        sorted_logs = sorted(logs, key=lambda x: x.review_datetime)

        # First review
        first_r = int(sorted_logs[0].rating)
        if 1 <= first_r <= 4:
            first_ratings[first_r - 1] += 1

        # Subsequent reviews
        for log in sorted_logs[1:]:
            r = int(log.rating)
            if 2 <= r <= 4:  # Success branch (recalled)
                success_ratings[r - 2] += 1

    # Normalize First Ratings
    total_first = sum(first_ratings)
    if total_first > 0:
        first_weights = [r / total_first for r in first_ratings]
    else:
        first_weights = [
            DEFAULT_PROB_FIRST_AGAIN,
            DEFAULT_PROB_FIRST_HARD,
            DEFAULT_PROB_FIRST_GOOD,
            DEFAULT_PROB_FIRST_EASY,
        ]

    # Normalize Success Ratings
    total_success = sum(success_ratings)
    if total_success > 0:
        success_weights = [r / total_success for r in success_ratings]
    else:
        success_weights = [
            DEFAULT_PROB_HARD,
            DEFAULT_PROB_GOOD,
            DEFAULT_PROB_EASY,
        ]

    return {"first": first_weights, "success": success_weights}


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

        # Convert ReviewLog objects to Rust-compatible FSRSItem objects.
        # We only process logs that aren't already covered by pre_constructed_items.
        # For simplicity in this optimized version, if we have
        # pre_constructed_items we assume they represent the static history
        # and review_logs represent the NEW reviews.
        # However, Anki optimization usually requires the FULL history per card.
        # So we'll maintain the snapshots logic but allow passing them in.

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


def decode_varint(data: bytes, pos: int) -> tuple[int, int]:
    """Decodes a Protobuf varint from bytes."""
    res = 0
    shift = 0
    while True:
        b = data[pos]
        res |= (b & 0x7F) << shift
        pos += 1
        if not (b & 0x80):
            break
        shift += 7
    return res, pos


def get_deck_config_id(blob: bytes) -> int | None:
    """Extracts config_id (field 1) from Anki's DeckCommon BLOB."""
    if not blob or len(blob) == 0 or blob[0] != 0x08:
        return None
    try:
        cid, _ = decode_varint(blob, 1)
        return cid
    except Exception:
        return None


def load_anki_history(
    path: str,
    deck_config_name: str | None = None,
    deck_name: str | None = None,
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
        # Check database version/schema
        cur.execute("SELECT ver FROM col")
        v_row = cur.fetchone()
        ver = int(v_row[0]) if v_row and v_row[0] is not None else 0

        valid_deck_ids: list[int] = []

        if ver >= 18:
            # New relational schema (Anki 23.10+)
            cur.execute("SELECT id, name, common FROM decks")
            decks = cur.fetchall()

            cur.execute("SELECT id, name FROM deck_config")
            configs = {str(row[1]): int(row[0]) for row in cur.fetchall()}

            if deck_config_name and deck_config_name not in configs:
                avail = ", ".join(configs.keys())
                tqdm.write(f"Warning: Config '{deck_config_name}' not found.")
                tqdm.write(f"Available configs: {avail}")
                return {}, START_DATE

            t_cid = configs.get(str(deck_config_name)) if deck_config_name else None

            # Map deck names to their rows for inheritance lookup
            d_name_map = {str(row[1]): row for row in decks}

            res_d_configs: dict[int, int] = {}
            for row in decks:
                if row[0] is None or row[1] is None:
                    continue
                d_id, d_name, d_blob_raw = int(row[0]), str(row[1]), row[2]
                d_blob = bytes(d_blob_raw) if d_blob_raw is not None else b""
                c_id = get_deck_config_id(d_blob)
                if c_id is None:
                    parts = d_name.split("::")
                    for i in range(len(parts) - 1, 0, -1):
                        p_name = "::".join(parts[:i])
                        if p_name in d_name_map:
                            p_row = d_name_map[p_name]
                            p_blob_raw = p_row[2]
                            if p_blob_raw is not None:
                                p_blob = bytes(p_blob_raw)
                                c_id = get_deck_config_id(p_blob)
                                if c_id is not None:
                                    break
                if c_id is None:
                    c_id = 1
                res_d_configs[d_id] = c_id

            # Filter decks
            for row in decks:
                if row[0] is None or row[1] is None:
                    continue
                d_id, d_name = int(row[0]), str(row[1])
                if deck_name and d_name != deck_name:
                    continue
                if t_cid is not None:
                    if d_id not in res_d_configs or res_d_configs[d_id] != t_cid:
                        continue
                valid_deck_ids.append(d_id)

            if deck_name and not valid_deck_ids:
                avail_decks = ", ".join(str(d[1]) for d in decks)
                tqdm.write(f"Warning: Deck '{deck_name}' not found.")
                tqdm.write(f"Available decks: {avail_decks}")
        else:
            # Old JSON schema (pre-23.10)
            cur.execute("SELECT decks, dconf FROM col")
            col_row = cur.fetchone()
            if not col_row:
                return {}, START_DATE

            decks_json = json.loads(col_row[0]) if col_row[0] else {}
            dconf_json = json.loads(col_row[1]) if col_row[1] else {}

            target_conf_id = None
            if deck_config_name:
                for cid_s, cfg in dconf_json.items():
                    if cfg.get("name") == deck_config_name:
                        target_conf_id = int(cid_s)
                        break
                if target_conf_id is None:
                    avail_cfg = [
                        str(c.get("name")) for c in dconf_json.values() if c.get("name")
                    ]
                    tqdm.write(
                        f"Warning: Config '{deck_config_name}' not found. "
                        f"Available: {', '.join(avail_cfg)}"
                    )

            for did_s, deck in decks_json.items():
                if deck_name and deck.get("name") != deck_name:
                    continue
                if target_conf_id is None or deck.get("conf") == target_conf_id:
                    valid_deck_ids.append(int(did_s))

        if not valid_deck_ids:
            return {}, START_DATE

        placeholders = ",".join(["?"] * len(valid_deck_ids))
        query = f"""
            SELECT r.cid, r.ease, r.id, r.type
            FROM revlog r
            JOIN cards c ON r.cid = c.id
            WHERE r.ease BETWEEN 1 AND 4
            AND c.did IN ({placeholders})
            ORDER BY r.id ASC
        """
        cur.execute(query, valid_deck_ids)
        rows = cur.fetchall()
    except (sqlite3.OperationalError, json.JSONDecodeError) as e:
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
    tqdm.write(f"Successfully loaded {total_revs} reviews for {len(card_logs)} cards.")
    return card_logs, last_review_time


def simulate_day(
    nature_scheduler: Scheduler,
    algo_scheduler: Scheduler,
    true_cards: list[Card],
    sys_cards: list[Card],
    card_logs: dict[int, list[ReviewLog]],
    current_date: datetime,
    review_limit: int,
    weights: dict[str, list[float]] | None = None,
) -> None:
    # Use NumPy for fast due-check across the deck
    # This replaces the O(N) Python loop scanner
    dues = np.array([c.due for c in sys_cards])
    due_mask = dues <= current_date
    due_indices = np.where(due_mask)[0].tolist()

    random.shuffle(due_indices)
    reviews_done = 0

    # Get weights or use defaults
    w_success = (
        weights["success"]
        if weights
        else [DEFAULT_PROB_HARD, DEFAULT_PROB_GOOD, DEFAULT_PROB_EASY]
    )
    w_first = (
        weights["first"]
        if weights
        else [
            DEFAULT_PROB_FIRST_AGAIN,
            DEFAULT_PROB_FIRST_HARD,
            DEFAULT_PROB_FIRST_GOOD,
            DEFAULT_PROB_FIRST_EASY,
        ]
    )

    for idx in due_indices:
        if reviews_done >= review_limit:
            break

        sys_card = sys_cards[idx]
        true_card = true_cards[idx]
        card_id = sys_card.card_id

        retrievability = nature_scheduler.get_card_retrievability(
            true_card, current_date
        )

        if random.random() < retrievability:
            rating = random.choices(
                [Rating.Hard, Rating.Good, Rating.Easy],
                weights=w_success,
            )[0]
        else:
            rating = Rating.Again

        updated_true_card, _ = nature_scheduler.review_card(
            true_card, rating, current_date
        )
        true_cards[idx] = updated_true_card

        updated_sys_card, log = algo_scheduler.review_card(
            sys_card, rating, current_date
        )
        sys_cards[idx] = updated_sys_card

        card_logs[card_id].append(log)
        reviews_done += 1

    while reviews_done < review_limit:
        base_card = Card()
        true_card = deepcopy(base_card)
        sys_card = deepcopy(base_card)

        rating = random.choices(
            [Rating.Again, Rating.Hard, Rating.Good, Rating.Easy],
            weights=w_first,
        )[0]

        updated_true_card, _ = nature_scheduler.review_card(
            true_card, rating, current_date
        )
        updated_sys_card, log = algo_scheduler.review_card(
            sys_card, rating, current_date
        )

        true_cards.append(updated_true_card)
        sys_cards.append(updated_sys_card)
        card_logs[updated_sys_card.card_id].append(log)
        reviews_done += 1


def parse_retention_schedule(
    schedule_str: str,
) -> list[tuple[int, float]]:
    try:
        if ":" not in schedule_str:
            return [(1, float(schedule_str))]
        segments = []
        for part in schedule_str.split(","):
            d, r = part.split(":")
            segments.append((int(d), float(r)))
        return segments
    except ValueError:
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
    try:
        parts = [float(p.strip()) for p in params_str.split(",")]
        if len(parts) != 21:
            return tuple(DEFAULT_PARAMETERS)
        return tuple(parts)
    except ValueError:
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
    deck_config: str | None = None,
    deck_name: str | None = None,
    initial_params: tuple[float, ...] | None = None,
    seeded_data: dict[str, Any] | None = None,
) -> tuple[list[float] | None, tuple[float, ...], dict[str, Any]]:
    parsed_schedule = parse_retention_schedule(retention)
    initial_retention = get_retention_for_day(0, parsed_schedule)

    random.seed(seed)
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
        current_date = START_DATE
        review_weights = None

        if seeded_data:
            # Use pre-calculated data
            current_date = seeded_data["last_rev"] + timedelta(days=1)
            # We must deepcopy card objects because they are
            # modified in-place during simulation.
            # However, we convert the dict values to a list for O(1) NumPy access.
            true_cards = list(deepcopy(seeded_data["true_cards"]).values())
            sys_cards = list(deepcopy(seeded_data["sys_cards"]).values())
            card_logs = deepcopy(seeded_data["logs"])
            review_weights = seeded_data.get("weights")
        elif seed_history:
            # Fallback to loading from disk (e.g. single simulation run)
            logs, last_rev = load_anki_history(seed_history, deck_config, deck_name)
            current_date = last_rev + timedelta(days=1)
            review_weights = infer_review_weights(logs)
            for cid, logs_list in logs.items():
                card = Card(card_id=cid)
                true_cards.append(nature_scheduler.reschedule_card(card, logs_list))
                sys_cards.append(algo_scheduler.reschedule_card(card, logs_list))
                card_logs[cid].extend(logs_list)

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
                weights=review_weights,
            )
            current_date += timedelta(days=1)

        if burn_in_days > 0 and burn_in_days < n_days:
            all_logs = [log for logs in card_logs.values() for log in logs]
            if len(all_logs) >= 512:
                optimizer = RustOptimizer(all_logs)
                fitted_bi = optimizer.compute_optimal_parameters(verbose=verbose)
                algo_scheduler = Scheduler(
                    parameters=tuple(fitted_bi),
                    desired_retention=algo_scheduler.desired_retention,
                )
                # Reschedule with new params
                for i, card in enumerate(sys_cards):
                    sys_cards[i] = algo_scheduler.reschedule_card(
                        Card(card_id=card.card_id), card_logs[card.card_id]
                    )

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
                    weights=review_weights,
                )
                current_date += timedelta(days=1)

        all_logs_final = [log for logs in card_logs.values() for log in logs]

        fitted_params = None
        try:
            optimizer = RustOptimizer(all_logs_final)
            fitted_params = optimizer.compute_optimal_parameters(verbose=verbose)
        except Exception:
            pass

        stabilities = []
        if fitted_params:
            final_algo_scheduler = Scheduler(parameters=tuple(fitted_params))
            for i in range(len(true_cards)):
                card_id = true_cards[i].card_id
                s_nat = true_cards[i].stability or 0.0
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
            },
        )
    finally:
        fsrs.optimizer.tqdm = original_optimizer_tqdm


def run_simulation_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--reviews", type=int, default=200)
    parser.add_argument("--retention", type=str, default="0.9")
    parser.add_argument("--burn-in", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ground-truth", type=str)
    parser.add_argument("--seed-history", type=str)
    parser.add_argument("--deck-config", type=str)
    parser.add_argument("--deck-name", type=str)

    args = parser.parse_args()
    gt_params = parse_parameters(args.ground_truth) if args.ground_truth else None

    # Handle pre-fitting if history is provided
    initial_params = None
    if args.seed_history:
        logs, _ = load_anki_history(args.seed_history, args.deck_config, args.deck_name)
        flat_logs = [log for card_logs in logs.values() for log in card_logs]
        if len(flat_logs) >= 512:
            opt = RustOptimizer(flat_logs)
            initial_params = tuple(opt.compute_optimal_parameters())

    run_simulation(
        n_days=args.days,
        review_limit=args.reviews,
        retention=args.retention,
        burn_in_days=args.burn_in,
        seed=args.seed,
        ground_truth=gt_params,
        seed_history=args.seed_history,
        deck_config=args.deck_config,
        deck_name=args.deck_name,
        initial_params=initial_params,
    )


if __name__ == "__main__":
    run_simulation_cli()
