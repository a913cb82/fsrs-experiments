import argparse
import json
import math
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
DEFAULT_PROB_HARD = 0.1
DEFAULT_PROB_GOOD = 0.8
DEFAULT_PROB_EASY = 0.1

# Weights for new cards (first review ratings) - Defaults
DEFAULT_PROB_FIRST_AGAIN = 0.5
DEFAULT_PROB_FIRST_HARD = 0.1
DEFAULT_PROB_FIRST_GOOD = 0.3
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
    "get_review_history_stats",
    "calculate_expected_d0",
]


def calculate_expected_d0(weights: list[float], parameters: tuple[float, ...]) -> float:
    """
    Calculates expected initial difficulty E[D0(G)] based on first-rating
    distribution and FSRS v6 parameters w4, w5.
    Formula: D0(G) = w4 - exp(w5*(G-1)) + 1
    """
    w4 = parameters[4]
    w5 = parameters[5]
    # G values: 1 (Again), 2 (Hard), 3 (Good), 4 (Easy)
    d0_vals = [w4 - math.exp(w5 * (g - 1)) + 1 for g in [1, 2, 3, 4]]
    return sum(p * d for p, d in zip(weights, d0_vals, strict=False))


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


def get_field_from_proto(blob: bytes, field_no: int) -> int | None:
    """Extracts a varint field from a Protobuf blob."""
    if not blob:
        return None
    pos = 0
    while pos < len(blob):
        try:
            tag, pos = decode_varint(blob, pos)
            f_num = tag >> 3
            w_type = tag & 0x07
            if f_num == field_no and w_type == 0:
                val, pos = decode_varint(blob, pos)
                return val
            # Skip field
            if w_type == 0:
                _, pos = decode_varint(blob, pos)
            elif w_type == 1:
                pos += 8
            elif w_type == 2:
                length, pos = decode_varint(blob, pos)
                pos += length
            elif w_type == 5:
                pos += 4
            else:
                break
        except Exception:
            break
    return None


def get_deck_config_id(common_blob: bytes, kind_blob: bytes) -> int:
    """
    Extracts config_id from Anki's DeckCommon or NormalDeck blobs.
    Defaults to 1 if not found.
    """
    # 1. Try NormalDeck.config_id (field 1 of sub-message in field 1 of kind blob)
    if kind_blob:
        pos = 0
        while pos < len(kind_blob):
            try:
                tag, pos = decode_varint(kind_blob, pos)
                f_num = tag >> 3
                w_type = tag & 0x07
                if f_num == 1 and w_type == 2:  # normal field
                    length, pos = decode_varint(kind_blob, pos)
                    normal_deck_blob = kind_blob[pos : pos + length]
                    cid = get_field_from_proto(normal_deck_blob, 1)
                    if cid is not None:
                        return cid
                    break
                # Skip other fields in kind
                if w_type == 0:
                    _, pos = decode_varint(kind_blob, pos)
                elif w_type == 1:
                    pos += 8
                elif w_type == 2:
                    length, pos = decode_varint(kind_blob, pos)
                    pos += length
                elif w_type == 5:
                    pos += 4
                else:
                    break
            except Exception:
                break

    # 2. Try DeckCommon.config_id (field 1 of common blob)
    cid = get_field_from_proto(common_blob, 1)
    if cid is not None:
        return cid

    return 1


def load_anki_history(
    path: str,
    deck_config_name: str | None = None,
    deck_name: str | None = None,
) -> tuple[dict[int, list[ReviewLog]], datetime]:
    """
    Extracts FSRS-compatible review logs from an Anki collection.anki2 file.
    Supports both old (JSON-in-col) and new (relational tables) schemas.
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
        tqdm.write(f"Anki database version {ver} detected.")

        valid_deck_ids: list[int] = []
        config_id_to_name: dict[int, str] = {}
        deck_to_config: dict[int, int] = {}

        if ver >= 18:
            # New relational schema (Anki 23.10+)
            cur.execute("SELECT id, name, common, kind FROM decks")
            decks = cur.fetchall()

            cur.execute("SELECT id, name FROM deck_config")
            config_id_to_name = {int(row[0]): str(row[1]) for row in cur.fetchall()}

            # Map deck names to their rows for inheritance lookup
            d_name_map = {str(row[1]): row for row in decks}

            for row in decks:
                if row[0] is None or row[1] is None:
                    continue
                d_id, d_name, d_common_raw, d_kind_raw = (
                    int(row[0]),
                    str(row[1]),
                    row[2],
                    row[3],
                )
                d_common = bytes(d_common_raw) if d_common_raw is not None else b""
                d_kind = bytes(d_kind_raw) if d_kind_raw is not None else b""

                cid_found = get_deck_config_id(d_common, d_kind)

                # Inheritance logic
                if cid_found == 1:
                    parts = d_name.split("::")
                    for i in range(len(parts) - 1, 0, -1):
                        p_name = "::".join(parts[:i])
                        if p_name in d_name_map:
                            p_row = d_name_map[p_name]
                            p_common = bytes(p_row[2]) if p_row[2] is not None else b""
                            p_kind = bytes(p_row[3]) if p_row[3] is not None else b""
                            p_cid = get_deck_config_id(p_common, p_kind)
                            if p_cid != 1:
                                cid_found = p_cid
                                break
                deck_to_config[d_id] = cid_found

            # Filter logic
            target_cid = None
            if deck_config_name:
                for cid, name in config_id_to_name.items():
                    if name == deck_config_name:
                        target_cid = cid
                        break

            for row in decks:
                if row[0] is None or row[1] is None:
                    continue
                d_id, d_name = int(row[0]), str(row[1])
                if deck_name and d_name != deck_name:
                    continue
                if deck_config_name:
                    if target_cid is None or deck_to_config.get(d_id) != target_cid:
                        continue
                valid_deck_ids.append(d_id)

        else:
            # Old JSON schema (pre-23.10)
            cur.execute("SELECT decks, dconf FROM col")
            col_row = cur.fetchone()
            if not col_row:
                return {}, START_DATE

            decks_json = json.loads(col_row[0]) if col_row[0] else {}
            dconf_json = json.loads(col_row[1]) if col_row[1] else {}

            for cid_s, cfg in dconf_json.items():
                config_id_to_name[int(cid_s)] = str(cfg.get("name", "Unknown"))

            target_cid = None
            if deck_config_name:
                for cid, name in config_id_to_name.items():
                    if name == deck_config_name:
                        target_cid = cid
                        break

            for did_s, deck in decks_json.items():
                did = int(did_s)
                cid = int(deck.get("conf", 1))
                deck_to_config[did] = cid

                d_name = str(deck.get("name", ""))
                if deck_name and d_name != deck_name:
                    continue
                if deck_config_name:
                    if target_cid is None or cid != target_cid:
                        continue
                valid_deck_ids.append(did)

        # Informative error if deck_config_name filter matched nothing
        if deck_config_name and not valid_deck_ids:
            cur.execute("SELECT did, count(*) FROM cards GROUP BY did")
            cards_per_deck = {int(r[0]): int(r[1]) for r in cur.fetchall()}

            cards_per_config: dict[str, int] = defaultdict(int)
            for did, cid in deck_to_config.items():
                cname = config_id_to_name.get(cid, f"ID {cid}")
                cards_per_config[cname] += cards_per_deck.get(did, 0)

            stats_list = [
                f"  - {name}: {count} cards"
                for name, count in sorted(cards_per_config.items())
            ]
            stats_str = "\n".join(stats_list)
            error_msg = (
                f"Error: Deck configuration '{deck_config_name}' matched 0 cards.\n"
                f"Available configurations and card counts:\n{stats_str}"
            )
            tqdm.write(error_msg)
            sys.exit(1)

        if not valid_deck_ids:
            tqdm.write("Warning: No matching decks found for filtering criteria.")
            return {}, START_DATE

        tqdm.write(f"Querying reviews for {len(valid_deck_ids)} matching decks...")
        placeholders = ",".join(["?" for _ in valid_deck_ids])
        query = f"""
            SELECT r.cid, r.ease, r.id, r.type, r.time
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
    first_review_time = None
    last_review_time = START_DATE

    for cid_v, ease, rev_id, _rev_type, duration_ms in rows:
        cid = int(cid_v)
        dt = datetime.fromtimestamp(rev_id / 1000.0, tz=timezone.utc)
        log = ReviewLog(
            card_id=cid,
            rating=Rating(ease),
            review_datetime=dt,
            review_duration=int(duration_ms) if duration_ms else None,
        )
        card_logs[cid].append(log)
        if first_review_time is None or dt < first_review_time:
            first_review_time = dt
        if dt > last_review_time:
            last_review_time = dt

    total_revs = len(rows)
    tqdm.write(f"Successfully loaded {total_revs} reviews for {len(card_logs)} cards.")
    if first_review_time is not None:
        tqdm.write(
            f"Review date range: {first_review_time.date()} to "
            f"{last_review_time.date()}"
        )

    return card_logs, last_review_time


def get_review_history_stats(
    card_logs: dict[int, list[ReviewLog]],
    parameters: tuple[float, ...],
) -> list[dict[str, Any]]:
    """
    Replays review history using provided parameters and returns a list of
    stats for each review (retention, rating, duration, etc).
    Includes first reviews (new cards) with stability 0.0 and expected D0.
    """
    scheduler = Scheduler(parameters=parameters)

    # Inferred features for new cards
    weights_inf = infer_review_weights(card_logs)
    w_first = weights_inf["first"]
    prob_first_success = 1.0 - w_first[0]
    expected_d0 = calculate_expected_d0(w_first, parameters)

    stats = []

    for cid, logs in card_logs.items():
        if not logs:
            continue

        sorted_logs = sorted(logs, key=lambda x: x.review_datetime)

        card = Card(card_id=cid)
        # Replay history
        for i, log in enumerate(sorted_logs):
            if i == 0:
                # Features for a BRAND NEW card
                ret = prob_first_success
                stab = 0.0
                diff = expected_d0
                elapsed = 0.0
            else:
                # Retention at the time of THIS review
                ret = scheduler.get_card_retrievability(card, log.review_datetime)
                stab = card.stability
                diff = card.difficulty
                elapsed = (
                    log.review_datetime - card.last_review
                ).total_seconds() / 86400.0

            stats.append(
                {
                    "card_id": cid,
                    "retention": ret,
                    "rating": int(log.rating),
                    "duration": log.review_duration,
                    "stability": stab,
                    "difficulty": diff,
                    "elapsed_days": elapsed,
                }
            )

            # Update card state for the NEXT review
            card, _ = scheduler.review_card(card, log.rating, log.review_datetime)

    return stats


def simulate_day(
    nature_scheduler: Scheduler,
    algo_scheduler: Scheduler,
    true_cards: list[Card],
    sys_cards: list[Card],
    card_logs: dict[int, list[ReviewLog]],
    current_date: datetime,
    review_limit: int | None,
    new_limit: int | None,
    weights: dict[str, list[float]] | None = None,
    time_limit: float | None = None,
    time_estimator: Any | None = None,
) -> None:
    # Use NumPy for fast due-check across the deck
    dues = np.array([c.due for c in sys_cards])
    due_mask = dues <= current_date
    due_indices = np.where(due_mask)[0].tolist()

    # Sort indices to keep deterministic if needed (NumPy where is stable)
    due_indices = sorted(due_indices)
    random.shuffle(due_indices)
    reviews_done = 0
    new_done = 0
    time_accumulated = 0.0

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
        # Check limits
        if time_limit is not None:
            if time_accumulated >= time_limit:
                break
        elif review_limit is not None and reviews_done >= review_limit:
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

        # Estimate time if estimator is provided
        if time_estimator is not None:
            # Estimator can take (card, rating, current_date)
            review_time = time_estimator(sys_card, rating, current_date)
            if time_limit is not None and time_accumulated + review_time > time_limit:
                break
            time_accumulated += review_time

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

    # New cards
    while True:
        if time_limit is not None:
            if time_accumulated >= time_limit:
                break
        elif review_limit is not None and reviews_done >= review_limit:
            break
        elif new_limit is not None and new_done >= new_limit:
            break

        base_card = Card()
        true_card = deepcopy(base_card)
        sys_card = deepcopy(base_card)

        rating = random.choices(
            [Rating.Again, Rating.Hard, Rating.Good, Rating.Easy],
            weights=w_first,
        )[0]

        # Estimate time for new card
        if time_estimator is not None:
            review_time = time_estimator(sys_card, rating, current_date)
            if time_limit is not None and time_accumulated + review_time > time_limit:
                break
            time_accumulated += review_time

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
        new_done += 1


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
    review_limit: int | None = 200,
    new_limit: int | None = 10,
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
    time_limit: float | None = None,
    time_estimator: Any | None = None,
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
            current_date = seeded_data["last_rev"] + timedelta(days=1)
            true_cards = list(deepcopy(seeded_data["true_cards"]).values())
            sys_cards = list(deepcopy(seeded_data["sys_cards"]).values())
            card_logs = deepcopy(seeded_data["logs"])
            review_weights = seeded_data.get("weights")
        elif seed_history:
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
                new_limit,
                weights=review_weights,
                time_limit=time_limit,
                time_estimator=time_estimator,
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
                    new_limit,
                    weights=review_weights,
                    time_limit=time_limit,
                    time_estimator=time_estimator,
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
        total_retention = 0.0
        if fitted_params:
            final_algo_scheduler = Scheduler(parameters=tuple(fitted_params))
            for i in range(len(true_cards)):
                card_id = true_cards[i].card_id
                s_nat = true_cards[i].stability or 0.0

                # Nature's actual retrievability at the end of simulation
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
