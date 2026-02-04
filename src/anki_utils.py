import json
import math
import os
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, cast

import pandas as pd
from fsrs import Card, Rating, Scheduler
from tqdm import tqdm

from proto_utils import get_deck_config_id

# Constants for Anki processing
START_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)

# Weights for new cards (first review ratings) - Defaults
DEFAULT_PROB_FIRST_AGAIN = 0.5
DEFAULT_PROB_FIRST_HARD = 0.1
DEFAULT_PROB_FIRST_GOOD = 0.3
DEFAULT_PROB_FIRST_EASY = 0.1

# Probabilities given recall (Success) - Defaults
DEFAULT_PROB_HARD = 0.1
DEFAULT_PROB_GOOD = 0.8
DEFAULT_PROB_EASY = 0.1


@dataclass
class RatingWeights:
    first: list[float] = field(default_factory=lambda: [0.5, 0.1, 0.3, 0.1])
    success: list[float] = field(default_factory=lambda: [0.1, 0.8, 0.1])


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


def _get_anki_schema_version(cur: sqlite3.Cursor) -> int:
    cur.execute("SELECT ver FROM col")
    v_row = cur.fetchone()
    return int(v_row[0]) if v_row and v_row[0] is not None else 0


def _get_valid_deck_ids_v18(
    cur: sqlite3.Cursor,
    deck_config_name: str | None = None,
    deck_name: str | None = None,
) -> list[int]:
    """Helper for Anki 23.10+ schema."""
    cur.execute("SELECT id, name, common, kind FROM decks")
    decks = cur.fetchall()

    cur.execute("SELECT id, name FROM deck_config")
    config_id_to_name = {int(row[0]): str(row[1]) for row in cur.fetchall()}

    d_name_map = {str(row[1]): row for row in decks}
    deck_to_config: dict[int, int] = {}

    for row in decks:
        if row[0] is None or row[1] is None:
            continue
        d_id, d_name = int(row[0]), str(row[1])
        d_common = bytes(row[2]) if row[2] is not None else b""
        d_kind = bytes(row[3]) if row[3] is not None else b""

        cid_found = get_deck_config_id(d_common, d_kind)

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

    target_cid = None
    if deck_config_name:
        for cid, name in config_id_to_name.items():
            if name == deck_config_name:
                target_cid = cid
                break

    valid_ids = []
    for row in decks:
        if row[0] is None or row[1] is None:
            continue
        d_id, d_name = int(row[0]), str(row[1])
        if deck_name and d_name != deck_name:
            continue
        if deck_config_name:
            if target_cid is None or deck_to_config.get(d_id) != target_cid:
                continue
        valid_ids.append(d_id)

    if deck_config_name and not valid_ids:
        _raise_informative_config_error(
            cur, deck_config_name, config_id_to_name, deck_to_config
        )

    return valid_ids


def _get_valid_deck_ids_legacy(
    cur: sqlite3.Cursor,
    deck_config_name: str | None = None,
    deck_name: str | None = None,
) -> list[int]:
    """Helper for legacy JSON-in-col schema."""
    cur.execute("SELECT decks, dconf FROM col")
    col_row = cur.fetchone()
    if not col_row:
        return []

    decks_json = json.loads(col_row[0]) if col_row[0] else {}
    dconf_json = json.loads(col_row[1]) if col_row[1] else {}

    config_id_to_name = {
        int(cid_s): str(cfg.get("name", "Unknown")) for cid_s, cfg in dconf_json.items()
    }

    target_cid = None
    if deck_config_name:
        for cid, name in config_id_to_name.items():
            if name == deck_config_name:
                target_cid = cid
                break

    valid_ids = []
    deck_to_config = {}
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
        valid_ids.append(did)

    if deck_config_name and not valid_ids:
        _raise_informative_config_error(
            cur, deck_config_name, config_id_to_name, deck_to_config
        )

    return valid_ids


def _raise_informative_config_error(
    cur: sqlite3.Cursor,
    target_name: str,
    config_id_to_name: dict[int, str],
    deck_to_config: dict[int, int],
) -> None:
    cur.execute("SELECT did, count(*) FROM cards GROUP BY did")
    cards_per_deck = {int(r[0]): int(r[1]) for r in cur.fetchall()}

    cards_per_config: dict[str, int] = defaultdict(int)
    for did, cid in deck_to_config.items():
        cname = config_id_to_name.get(cid, f"ID {cid}")
        cards_per_config[cname] += cards_per_deck.get(did, 0)

    stats_list = [
        f"  - {name}: {count} cards" for name, count in sorted(cards_per_config.items())
    ]
    stats_str = "\n".join(stats_list)
    error_msg = (
        f"Error: Deck configuration '{target_name}' matched 0 cards.\n"
        f"Available configurations and card counts:\n{stats_str}"
    )
    tqdm.write(error_msg)
    import sys

    sys.exit(1)


def load_anki_history(
    path: str,
    deck_config_name: str | None = None,
    deck_name: str | None = None,
) -> tuple[pd.DataFrame, datetime]:
    """
    Extracts FSRS-compatible review logs from an Anki collection.anki2 file.
    Supports both old (JSON-in-col) and new (relational tables) schemas.
    Returns a DataFrame with columns: card_id, rating, review_datetime, review_duration
    """
    if not os.path.exists(path):
        tqdm.write(f"Error: Anki database not found at {path}")
        return pd.DataFrame(), START_DATE

    conn = sqlite3.connect(path)
    cur = conn.cursor()

    try:
        ver = _get_anki_schema_version(cur)
        tqdm.write(f"Anki database version {ver} detected.")

        if ver >= 18:
            valid_deck_ids = _get_valid_deck_ids_v18(cur, deck_config_name, deck_name)
        else:
            valid_deck_ids = _get_valid_deck_ids_legacy(
                cur, deck_config_name, deck_name
            )

        if not valid_deck_ids:
            tqdm.write("Warning: No matching decks found for filtering criteria.")
            return pd.DataFrame(), START_DATE

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
        return pd.DataFrame(), START_DATE
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame(), START_DATE

    df = pd.DataFrame(
        rows, columns=["card_id", "rating", "id", "type", "review_duration"]
    )
    df["review_datetime"] = pd.to_datetime(df["id"], unit="ms", utc=True)
    df["rating"] = df["rating"].astype(int)
    # duration can be None if duration_ms is 0 or null
    df["review_duration"] = df["review_duration"].apply(lambda x: int(x) if x else None)

    last_review_time = df["review_datetime"].max()
    first_review_time = df["review_datetime"].min()

    t_df = df[["card_id", "rating", "review_datetime", "review_duration"]]

    tqdm.write(
        f"Successfully loaded {len(df)} reviews for {df['card_id'].nunique()} cards."
    )
    tqdm.write(
        f"Review date range: {first_review_time.date()} to {last_review_time.date()}"
    )

    return t_df, last_review_time


def infer_review_weights(
    card_logs: pd.DataFrame,
) -> RatingWeights:
    """
    Infers rating probabilities from real review history.
    card_logs is a DataFrame with [card_id, rating, review_datetime]
    """
    if card_logs.empty:
        return RatingWeights()

    sorted_df = card_logs.sort_values("review_datetime").copy()
    sorted_df["rank"] = sorted_df.groupby("card_id")["review_datetime"].rank(
        method="first"
    )

    first_reviews = sorted_df[sorted_df["rank"] == 1]
    subsequent_reviews = sorted_df[sorted_df["rank"] > 1]

    first_ratings_counts = (
        first_reviews["rating"].value_counts().reindex([1, 2, 3, 4], fill_value=0)
    )
    success_subsequent = subsequent_reviews[subsequent_reviews["rating"] > 1]
    success_ratings_counts = (
        success_subsequent["rating"].value_counts().reindex([2, 3, 4], fill_value=0)
    )

    # Normalize First Ratings
    total_first = first_ratings_counts.sum()
    if total_first > 0:
        first_weights = (first_ratings_counts / total_first).tolist()
    else:
        first_weights = [
            DEFAULT_PROB_FIRST_AGAIN,
            DEFAULT_PROB_FIRST_HARD,
            DEFAULT_PROB_FIRST_GOOD,
            DEFAULT_PROB_FIRST_EASY,
        ]

    # Normalize Success Ratings
    total_success = success_ratings_counts.sum()
    if total_success > 0:
        success_weights = (success_ratings_counts / total_success).tolist()
    else:
        success_weights = [
            DEFAULT_PROB_HARD,
            DEFAULT_PROB_GOOD,
            DEFAULT_PROB_EASY,
        ]

    return RatingWeights(first=first_weights, success=success_weights)


def get_review_history_stats(
    card_logs: pd.DataFrame,
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
    w_first = weights_inf.first
    prob_first_success = 1.0 - w_first[0]
    expected_d0 = calculate_expected_d0(w_first, parameters)

    stats = []

    for cid, group in card_logs.groupby("card_id"):
        sorted_group = group.sort_values("review_datetime")

        card = Card(card_id=cid)
        # Replay history
        for i, row in enumerate(sorted_group.itertuples()):
            if i == 0:
                # Features for a BRAND NEW card
                ret = prob_first_success
                stab = 0.0
                diff = expected_d0
                elapsed = 0.0
            else:
                # Retention at the time of THIS review
                ret = scheduler.get_card_retrievability(card, row.review_datetime)
                stab = card.stability
                diff = card.difficulty
                elapsed = (
                    row.review_datetime - card.last_review
                ).total_seconds() / 86400.0

            stats.append(
                {
                    "card_id": cid,
                    "retention": ret,
                    "rating": int(cast(Any, row).rating),
                    "duration": row.review_duration,
                    "stability": stab,
                    "difficulty": diff,
                    "elapsed_days": elapsed,
                }
            )

            # Update card state for the NEXT review
            card, _ = scheduler.review_card(
                card, Rating(int(cast(Any, row).rating)), row.review_datetime
            )

    return stats
