import json
import os
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from fsrs import Card, Rating, ReviewLog, Scheduler
from tqdm import tqdm


@dataclass
class RatingWeights:
    first: list[float] = field(default_factory=lambda: [0.5, 0.1, 0.3, 0.1])
    success: list[float] = field(default_factory=lambda: [0.1, 0.8, 0.1])


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


def calculate_expected_d0(weights: list[float], parameters: tuple[float, ...]) -> float:
    """
    Calculates expected initial difficulty E[D0(G)] based on first-rating
    distribution and FSRS v6 parameters w4, w5.
    Formula: D0(G) = w4 - exp(w5*(G-1)) + 1
    """
    import math

    w4 = parameters[4]
    w5 = parameters[5]
    # G values: 1 (Again), 2 (Hard), 3 (Good), 4 (Easy)
    d0_vals = [w4 - math.exp(w5 * (g - 1)) + 1 for g in [1, 2, 3, 4]]
    return sum(p * d for p, d in zip(weights, d0_vals, strict=False))


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
            import sys

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


def infer_review_weights(
    card_logs: dict[int, list[ReviewLog]],
) -> RatingWeights:
    """
    Infers rating probabilities from real review history.
    Returns RatingWeights for first reviews and subsequent successful reviews.
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

    return RatingWeights(first=first_weights, success=success_weights)


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
    w_first = weights_inf.first
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
