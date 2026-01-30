import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from fsrs import Card, Rating, ReviewLog
from fsrs.scheduler import DEFAULT_PARAMETERS

import src.simulate_fsrs
from src.anki_utils import (
    START_DATE,
    get_deck_config_id,
    get_field_from_proto,
    get_review_history_stats,
    load_anki_history,
)
from src.simulate_fsrs import (
    RustOptimizer,
    run_simulation,
)
from src.simulation_config import RatingWeights, SeededData, SimulationConfig
from src.utils import get_retention_for_day, parse_parameters, parse_retention_schedule


def test_rust_optimizer() -> None:
    now = datetime.now(timezone.utc)
    logs = [
        ReviewLog(
            card_id=1,
            rating=Rating.Good,
            review_datetime=now - timedelta(days=10),
            review_duration=None,
        ),
        ReviewLog(
            card_id=1,
            rating=Rating.Good,
            review_datetime=now - timedelta(days=5),
            review_duration=None,
        ),
    ]
    opt = RustOptimizer(logs)
    params = opt.compute_optimal_parameters()
    assert len(params) == 21
    assert any(p != 0 for p in params)


def test_rust_optimizer_no_items() -> None:
    now = datetime.now(timezone.utc)
    logs = [
        ReviewLog(
            card_id=1,
            rating=Rating.Good,
            review_datetime=now,
            review_duration=None,
        )
    ]
    opt = RustOptimizer(logs)
    params = opt.compute_optimal_parameters()
    assert len(params) == 21


def test_run_simulation_basic() -> None:
    config = SimulationConfig(n_days=2, review_limit=10, verbose=False)
    fitted, gt, metrics = run_simulation(config)
    assert gt == DEFAULT_PARAMETERS
    assert fitted is not None
    assert len(fitted) == 21
    assert metrics["review_count"] > 0
    assert metrics["card_count"] > 0


def test_run_simulation_with_seeded_data() -> None:
    now = datetime.now(timezone.utc)
    logs: dict[int, list[ReviewLog]] = defaultdict(list)
    log = ReviewLog(
        card_id=1,
        rating=Rating.Good,
        review_datetime=now - timedelta(days=1),
        review_duration=None,
    )
    logs[1] = [log]
    true_cards = {1: Card(card_id=1, due=now)}
    sys_cards = {1: Card(card_id=1, due=now)}

    seeded_payload = SeededData(
        logs=logs,
        last_rev=now - timedelta(days=1),
        true_cards=true_cards,
        sys_cards=sys_cards,
    )

    config = SimulationConfig(n_days=2, review_limit=5, verbose=False)
    fitted, _, metrics = run_simulation(config, seeded_data=seeded_payload)
    assert metrics["card_count"] >= 1
    assert metrics["review_count"] > 1


def test_run_simulation_with_burn_in() -> None:
    config = SimulationConfig(
        n_days=10, burn_in_days=5, review_limit=100, verbose=False
    )
    fitted, _, _ = run_simulation(config)
    assert fitted is not None


def test_run_simulation_with_burn_in_triggered() -> None:
    config = SimulationConfig(
        n_days=10, burn_in_days=5, review_limit=150, new_limit=150, verbose=False
    )
    fitted, _, metrics = run_simulation(config)
    assert fitted is not None
    assert metrics["review_count"] >= 512


def test_simulate_day_again_branch() -> None:
    config = SimulationConfig(
        n_days=1, review_limit=10, retention="0.01", verbose=False
    )
    _, _, metrics = run_simulation(config)
    assert metrics["review_count"] > 0


def test_simulate_day_review_limit_break() -> None:
    test_db = "tests/test_collection.anki2"
    config = SimulationConfig(n_days=100, review_limit=1, verbose=False)
    _, _, metrics = run_simulation(config, seed_history=test_db)
    assert metrics["review_count"] == 105


def test_run_simulation_no_optimizer_triggered() -> None:
    config = SimulationConfig(n_days=1, review_limit=10, verbose=False)
    fitted, _, _ = run_simulation(config)
    assert fitted is not None


def test_parse_retention_schedule_error() -> None:
    assert parse_retention_schedule("invalid:schedule") == [(1, 0.9)]
    assert parse_retention_schedule("invalid") == [(1, 0.9)]


def test_load_anki_history_missing_file() -> None:
    logs, date = load_anki_history("non_existent.anki2")
    assert logs == {}
    assert date == START_DATE


def test_run_simulation_with_custom_weights() -> None:
    weights = RatingWeights(first=[0.1, 0.1, 0.7, 0.1], success=[0.1, 0.8, 0.1])
    config = SimulationConfig(n_days=5, review_limit=10, verbose=False)
    fitted, _, metrics = run_simulation(
        config,
        seeded_data=SeededData(
            last_rev=datetime.now(timezone.utc),
            true_cards={},
            sys_cards={},
            logs=defaultdict(list),
            weights=weights,
        ),
    )
    assert fitted is not None
    assert metrics["review_count"] > 0


def test_run_simulation_with_rating_estimator() -> None:
    def rating_estimator(card: Card, current_date: datetime) -> int:
        return 4  # Always 'Easy'

    config = SimulationConfig(
        n_days=5,
        review_limit=20,
        new_limit=10,
        rating_estimator=rating_estimator,
        verbose=False,
    )
    _, _, metrics = run_simulation(config)

    assert metrics["review_count"] > 0
    for log in metrics["logs"]:
        assert log.rating == Rating.Easy


def test_get_review_history_stats() -> None:
    now = datetime.now(timezone.utc)
    logs = {
        1: [
            ReviewLog(
                card_id=1,
                rating=Rating.Good,
                review_datetime=now - timedelta(days=10),
                review_duration=5000,
            ),
            ReviewLog(
                card_id=1,
                rating=Rating.Good,
                review_datetime=now - timedelta(days=5),
                review_duration=10000,
            ),
        ]
    }
    stats = get_review_history_stats(logs, DEFAULT_PARAMETERS)
    assert len(stats) == 2
    assert stats[0]["elapsed_days"] == 0.0
    assert stats[1]["elapsed_days"] == 5.0


def test_get_field_from_proto_skips() -> None:
    blob_w0 = bytes([0x10, 1, 0x08, 10])
    assert get_field_from_proto(blob_w0, 1) == 10
    blob_w1 = bytes([(2 << 3) | 1]) + bytes([0] * 8) + bytes([(1 << 3) | 0, 10])
    assert get_field_from_proto(blob_w1, 1) == 10
    blob_w2 = bytes([(2 << 3) | 2, 3, 1, 2, 3]) + bytes([(1 << 3) | 0, 20])
    assert get_field_from_proto(blob_w2, 1) == 20
    blob_w5 = bytes([(2 << 3) | 5]) + bytes([0] * 4) + bytes([(1 << 3) | 0, 30])
    assert get_field_from_proto(blob_w5, 1) == 30
    blob_unknown = bytes([(2 << 3) | 4]) + bytes([(1 << 3) | 0, 40])
    assert get_field_from_proto(blob_unknown, 1) is None


def test_get_field_from_proto_exception() -> None:
    assert get_field_from_proto(bytes([0x08]), 1) is None


def test_get_deck_config_id_kind() -> None:
    normal_deck_blob = bytes([0x08, 100])
    kind_blob = bytes([0x0A, len(normal_deck_blob)]) + normal_deck_blob
    assert get_deck_config_id(b"", kind_blob) == 100
    kind_blob_skip = (
        bytes(
            [
                0x10,
                1,
                0x19,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0x22,
                3,
                1,
                2,
                3,
                0x2D,
                0,
                0,
                0,
                0,
                0x0A,
                len(normal_deck_blob),
            ]
        )
        + normal_deck_blob
    )
    assert get_deck_config_id(b"", kind_blob_skip) == 100
    normal_deck_blob_no_cid = bytes([0x10, 100])
    kind_blob_no_cid = (
        bytes([0x0A, len(normal_deck_blob_no_cid)]) + normal_deck_blob_no_cid
    )
    assert get_deck_config_id(b"", kind_blob_no_cid) == 1
    assert get_deck_config_id(b"", bytes([0x0A])) == 1


def test_load_anki_history_none_rows(tmp_path: Any) -> None:
    db_path = str(tmp_path / "none_collection.anki2")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE col (id integer primary key, ver integer)")
    cur.execute("INSERT INTO col (id, ver) VALUES (1, 18)")
    cur.execute("CREATE TABLE decks (id integer, name text, common blob, kind blob)")
    cur.execute(
        "INSERT INTO decks (id, name, common, kind) VALUES (NULL, 'NullID', x'', x'')"
    )
    cur.execute("CREATE TABLE deck_config (id integer, name text)")
    cur.execute("CREATE TABLE cards (id integer, did integer)")
    cur.execute(
        "CREATE TABLE revlog "
        "(id integer, cid integer, ease integer, type integer, time integer)"
    )
    conn.commit()
    conn.close()
    logs, _ = load_anki_history(db_path)
    assert logs == {}


def test_load_anki_history_empty_col(tmp_path: Any) -> None:
    db_path = str(tmp_path / "empty_col.anki2")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE col (id integer primary key, ver integer)")
    conn.commit()
    conn.close()
    logs, _ = load_anki_history(db_path)
    assert logs == {}


def test_load_anki_history_old_empty_col(tmp_path: Any) -> None:
    db_path = str(tmp_path / "old_empty_col.anki2")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE col (id integer primary key, ver integer, decks text, dconf text)"
    )
    conn.commit()
    conn.close()
    logs, _ = load_anki_history(db_path)
    assert logs == {}


def test_load_anki_history_no_ver(tmp_path: Any) -> None:
    db_path = str(tmp_path / "no_ver_collection.anki2")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE col (dummy integer)")
    conn.commit()
    conn.close()

    logs, date = load_anki_history(db_path)
    assert logs == {}
    assert date == START_DATE


def test_parse_retention_schedule_multi() -> None:
    sched = parse_retention_schedule("1:0.9,1:0.8")
    assert len(sched) == 2
    assert get_retention_for_day(0, sched) == 0.9
    assert get_retention_for_day(1, sched) == 0.8
    assert get_retention_for_day(2, sched) == 0.9


def test_parse_parameters_value_error() -> None:
    assert parse_parameters(
        "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,not_a_float"
    ) == tuple(DEFAULT_PARAMETERS)


def test_parse_parameters_valid() -> None:
    p_str = ",".join(["1.0"] * 21)
    p = parse_parameters(p_str)
    assert p == tuple([1.0] * 21)


def test_parse_parameters_wrong_length() -> None:
    assert parse_parameters("1.0,2.0") == tuple(DEFAULT_PARAMETERS)


def test_simulate_day_time_limit_at_start_of_iter() -> None:
    def time_estimator(card: Any, rating: Any, current_date: Any) -> float:
        return 100.0

    now = datetime.now(timezone.utc)
    seeded_data = SeededData(
        logs=defaultdict(list),
        last_rev=now - timedelta(days=1),
        true_cards={1: Card(card_id=1, due=now), 2: Card(card_id=2, due=now)},
        sys_cards={1: Card(card_id=1, due=now), 2: Card(card_id=2, due=now)},
    )
    config = SimulationConfig(
        n_days=1, time_limit=100.0, time_estimator=time_estimator, verbose=False
    )
    fitted, _, metrics = run_simulation(config, seeded_data=seeded_data)
    assert metrics["review_count"] == 1


def test_decode_varint_multi_byte() -> None:
    from src.anki_utils import decode_varint

    res, pos = decode_varint(bytes([0x80, 0x01]), 0)
    assert res == 128
    assert pos == 2


def test_run_simulation_triggers_patched_tqdm_more(monkeypatch: Any) -> None:
    import fsrs.optimizer

    def mock_compute_optimal(*args: Any, **kwargs: Any) -> Any:
        fsrs.optimizer.tqdm(total=100)
        return list(DEFAULT_PARAMETERS)

    monkeypatch.setattr(
        src.simulate_fsrs.RustOptimizer,
        "compute_optimal_parameters",
        mock_compute_optimal,
    )
    config = SimulationConfig(n_days=1, verbose=True)
    run_simulation(config)


def test_run_simulation_no_fitted_params(monkeypatch: Any) -> None:
    def mock_compute(*args: Any, **kwargs: Any) -> Any:
        raise Exception("Fitting failed")

    monkeypatch.setattr(
        src.simulate_fsrs.RustOptimizer, "compute_optimal_parameters", mock_compute
    )
    config = SimulationConfig(n_days=1, verbose=False)
    fitted, _, metrics = run_simulation(config)
    assert fitted is None
    assert metrics["stabilities"] == []
