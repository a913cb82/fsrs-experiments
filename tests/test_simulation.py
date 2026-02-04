from datetime import timedelta
from typing import Any

import numpy as np

from src.anki_utils import (
    START_DATE,
    get_review_history_stats,
    load_anki_history,
)
from src.deck import Deck
from src.optimizer import RustOptimizer
from src.proto_utils import get_deck_config_id, get_field_from_proto
from src.simulate_fsrs import (
    run_simulation,
)
from src.simulation_config import LogData, SeededData, SimulationConfig
from src.utils import (
    DEFAULT_PARAMETERS,
    get_retention_for_day,
    parse_parameters,
    parse_retention_schedule,
)


def test_rust_optimizer_from_arrays() -> None:
    # 5 items for card 1
    card_ids = np.array([1, 1, 1, 1, 1], dtype=np.int64)
    ratings = np.array([3, 3, 3, 3, 3], dtype=np.int8)
    days = np.array([0, 1, 5, 10, 20], dtype=np.int32)

    opt = RustOptimizer(card_ids, ratings, days)
    params = opt.compute_optimal_parameters()
    assert params is not None
    assert len(params) == 21
    assert any(p != 0 for p in params)


def test_rust_optimizer_no_items() -> None:
    card_ids = np.array([1], dtype=np.int64)
    ratings = np.array([3], dtype=np.int8)
    days = np.array([0], dtype=np.int32)

    opt = RustOptimizer(card_ids, ratings, days)
    params = opt.compute_optimal_parameters()
    assert params is None


def test_run_simulation_basic() -> None:
    config = SimulationConfig(n_days=2, review_limit=10, verbose=False)
    fitted, gt, metrics = run_simulation(config)
    assert gt == DEFAULT_PARAMETERS
    assert fitted is not None
    assert len(fitted) == 21
    assert metrics["review_count"] > 0
    assert metrics["card_count"] > 0


def test_run_simulation_with_seeded_data() -> None:
    logs = LogData(
        card_ids=np.array([1], dtype=np.int64),
        ratings=np.array([3], dtype=np.int8),
        review_timestamps=np.array([START_DATE], dtype="datetime64[ns]"),
        review_durations=np.array([5.0], dtype=np.float32),
    )

    cids = np.array([1], dtype=np.int64)
    stabs = np.array([0.212], dtype=np.float64)
    diffs = np.array([5.0], dtype=np.float64)
    dues = np.array([START_DATE], dtype="datetime64[ns]")
    lrs = np.array([np.datetime64("NaT")], dtype="datetime64[ns]")

    true_deck = Deck(
        cids,
        stabs,
        diffs,
        np.full(1, np.datetime64("NaT"), dtype="datetime64[ns]"),
        lrs,
    )
    sys_deck = Deck(cids, stabs, diffs, dues, lrs)

    seeded_payload = SeededData(
        logs=logs,
        last_rev=START_DATE,
        true_cards=true_deck,
        sys_cards=sys_deck,
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
    # 5 initial + 100 simulated
    assert metrics["review_count"] == 105


def test_run_simulation_no_optimizer_triggered() -> None:
    config = SimulationConfig(n_days=1, review_limit=10, verbose=False)
    fitted, _, _ = run_simulation(config)
    assert fitted is None


def test_parse_retention_schedule_error() -> None:
    assert parse_retention_schedule("invalid:schedule") == [(0, 0.9)]
    assert parse_retention_schedule("invalid") == [(0, 0.9)]


def test_load_anki_history_missing_file() -> None:
    logs, date = load_anki_history("non_existent.anki2")
    assert len(logs.card_ids) == 0
    assert date == START_DATE


def test_run_simulation_with_seeded_data_no_weights() -> None:
    config = SimulationConfig(n_days=5, review_limit=10, verbose=False)
    fitted, _, metrics = run_simulation(
        config,
        seeded_data=SeededData(
            last_rev=START_DATE,
            true_cards=Deck(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype="datetime64[ns]"),
                np.array([], dtype="datetime64[ns]"),
            ),
            sys_cards=Deck(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype="datetime64[ns]"),
                np.array([], dtype="datetime64[ns]"),
            ),
            logs=LogData.concatenate([]),
        ),
    )
    assert fitted is not None
    assert metrics["review_count"] > 0


def test_run_simulation_with_rating_estimator() -> None:
    def rating_estimator(deck: Any, indices: Any, date: Any, params: Any) -> Any:
        return np.full(len(indices), 4, dtype=np.int8)  # Always 'Easy'

    config = SimulationConfig(
        n_days=5,
        review_limit=20,
        new_limit=10,
        rating_estimator=rating_estimator,
        verbose=False,
    )
    _, _, metrics = run_simulation(config)

    assert metrics["review_count"] > 0
    # LogData object
    for rating in metrics["logs"].ratings:
        assert int(rating) == 4  # Easy


def test_run_simulation_with_time_estimator() -> None:
    def time_estimator(
        deck: Any, indices: Any, date: Any, params: Any, ratings: Any
    ) -> Any:
        return np.full(len(indices), 5.0, dtype=np.float32)  # Always 5 seconds

    config = SimulationConfig(
        n_days=5,
        review_limit=20,
        new_limit=10,
        time_estimator=time_estimator,
        verbose=False,
    )
    _, _, metrics = run_simulation(config)

    assert metrics["review_count"] > 0

    config_with_limit = SimulationConfig(
        n_days=1,
        time_limit=11.0,
        time_estimator=time_estimator,
        verbose=False,
    )
    _, _, metrics_limit = run_simulation(config_with_limit)
    assert metrics_limit["review_count"] == 2


def test_get_review_history_stats() -> None:
    logs = LogData(
        card_ids=np.array([1, 1], dtype=np.int64),
        ratings=np.array([3, 3], dtype=np.int8),
        review_timestamps=np.array(
            [START_DATE, START_DATE + timedelta(days=5)], dtype="datetime64[ns]"
        ),
        review_durations=np.array([5.0, 10.0], dtype=np.float32),
    )
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
    assert get_field_from_proto(normal_deck_blob, 1) == 100  # Verify basic proto
    assert get_deck_config_id(b"", kind_blob) == 100


def test_parse_retention_schedule_multi() -> None:
    sched = parse_retention_schedule("0:0.9,1:0.8")
    assert len(sched) == 2
    assert get_retention_for_day(0, sched) == 0.9
    assert get_retention_for_day(1, sched) == 0.8
    assert get_retention_for_day(2, sched) == 0.8


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
    def time_estimator(
        deck: Any, indices: Any, date: Any, params: Any, ratings: Any
    ) -> Any:
        return np.full(len(indices), 100.0, dtype=np.float32)

    seeded_data = SeededData(
        logs=LogData.concatenate([]),
        last_rev=START_DATE,
        true_cards=Deck(
            np.array([1, 2], dtype=np.int64),
            np.array([0.2, 0.2]),
            np.array([5.0, 5.0]),
            np.full(2, np.datetime64("NaT"), dtype="datetime64[ns]"),
            np.full(2, np.datetime64("NaT"), dtype="datetime64[ns]"),
        ),
        sys_cards=Deck(
            np.array([1, 2], dtype=np.int64),
            np.array([0.2, 0.2]),
            np.array([5.0, 5.0]),
            np.full(2, np.datetime64("NaT"), dtype="datetime64[ns]"),
            np.full(2, np.datetime64("NaT"), dtype="datetime64[ns]"),
        ),
    )
    config = SimulationConfig(
        n_days=1,
        time_limit=100.0,
        time_estimator=time_estimator,
        verbose=False,
    )
    fitted, _, metrics = run_simulation(config, seeded_data=seeded_data)
    assert metrics["review_count"] == 1


def test_decode_varint_multi_byte() -> None:
    from src.proto_utils import decode_varint

    res, pos = decode_varint(bytes([0x80, 0x01]), 0)
    assert res == 128
    assert pos == 2


def test_run_simulation_triggers_patched_tqdm_more(monkeypatch: Any) -> None:
    import fsrs.optimizer

    def mock_compute_optimal(*args: Any, **kwargs: Any) -> Any:
        fsrs.optimizer.tqdm(total=100)
        return list(DEFAULT_PARAMETERS)

    monkeypatch.setattr(
        RustOptimizer,
        "compute_optimal_parameters",
        mock_compute_optimal,
    )
    config = SimulationConfig(n_days=1, verbose=True)
    run_simulation(config)


def test_run_simulation_no_fitted_params(monkeypatch: Any) -> None:
    def mock_compute(*args: Any, **kwargs: Any) -> Any:
        return None

    monkeypatch.setattr(RustOptimizer, "compute_optimal_parameters", mock_compute)
    config = SimulationConfig(n_days=1, verbose=False)
    fitted, _, metrics = run_simulation(config)
    assert fitted is None
    s_nat, s_alg = metrics["stabilities"]
    assert len(s_nat) > 0
    assert len(s_alg) > 0
