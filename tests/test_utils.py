import numpy as np
import pytest

from src.anki_utils import (
    DEFAULT_PROB_EASY,
    DEFAULT_PROB_FIRST_AGAIN,
    DEFAULT_PROB_FIRST_EASY,
    DEFAULT_PROB_FIRST_GOOD,
    DEFAULT_PROB_FIRST_HARD,
    DEFAULT_PROB_GOOD,
    DEFAULT_PROB_HARD,
    calculate_expected_d0,
    infer_review_weights,
)
from src.simulation_config import LogData
from src.utils import (
    DEFAULT_PARAMETERS,
    calculate_metrics,
    get_retention_for_day,
    parse_parameters,
    parse_retention_schedule,
)


def test_calculate_expected_d0() -> None:
    weights = [0.25, 0.25, 0.25, 0.25]
    d0 = calculate_expected_d0(weights, DEFAULT_PARAMETERS)
    assert isinstance(d0, float)
    assert 1.0 <= d0 <= 10.0


def test_calculate_metrics() -> None:
    gt = DEFAULT_PARAMETERS
    fitted = DEFAULT_PARAMETERS
    s_nat = np.array([1.0, 5.0, 10.0])
    s_alg = np.array([1.0, 5.0, 10.0])
    rmse, kl = calculate_metrics(gt, fitted, s_nat, s_alg)
    assert rmse == pytest.approx(0.0)
    assert kl == pytest.approx(0.0)


def test_calculate_metrics_different() -> None:
    gt = DEFAULT_PARAMETERS
    fitted = list(DEFAULT_PARAMETERS)
    fitted[0] += 0.1
    s_nat = np.array([1.0, 5.0, 10.0])
    s_alg = np.array([1.1, 5.5, 11.0])
    rmse, kl = calculate_metrics(gt, tuple(fitted), s_nat, s_alg)
    assert rmse > 0
    assert kl > 0


def test_get_retention_for_day() -> None:
    schedule = [(10, 0.9), (20, 0.85)]
    # Default is 0.9 if day < 10
    assert get_retention_for_day(0, schedule) == 0.9
    assert get_retention_for_day(5, schedule) == 0.9
    assert get_retention_for_day(10, schedule) == 0.9
    assert get_retention_for_day(11, schedule) == 0.9
    assert get_retention_for_day(20, schedule) == 0.85
    assert get_retention_for_day(25, schedule) == 0.85


def test_parse_retention_schedule_single() -> None:
    assert parse_retention_schedule("0.95") == [(0, 0.95)]


def test_parse_retention_schedule_complex() -> None:
    res = parse_retention_schedule("10:0.9,20:0.8")
    assert res == [(10, 0.9), (20, 0.8)]


def test_parse_parameters_too_short() -> None:
    p_str = "1.0,2.0"
    params = parse_parameters(p_str)
    assert params == DEFAULT_PARAMETERS


def test_infer_review_weights_empty() -> None:
    empty_logs = LogData(
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int8),
        np.array([], dtype="datetime64[ns]"),
        np.array([], dtype=np.float32),
    )
    weights = infer_review_weights(empty_logs)
    assert weights.first == [
        DEFAULT_PROB_FIRST_AGAIN,
        DEFAULT_PROB_FIRST_HARD,
        DEFAULT_PROB_FIRST_GOOD,
        DEFAULT_PROB_FIRST_EASY,
    ]
    assert weights.success == [
        DEFAULT_PROB_HARD,
        DEFAULT_PROB_GOOD,
        DEFAULT_PROB_EASY,
    ]


def test_infer_review_weights() -> None:
    now = np.datetime64("2026-01-01")
    logs = LogData(
        card_ids=np.array([1, 1, 2, 2, 2], dtype=np.int64),
        ratings=np.array([3, 3, 1, 2, 4], dtype=np.int8),
        review_timestamps=np.array(
            [
                now - np.timedelta64(10, "D"),
                now - np.timedelta64(5, "D"),
                now - np.timedelta64(20, "D"),
                now - np.timedelta64(15, "D"),
                now - np.timedelta64(10, "D"),
            ],
            dtype="datetime64[ns]",
        ),
        review_durations=np.zeros(5, dtype=np.float32),
    )

    weights = infer_review_weights(logs)
    assert weights.first == [0.5, 0.0, 0.5, 0.0]
    assert weights.success == [
        pytest.approx(1 / 3),
        pytest.approx(1 / 3),
        pytest.approx(1 / 3),
    ]
