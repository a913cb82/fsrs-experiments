import numpy as np
import pandas as pd
import pytest
from fsrs import Rating

from src.anki_utils import (
    RatingWeights,
    calculate_expected_d0,
    infer_review_weights,
)
from src.utils import (
    calculate_metrics,
    calculate_population_retrievability,
    get_retention_for_day,
    parse_parameters,
    parse_retention_schedule,
)


def test_calculate_population_retrievability() -> None:
    t = np.linspace(0, 10, 11)
    stabilities = np.array([1.0, 5.0, 10.0])
    params = [0.1] * 21

    res = calculate_population_retrievability(t, stabilities, params)
    assert len(res) == 11
    assert res[0] == 1.0
    assert np.all(res <= 1.0)
    assert np.all(res >= 0.0)

    assert np.all(calculate_population_retrievability(t, np.array([]), params) == 1.0)


def test_calculate_metrics() -> None:
    gt_params = [0.1] * 21
    fit_params = [0.1] * 21
    s_nat = np.array([1.0, 5.0])
    s_alg = np.array([1.0, 5.0])

    rmse, kl = calculate_metrics(gt_params, fit_params, s_nat, s_alg)
    assert rmse == 0.0
    assert kl == 0.0

    fit_params_diff = list(gt_params)
    fit_params_diff[20] = 0.5
    rmse, kl = calculate_metrics(gt_params, fit_params_diff, s_nat, s_alg)
    assert rmse > 0
    assert kl > 0

    assert calculate_metrics(gt_params, fit_params, np.array([]), np.array([])) == (
        0.0,
        0.0,
    )


def test_parse_retention_schedule() -> None:
    assert parse_retention_schedule("0.9") == [(1, 0.9)]
    assert parse_retention_schedule("1:0.9,2:0.8") == [(1, 0.9), (2, 0.8)]


def test_get_retention_for_day() -> None:
    schedule = [(1, 0.9), (2, 0.8)]
    assert get_retention_for_day(0, schedule) == 0.9
    assert get_retention_for_day(1, schedule) == 0.8
    assert get_retention_for_day(2, schedule) == 0.8
    assert get_retention_for_day(3, schedule) == 0.9


def test_parse_parameters() -> None:
    p_str = ",".join(["0.1"] * 21)
    res = parse_parameters(p_str)
    assert isinstance(res, tuple)
    assert len(res) == 21
    assert res[0] == 0.1


def test_calculate_expected_d0() -> None:
    from fsrs.scheduler import DEFAULT_PARAMETERS

    weights = [0.5, 0.1, 0.3, 0.1]
    res = calculate_expected_d0(weights, DEFAULT_PARAMETERS)
    assert isinstance(res, float)
    assert res > 0


def test_infer_review_weights() -> None:
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    # Card 1: 1st Good, 2nd Good
    # Card 2: 1st Again, 2nd Hard, 3rd Easy
    data = [
        {
            "card_id": 1,
            "rating": int(Rating.Good),
            "review_datetime": now - timedelta(days=10),
        },
        {
            "card_id": 1,
            "rating": int(Rating.Good),
            "review_datetime": now - timedelta(days=5),
        },
        {
            "card_id": 2,
            "rating": int(Rating.Again),
            "review_datetime": now - timedelta(days=20),
        },
        {
            "card_id": 2,
            "rating": int(Rating.Hard),
            "review_datetime": now - timedelta(days=15),
        },
        {
            "card_id": 2,
            "rating": int(Rating.Easy),
            "review_datetime": now - timedelta(days=10),
        },
    ]
    df = pd.DataFrame(data)

    weights = infer_review_weights(df)
    assert isinstance(weights, RatingWeights)
    assert len(weights.first) == 4
    assert len(weights.success) == 3
    # Check probabilities sum to 1
    assert pytest.approx(sum(weights.first)) == 1.0
    assert pytest.approx(sum(weights.success)) == 1.0
