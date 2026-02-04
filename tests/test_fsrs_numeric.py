from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fsrs import Card, Rating, Scheduler

from src.fsrs_engine import (
    init_difficulty,
    init_stability,
    predict_retrievability,
    update_state_forget,
    update_state_recall,
)

DEFAULT_PARAMS = (
    0.212,
    1.2931,
    2.3065,
    8.2956,
    6.4133,
    0.8334,
    3.0194,
    0.001,
    1.8722,
    0.1666,
    0.796,
    1.4835,
    0.0614,
    0.2629,
    1.6483,
    0.6014,
    1.8729,
    0.5425,
    0.0912,
    0.0658,
    0.1542,
)


@pytest.fixture  # type: ignore[misc]
def scheduler() -> Scheduler:
    return Scheduler(parameters=DEFAULT_PARAMS)


def test_numeric_consistency_retrievability(scheduler: Scheduler) -> None:
    stabilities = np.array([0.1, 1.0, 10.0, 100.0])
    elapsed_days = np.array([0, 1, 10, 100])

    # Reference results from fsrs library
    expected = []
    for s, d in zip(stabilities, elapsed_days, strict=False):
        card = Card(
            stability=float(s),
            last_review=datetime.now(timezone.utc) - timedelta(days=float(d)),
        )
        expected.append(
            scheduler.get_card_retrievability(card, datetime.now(timezone.utc))
        )

    actual = predict_retrievability(stabilities, elapsed_days, DEFAULT_PARAMS)

    np.testing.assert_allclose(actual, expected, rtol=1e-8)


def test_numeric_consistency_initial_state(scheduler: Scheduler) -> None:
    ratings = [Rating.Again, Rating.Hard, Rating.Good, Rating.Easy]

    for r in ratings:
        expected_s = scheduler._initial_stability(rating=r)
        expected_d = scheduler._initial_difficulty(rating=r, clamp=True)

        actual_s = init_stability(np.array([r]), DEFAULT_PARAMS)[0]
        actual_d = init_difficulty(np.array([r]), DEFAULT_PARAMS)[0]

        assert actual_s == pytest.approx(expected_s, rel=1e-9)
        assert actual_d == pytest.approx(expected_d, rel=1e-9)


def test_numeric_consistency_next_interval(scheduler: Scheduler) -> None:
    stabilities = np.array([0.1, 1.0, 10.0, 100.0])
    desired_retention = 0.9

    expected = [scheduler._next_interval(stability=float(s)) for s in stabilities]
    from src.fsrs_engine import next_interval

    actual = next_interval(stabilities, desired_retention, DEFAULT_PARAMS)

    np.testing.assert_array_equal(actual, expected)


def test_numeric_consistency_recall_updates(scheduler: Scheduler) -> None:
    # Test cases: (stability, difficulty, rating, elapsed_days)
    test_cases = [
        (1.0, 5.0, Rating.Hard, 1),
        (1.0, 5.0, Rating.Good, 1),
        (1.0, 5.0, Rating.Easy, 1),
        (10.0, 3.0, Rating.Good, 5),
        (100.0, 2.0, Rating.Easy, 50),
    ]

    for s, d, r, elap in test_cases:
        last_rev = datetime.now(timezone.utc) - timedelta(days=elap)
        now = datetime.now(timezone.utc)
        card = Card(stability=s, difficulty=d, last_review=last_rev)

        ret = scheduler.get_card_retrievability(card, now)
        # We manually call the internal recall stability update to isolate it
        expected_s = scheduler._next_recall_stability(
            difficulty=d, stability=s, retrievability=ret, rating=r
        )
        expected_d = scheduler._next_difficulty(difficulty=d, rating=r)

        actual_s, actual_d = update_state_recall(
            np.array([s]), np.array([d]), np.array([r]), np.array([ret]), DEFAULT_PARAMS
        )

        assert actual_s[0] == pytest.approx(expected_s, rel=1e-7)
        assert actual_d[0] == pytest.approx(expected_d, rel=1e-7)


def test_numeric_consistency_forget_updates(scheduler: Scheduler) -> None:
    # Test cases: (stability, difficulty, rating, elapsed_days)
    test_cases = [
        (1.0, 5.0, 1),
        (10.0, 3.0, 5),
        (100.0, 2.0, 50),
    ]

    for s, d, elap in test_cases:
        last_rev = datetime.now(timezone.utc) - timedelta(days=elap)
        now = datetime.now(timezone.utc)
        card = Card(stability=s, difficulty=d, last_review=last_rev)

        ret = scheduler.get_card_retrievability(card, now)
        expected_s = scheduler._next_forget_stability(
            difficulty=d, stability=s, retrievability=ret
        )
        expected_d = scheduler._next_difficulty(difficulty=d, rating=Rating.Again)

        actual_s, actual_d = update_state_forget(
            np.array([s]), np.array([d]), np.array([ret]), DEFAULT_PARAMS
        )

        assert actual_s[0] == pytest.approx(expected_s, rel=1e-7)
        assert actual_d[0] == pytest.approx(expected_d, rel=1e-7)
