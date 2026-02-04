from typing import Any

import numpy as np

from src.simulate_fsrs import run_simulation
from src.simulation_config import SimulationConfig


def test_sanity_review_and_new_limit() -> None:
    """
    Sanity Check 1: Review and New Card Limits
    Verifies that the simulator respects the daily review and new card limits.
    """
    # Run for 1 day to easily check limits
    review_limit = 5
    new_limit = 2
    config = SimulationConfig(
        n_days=1,
        review_limit=review_limit,
        new_limit=new_limit,
        verbose=False,
        return_logs=True,
    )

    # We start with no seed data, so all reviews will be new cards.
    # The simulator prioritizes reviews over new cards.
    # Since there are no existing cards, it should just learn 'new_limit' cards.
    # Wait, if review_limit < new_limit, it is capped by review_limit?
    # In simulate_day_numpy:
    #   remaining_total = review_limit - reviews_done
    #   remaining_new = new_limit - new_done
    #   batch_size = min(new_batch_size, remaining_total, remaining_new)
    # So yes, it respects both.

    _, _, metrics = run_simulation(config)
    logs = metrics["logs"]

    # Total reviews should be equal to new_limit (since we have no due reviews)
    # But it must also not exceed review_limit.
    assert len(logs) == new_limit
    assert len(logs) <= review_limit

    # Now let's try a case where review_limit constrains new cards
    review_limit_strict = 1
    new_limit_loose = 10
    config_strict = SimulationConfig(
        n_days=1,
        review_limit=review_limit_strict,
        new_limit=new_limit_loose,
        verbose=False,
        return_logs=True,
    )
    _, _, metrics_strict = run_simulation(config_strict)
    logs_strict = metrics_strict["logs"]

    assert len(logs_strict) == review_limit_strict


def test_sanity_time_limit() -> None:
    """
    Sanity Check 2: Time Limit
    Verifies that the simulator stops reviewing when the time limit is reached.
    """
    fixed_duration = 10.0  # seconds
    time_limit = 55.0  # seconds. Should allow 6 reviews (checked after batch?)
    # The batch size for new cards is default 50.
    # Logic:
    # batch_completed = searchsorted(
    #     cumulative_times + time_accumulated, time_limit, side="right"
    # )
    # If cumulative times are [10, 20, 30, 40, 50, 60...], and limit is 55.
    # searchsorted(..., 55, side="right") will return index 5 (which corresponds to 50).
    # Wait:
    # cumsum: [10, 20, 30, 40, 50, 60]
    # searchsorted([10...60], 55, 'right') -> index 5.
    # index 5 means 5 items are taken (0,1,2,3,4).
    # So it stops BEFORE exceeding the limit?
    # Let's verify.

    def constant_time_estimator(
        deck: Any, indices: Any, date: Any, params: Any, ratings: Any
    ) -> Any:
        return np.full(len(indices), fixed_duration, dtype=np.float32)

    config = SimulationConfig(
        n_days=1,
        time_limit=time_limit,
        time_estimator=constant_time_estimator,
        new_limit=100,  # plenty of new cards
        verbose=False,
        return_logs=True,
    )

    _, _, metrics = run_simulation(config)
    logs = metrics["logs"]

    # We expect 5 reviews if strict, or maybe 6 if loose.
    # Based on searchsorted logic reading:
    # it seems it finds the insertion point to maintain order.
    # if array is [10, 20, 30, 40, 50, 60], 55 goes after 50. Index is 5.
    # slicing [:5] gives 5 elements. Total time 50.
    # So it strictly respects the limit (does not exceed).

    assert len(logs) == 5
    total_time = np.sum(logs.review_durations)
    assert total_time <= time_limit
    assert total_time == 50.0


def test_sanity_retention_targets() -> None:
    """
    Sanity Check 3: Retention Targets
    Verifies that higher retention targets lead to more reviews over a long period.
    """
    # We need a longer period to see the effect of retention on review counts
    # because with no history, initial reviews are just new cards.
    # We rely on re-reviews.

    n_days = 30
    new_limit = 5  # Introduce a few cards per day

    # Low retention: 0.7
    config_low = SimulationConfig(
        n_days=n_days,
        retention="0.7",
        new_limit=new_limit,
        review_limit=1000,  # unlimited effectively
        verbose=False,
    )
    _, _, metrics_low = run_simulation(config_low)

    # High retention: 0.95
    config_high = SimulationConfig(
        n_days=n_days,
        retention="0.95",
        new_limit=new_limit,
        review_limit=1000,
        verbose=False,
    )
    _, _, metrics_high = run_simulation(config_high)

    # High retention should produce more reviews because intervals are shorter
    print(f"Low retention reviews: {metrics_low['review_count']}")
    print(f"High retention reviews: {metrics_high['review_count']}")

    assert metrics_high["review_count"] > metrics_low["review_count"]

    # Also check total retention (sum of retrievabilities)
    # Higher target retention should maintain the deck at a higher state
    assert metrics_high["total_retention"] > metrics_low["total_retention"]


def test_sanity_custom_estimators() -> None:
    """
    Sanity Check 4: Custom Estimators
    Verifies that custom rating and time estimators are correctly used.
    """

    # Custom rating estimator: Always "Again" (1)
    def rating_estimator_again(deck: Any, indices: Any, date: Any, params: Any) -> Any:
        return np.full(len(indices), 1, dtype=np.int8)

    # Custom time estimator: Always 12.34 seconds
    def time_estimator_fixed(
        deck: Any, indices: Any, date: Any, params: Any, ratings: Any
    ) -> Any:
        return np.full(len(indices), 12.34, dtype=np.float32)

    config = SimulationConfig(
        n_days=1,
        new_limit=10,
        rating_estimator=rating_estimator_again,
        time_estimator=time_estimator_fixed,
        verbose=False,
        return_logs=True,
    )

    _, _, metrics = run_simulation(config)
    logs = metrics["logs"]

    assert len(logs) > 0
    assert np.all(logs.ratings == 1)
    assert np.all(np.isclose(logs.review_durations, 12.34))


def test_sanity_seed_determinism() -> None:
    """
    Sanity Check 5: Determinism with Seed
    Verifies that running the simulation twice with the same seed
    produces identical results.
    """
    config1 = SimulationConfig(n_days=5, seed=123, verbose=False, return_logs=True)
    _, _, metrics1 = run_simulation(config1)

    config2 = SimulationConfig(n_days=5, seed=123, verbose=False, return_logs=True)
    _, _, metrics2 = run_simulation(config2)

    assert metrics1["review_count"] == metrics2["review_count"]
    assert np.array_equal(metrics1["logs"].ratings, metrics2["logs"].ratings)

    # Different seed
    config3 = SimulationConfig(n_days=5, seed=456, verbose=False, return_logs=True)
    _, _, metrics3 = run_simulation(config3)

    # It's statistically very likely they are different given random choices
    # (ratings, etc are probabilistic by default estimator)
    # But strictly speaking, if we rely on random choices, they should differ.
    # However, default estimator uses random choices.
    assert not np.array_equal(metrics1["logs"].ratings, metrics3["logs"].ratings)


def test_sanity_no_mutation() -> None:
    """
    Sanity Check 6: No Mutation of Seeded Data
    Verifies that run_simulation does not mutate the seeded_data objects.
    """
    from src.anki_utils import START_DATE
    from src.deck import Deck
    from src.simulation_config import LogData, SeededData

    # Create minimal seeded data
    cids = np.array([1], dtype=np.int64)
    stabs = np.array([1.0], dtype=np.float64)
    diffs = np.array([5.0], dtype=np.float64)
    dues = np.array([START_DATE], dtype="datetime64[ns]")
    lrs = np.array([START_DATE], dtype="datetime64[ns]")

    true_deck = Deck(cids.copy(), stabs.copy(), diffs.copy(), dues.copy(), lrs.copy())
    sys_deck = Deck(cids.copy(), stabs.copy(), diffs.copy(), dues.copy(), lrs.copy())
    logs = LogData(
        card_ids=cids.copy(),
        ratings=np.array([3], dtype=np.int8),
        review_timestamps=lrs.copy(),
        review_durations=np.array([10.0], dtype=np.float32),
    )

    seeded_data = SeededData(
        logs=logs, last_rev=START_DATE, true_cards=true_deck, sys_cards=sys_deck
    )

    initial_card_count = len(seeded_data.true_cards)
    initial_log_count = len(seeded_data.logs)

    config = SimulationConfig(
        n_days=1,
        new_limit=10,
        review_limit=100,
        verbose=False,
        compute_final_params=False,
    )

    run_simulation(config, seeded_data=seeded_data)

    # Check that seeded_data remains unchanged
    assert len(seeded_data.true_cards) == initial_card_count
    assert len(seeded_data.logs) == initial_log_count
    assert np.array_equal(seeded_data.true_cards.card_ids[:initial_card_count], cids)


def test_sanity_retention_peak() -> None:
    """
    Sanity Check 7: Retention Peak under Workload Constraint
    Verifies that with a fixed time budget, total recall peaks at an intermediate
    retention target rather than strictly increasing up to 0.99.
    """
    from src.anki_utils import DEFAULT_PROB_FIRST_AGAIN

    n_days = 60  # Sufficient time to see effects
    fixed_duration = 10.0
    time_limit_sec = 10 * 60  # 10 minutes per day

    def constant_time_estimator(
        deck: Any, indices: Any, date: Any, params: Any, ratings: Any
    ) -> Any:
        return np.full(len(indices), fixed_duration, dtype=np.float32)

    retentions = [0.7, 0.8, 0.9, 0.99]
    results = []

    for ret in retentions:
        config = SimulationConfig(
            n_days=n_days,
            retention=str(ret),
            review_limit=None,
            new_limit=None,  # Unlimited new cards available
            time_limit=time_limit_sec,
            time_estimator=constant_time_estimator,
            verbose=False,
            compute_final_params=False,
            return_logs=False,
            seed=42,
        )
        _, _, metrics = run_simulation(config)
        results.append(metrics)

    # Calculate Adjusted Retention
    max_cards = max(m["card_count"] for m in results)
    r_baseline = 1.0 - DEFAULT_PROB_FIRST_AGAIN

    adj_totals = []
    for m in results:
        padding = max(0, max_cards - m["card_count"]) * r_baseline
        adj_totals.append(m["total_retention"] + padding)

    # We expect 0.99 NOT to be the peak.
    peak_idx = np.argmax(adj_totals)
    assert retentions[peak_idx] < 0.99
    assert adj_totals[len(retentions) - 1] < adj_totals[peak_idx]
