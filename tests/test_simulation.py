from datetime import datetime, timedelta, timezone

from src.simulate_fsrs import (
    DEFAULT_PARAMETERS,
    Rating,
    ReviewLog,
    RustOptimizer,
    run_simulation,
)


def test_rust_optimizer() -> None:
    # Create some dummy logs
    now = datetime.now(timezone.utc)
    # We need delta_t > 0 for Rust optimizer, so let's adjust dates
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
    # Check that it returned something reasonable (not just zeros)
    assert any(p != 0 for p in params)


def test_run_simulation_basic() -> None:
    # Run a very short simulation
    fitted, gt, metrics = run_simulation(n_days=2, review_limit=10, verbose=False)

    assert gt == DEFAULT_PARAMETERS
    # With only 2 days and default reviews, it might return defaults or very close
    assert fitted is not None
    assert len(fitted) == 21
    assert metrics["review_count"] > 0
    assert metrics["card_count"] > 0
    assert "stabilities" in metrics


def test_run_simulation_retention_schedule() -> None:
    # Run with a schedule
    fitted, _gt, _metrics = run_simulation(
        n_days=5, review_limit=10, retention="2:0.7,1:0.9", verbose=False
    )
    assert fitted is not None
    assert len(fitted) == 21


def test_run_simulation_with_seed_history() -> None:
    # Test seeding from our test DB
    test_db = "tests/test_collection.anki2"
    _fitted, _gt, metrics = run_simulation(
        n_days=5, review_limit=10, seed_history=test_db, verbose=False
    )
    # Cards from test DB should be present
    assert metrics["card_count"] >= 2
    assert metrics["review_count"] >= 5  # 5 from DB + simulation reviews
