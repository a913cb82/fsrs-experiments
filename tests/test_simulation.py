from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from src.simulate_fsrs import (
    DEFAULT_PARAMETERS,
    Card,
    Rating,
    ReviewLog,
    RustOptimizer,
    run_simulation,
    run_simulation_cli,
)


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


def test_run_simulation_basic() -> None:
    fitted, gt, metrics = run_simulation(n_days=2, review_limit=10, verbose=False)
    assert gt == DEFAULT_PARAMETERS
    assert fitted is not None
    assert len(fitted) == 21
    assert metrics["review_count"] > 0
    assert metrics["card_count"] > 0


def test_run_simulation_with_seeded_data() -> None:
    # Pre-calculate seeded data payload
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

    seeded_payload = {
        "logs": logs,
        "last_rev": now - timedelta(days=1),
        "true_cards": true_cards,
        "sys_cards": sys_cards,
    }

    fitted, _, metrics = run_simulation(
        n_days=2, review_limit=5, seeded_data=seeded_payload, verbose=False
    )
    assert metrics["card_count"] >= 1
    assert metrics["review_count"] > 1


def test_run_simulation_with_burn_in() -> None:
    fitted, _, _ = run_simulation(
        n_days=10, burn_in_days=5, review_limit=100, verbose=False
    )
    assert fitted is not None


def test_run_simulation_with_burn_in_triggered() -> None:
    # We need 512 reviews to trigger optimizer
    fitted, _, metrics = run_simulation(
        n_days=10, burn_in_days=5, review_limit=150, verbose=False
    )
    assert fitted is not None
    assert metrics["review_count"] >= 512


def test_simulate_day_again_branch() -> None:
    # Set retention very low to force 'Again' ratings
    _, _, metrics = run_simulation(
        n_days=1, review_limit=10, retention="0.01", verbose=False
    )
    assert metrics["review_count"] > 0


def test_simulate_day_review_limit_break() -> None:
    test_db = "tests/test_collection.anki2"
    # To hit the break, we need reviews_done >= review_limit.
    _, _, metrics = run_simulation(
        n_days=100, review_limit=1, seed_history=test_db, verbose=False
    )
    # Total reviews = 5 (seed) + 100 (sim) = 105
    assert metrics["review_count"] == 105


def test_run_simulation_no_optimizer_triggered() -> None:
    # Less than 512 reviews
    fitted, _, _ = run_simulation(n_days=1, review_limit=10, verbose=False)
    assert fitted is not None


def test_parse_retention_schedule_error() -> None:
    from src.simulate_fsrs import parse_retention_schedule

    # This should trigger the ValueError catch
    assert parse_retention_schedule("invalid:schedule") == [(1, 0.9)]
    assert parse_retention_schedule("invalid") == [(1, 0.9)]


def test_load_anki_history_missing_file() -> None:
    from src.simulate_fsrs import START_DATE, load_anki_history

    logs, date = load_anki_history("non_existent.anki2")
    assert logs == {}
    assert date == START_DATE


def test_run_simulation_cli_basic(monkeypatch: Any) -> None:
    import sys

    monkeypatch.setattr(
        sys, "argv", ["src/simulate_fsrs.py", "--days", "1", "--reviews", "10"]
    )
    run_simulation_cli()


def test_run_simulation_cli_with_history(monkeypatch: Any) -> None:
    import sys

    import src.simulate_fsrs

    test_db = "tests/test_collection.anki2"
    args = [
        "src/simulate_fsrs.py",
        "--days",
        "1",
        "--reviews",
        "10",
        "--seed-history",
        test_db,
    ]
    monkeypatch.setattr(sys, "argv", args)

    # Mock pre-fitting to trigger the > 512 branch
    logs: dict[int, list[ReviewLog]] = defaultdict(list)
    for _ in range(600):
        log = ReviewLog(
            card_id=1,
            rating=Rating.Good,
            review_datetime=datetime.now(timezone.utc),
            review_duration=None,
        )
        logs[1].append(log)

    monkeypatch.setattr(
        src.simulate_fsrs,
        "load_anki_history",
        lambda *a: (logs, datetime.now(timezone.utc)),
    )

    run_simulation_cli()
