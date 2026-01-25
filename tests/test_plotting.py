import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import numpy as np

from src.plot_fsrs_divergence import (
    calculate_metrics,
    calculate_population_retrievability,
    init_worker,
    main,
    plot_forgetting_curves,
    run_single_task,
)
from src.simulate_fsrs import DEFAULT_PARAMETERS, Rating, ReviewLog


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
    stabilities = [(1.0, 1.0), (5.0, 5.0)]

    rmse, kl = calculate_metrics(gt_params, fit_params, stabilities)
    assert rmse == 0.0
    assert kl == 0.0

    fit_params_diff = list(gt_params)
    fit_params_diff[20] = 0.5
    rmse, kl = calculate_metrics(gt_params, fit_params_diff, stabilities)
    assert rmse > 0
    assert kl > 0

    assert calculate_metrics(gt_params, fit_params, []) == (0.0, 0.0)


def test_init_worker() -> None:
    now = datetime.now(timezone.utc)
    payload = {
        "logs": defaultdict(list),
        "last_rev": now - timedelta(days=1),
        "true_cards": {},
        "sys_cards": {},
    }
    init_worker(payload)
    import src.plot_fsrs_divergence

    assert src.plot_fsrs_divergence._worker_seeded_data == payload


def test_run_single_task() -> None:
    task = {
        "days": 1,
        "reviews": 10,
        "retention": "0.9",
        "burn_in": 0,
        "seed": 42,
        "config_key": (0, 1, 10, "0.9"),
    }
    res = run_single_task(task)
    assert res["success"]
    assert res["config_key"] == task["config_key"]
    assert "fitted" in res
    assert "gt" in res
    assert "stabilities" in res


def test_plot_forgetting_curves() -> None:
    if os.path.exists("forgetting_curve_divergence.png"):
        os.remove("forgetting_curve_divergence.png")

    results = [
        {
            "label": "Test",
            "r_fit_avg": np.ones(200) * 0.9,
            "r_fit_std": np.ones(200) * 0.05,
            "r_nat_avg": np.ones(200) * 0.85,
            "rmse": 0.05,
            "kl": 0.01,
        }
    ]

    plot_forgetting_curves(results)
    assert os.path.exists("forgetting_curve_divergence.png")


def test_main_mocked(monkeypatch: Any) -> None:
    import sys

    import src.plot_fsrs_divergence

    args = [
        "src/plot_fsrs_divergence.py",
        "--days",
        "1",
        "--repeats",
        "1",
        "--concurrency",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", args)

    monkeypatch.setattr(
        src.plot_fsrs_divergence,
        "load_anki_history",
        lambda *a, **k: ({}, datetime.now()),
    )
    monkeypatch.setattr(
        src.plot_fsrs_divergence,
        "run_simulation",
        lambda *a, **k: (list(DEFAULT_PARAMETERS), list(DEFAULT_PARAMETERS), {}),
    )
    monkeypatch.setattr(
        src.plot_fsrs_divergence, "plot_forgetting_curves", lambda *a: None
    )

    main()


def test_main_with_seeded_history(monkeypatch: Any) -> None:
    import sys

    import src.plot_fsrs_divergence

    args = [
        "src/plot_fsrs_divergence.py",
        "--days",
        "1",
        "--repeats",
        "1",
        "--concurrency",
        "1",
        "--seed-history",
        "dummy.anki2",
    ]
    monkeypatch.setattr(sys, "argv", args)

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
        src.plot_fsrs_divergence,
        "load_anki_history",
        lambda *a, **k: (logs, datetime.now()),
    )
    monkeypatch.setattr(
        src.plot_fsrs_divergence,
        "run_simulation",
        lambda *a, **k: (list(DEFAULT_PARAMETERS), list(DEFAULT_PARAMETERS), {}),
    )
    monkeypatch.setattr(
        src.plot_fsrs_divergence, "plot_forgetting_curves", lambda *a: None
    )

    class MockOpt:
        def __init__(self, *a: Any):
            pass

        def compute_optimal_parameters(self, verbose: bool = False) -> list[float]:
            return list(DEFAULT_PARAMETERS)

    monkeypatch.setattr(src.plot_fsrs_divergence, "RustOptimizer", MockOpt)

    main()


def test_main_with_few_logs(monkeypatch: Any) -> None:
    import sys

    import src.plot_fsrs_divergence

    args = ["src/plot_fsrs_divergence.py", "--seed-history", "dummy.anki2"]
    monkeypatch.setattr(sys, "argv", args)

    logs: dict[int, list[ReviewLog]] = defaultdict(list)
    log = ReviewLog(
        card_id=1,
        rating=Rating.Good,
        review_datetime=datetime.now(timezone.utc),
        review_duration=None,
    )
    logs[1].append(log)

    monkeypatch.setattr(
        src.plot_fsrs_divergence,
        "load_anki_history",
        lambda *a, **k: (logs, datetime.now()),
    )
    monkeypatch.setattr(
        src.plot_fsrs_divergence,
        "run_simulation",
        lambda *a, **k: (list(DEFAULT_PARAMETERS), list(DEFAULT_PARAMETERS), {}),
    )
    monkeypatch.setattr(
        src.plot_fsrs_divergence, "plot_forgetting_curves", lambda *a: None
    )

    main()


def test_run_single_task_exception() -> None:
    with patch(
        "src.plot_fsrs_divergence.run_simulation", side_effect=Exception("Test error")
    ):
        task = {
            "config_key": "test",
            "days": 1,
            "reviews": 10,
            "retention": "0.9",
            "burn_in": 0,
            "seed": 42,
        }
        res = run_single_task(task)
        assert not res["success"]
        assert res["error"] == "Test error"
