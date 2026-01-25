import numpy as np
import pytest

from src.plot_fsrs_divergence import (
    calculate_metrics,
    calculate_population_retrievability,
)
from src.simulate_fsrs import DEFAULT_PARAMETERS


def test_calculate_population_retrievability() -> None:
    t = np.array([0, 1, 10])
    stabilities = np.array([5.0, 10.0, 50.0])

    r_agg = calculate_population_retrievability(t, stabilities, DEFAULT_PARAMETERS)

    assert len(r_agg) == 3
    assert r_agg[0] == 1.0  # R at t=0 is always 1.0
    assert 0 <= r_agg[2] <= 1.0


def test_calculate_metrics_identical() -> None:
    # Metrics between identical params and stabilities should be 0
    params = list(DEFAULT_PARAMETERS)
    stabilities = [(10.0, 10.0), (20.0, 20.0)]

    rmse, kl = calculate_metrics(params, params, stabilities)

    assert rmse == pytest.approx(0.0)
    assert kl == pytest.approx(0.0)


def test_calculate_metrics_divergent() -> None:
    params_gt = list(DEFAULT_PARAMETERS)
    params_fit = list(DEFAULT_PARAMETERS)
    params_fit[20] += 0.05  # Change decay parameter

    stabilities = [(10.0, 10.0)]

    rmse, kl = calculate_metrics(params_gt, params_fit, stabilities)

    assert rmse > 0
    assert kl > 0
