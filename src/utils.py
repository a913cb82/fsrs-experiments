from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.stats import entropy

from simulation_config import FSRSParameters

__all__ = [
    "DEFAULT_PARAMETERS",
    "calculate_metrics",
    "get_retention_for_day",
    "parse_parameters",
    "parse_retention_schedule",
]

DEFAULT_PARAMETERS: FSRSParameters = (
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


def calculate_metrics(
    gt_params: FSRSParameters,
    fit_params: FSRSParameters,
    s_nat_raw: npt.NDArray[np.float64],
    s_alg_raw: npt.NDArray[np.float64],
) -> tuple[float, float]:
    """Calculates RMSE and KL-Divergence between Nature and System stabilities."""
    w_nat = np.array(gt_params)
    w_fit = np.array(fit_params)

    # RMSE of parameters
    rmse = float(np.sqrt(np.mean((w_nat - w_fit) ** 2)))

    # KL-Divergence of stability distributions
    min_s = float(min(cast(float, np.min(s_nat_raw)), cast(float, np.min(s_alg_raw))))
    max_s = float(max(cast(float, np.max(s_nat_raw)), cast(float, np.max(s_alg_raw))))
    bins = np.linspace(min_s, max_s, 50)

    p, _ = np.histogram(s_nat_raw, bins=bins, density=True)
    q, _ = np.histogram(s_alg_raw, bins=bins, density=True)

    # Avoid zeros for KL
    p = p + 1e-10
    q = q + 1e-10

    kl = float(entropy(p, q))

    return rmse, kl


def get_retention_for_day(day: int, schedule: list[tuple[int, float]]) -> float:
    """Gets the target retention for a given day index based on schedule."""
    current_ret = 0.9
    for s_day, s_ret in schedule:
        if day >= s_day:
            current_ret = s_ret
        else:
            break
    return current_ret


def parse_retention_schedule(retention_str: str) -> list[tuple[int, float]]:
    """
    Parses retention schedule string.
    Format: "0.9" (constant) or "10:0.9,20:0.85" (day:retention pairs)
    """
    try:
        if ":" not in retention_str:
            return [(0, float(retention_str))]

        parts = retention_str.split(",")
        schedule = []
        for p in parts:
            d_s, r_s = p.split(":")
            schedule.append((int(d_s), float(r_s)))
        return sorted(schedule)
    except (ValueError, IndexError):
        return [(0, 0.9)]


def parse_parameters(p_str: str) -> FSRSParameters:
    """Parses comma-separated FSRS parameters."""
    try:
        vals = [float(x.strip()) for x in p_str.split(",")]
        if len(vals) != 21:
            return DEFAULT_PARAMETERS
        return tuple(vals)
    except (ValueError, AttributeError):
        return DEFAULT_PARAMETERS
