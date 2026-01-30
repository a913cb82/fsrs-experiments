from typing import Any

import numpy as np
from fsrs.scheduler import DEFAULT_PARAMETERS
from scipy.stats import entropy


def calculate_population_retrievability(
    t: np.ndarray[Any, Any],
    stabilities: np.ndarray[Any, Any],
    parameters: list[float] | tuple[float, ...],
) -> np.ndarray[Any, Any]:
    """
    Calculate the aggregate (average) forgetting curve for a population.
    Returns a 1D array of retrievability values for each time point in t.
    """
    if len(stabilities) == 0:
        res_ones: np.ndarray[Any, Any] = np.ones_like(t)
        return res_ones

    decay = -parameters[20]
    factor = 0.9 ** (1 / decay) - 1

    s_safe = np.maximum(stabilities, 0.001)

    # Broadcast: (T, 1) and (1, N) -> (T, N)
    r_matrix = (1 + factor * t[:, np.newaxis] / s_safe[np.newaxis, :]) ** decay

    # Average across the population
    res_mean: np.ndarray[Any, Any] = np.mean(r_matrix, axis=1)
    return res_mean


def calculate_metrics(
    gt_params: list[float] | tuple[float, ...],
    fit_params: list[float] | tuple[float, ...],
    stabilities: list[tuple[float, float]],
) -> tuple[float, float]:
    """
    Calculate RMSE and Mean KL Divergence between ground truth and fitted curves,
    averaged across all individual cards in the simulation using vectorization.
    """
    if not stabilities:
        return 0.0, 0.0

    t_eval = np.linspace(0, 100, 200)  # Shape (T,)
    s_nat = np.array([max(s[0], 0.001) for s in stabilities])  # Shape (N,)
    s_alg = np.array([max(s[1], 0.001) for s in stabilities])  # Shape (N,)

    # Pre-calculate constants
    decay_nat = -gt_params[20]
    factor_nat = 0.9 ** (1 / decay_nat) - 1
    decay_alg = -fit_params[20]
    factor_alg = 0.9 ** (1 / decay_alg) - 1

    # Broadcast evaluation across time and population
    # Resulting shape: (T, N)
    r_nat = (1 + factor_nat * t_eval[:, np.newaxis] / s_nat[np.newaxis, :]) ** decay_nat
    r_alg = (1 + factor_alg * t_eval[:, np.newaxis] / s_alg[np.newaxis, :]) ** decay_alg

    # RMSE per card: mean across time axis, then sqrt
    rmse_per_card = np.sqrt(np.mean((r_nat - r_alg) ** 2, axis=0))

    # KL Divergence per card
    p = np.clip(r_nat, 1e-10, 1 - 1e-10)
    q = np.clip(r_alg, 1e-10, 1 - 1e-10)
    kl_per_card = np.mean(entropy(p, q, axis=0) + entropy(1 - p, 1 - q, axis=0))

    return float(np.mean(rmse_per_card)), float(kl_per_card)


# Global storage for worker processes to avoid pickling overhead
_worker_seeded_data: dict[str, Any] | None = None


def init_worker(seeded_payload: dict[str, Any] | None) -> None:
    global _worker_seeded_data
    _worker_seeded_data = seeded_payload


def parse_retention_schedule(
    schedule_str: str,
) -> list[tuple[int, float]]:
    try:
        if ":" not in schedule_str:
            return [(1, float(schedule_str))]
        segments = []
        for part in schedule_str.split(","):
            d, r = part.split(":")
            segments.append((int(d), float(r)))
        return segments
    except ValueError:
        return [(1, 0.9)]


def get_retention_for_day(
    day: int, schedule_segments: list[tuple[int, float]]
) -> float:
    total_duration = sum(d for d, r in schedule_segments)
    day_in_cycle = day % total_duration
    current_pos = 0
    for duration, retention in schedule_segments:
        if current_pos <= day_in_cycle < current_pos + duration:
            return retention
        current_pos += duration
    return schedule_segments[-1][1]


def parse_parameters(params_str: str) -> tuple[float, ...]:
    try:
        parts = [float(p.strip()) for p in params_str.split(",")]
        if len(parts) != 21:
            return tuple(DEFAULT_PARAMETERS)
        return tuple(parts)
    except ValueError:
        return tuple(DEFAULT_PARAMETERS)
