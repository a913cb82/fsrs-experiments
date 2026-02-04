from typing import Any, cast

import numpy as np

MIN_DIFFICULTY = 1.0
MAX_DIFFICULTY = 10.0
STABILITY_MIN = 0.001


def predict_retrievability(
    stabilities: np.ndarray[Any, Any],
    elapsed_days: np.ndarray[Any, Any],
    params: tuple[float, ...],
) -> np.ndarray[Any, Any]:
    """Vectorized retrievability prediction using FSRS v6 formulas."""
    decay = -params[20]
    factor = 0.9 ** (1.0 / decay) - 1.0
    # Avoid divide by zero and ensure base is safe for power
    safe_stabilities = np.maximum(stabilities, 1e-10)
    base = 1.0 + factor * elapsed_days / safe_stabilities
    # Base must be positive for power operation
    res = np.maximum(base, 1e-10) ** decay
    return cast(
        np.ndarray[Any, Any], np.where(stabilities > 0, res, 0.0).astype(np.float64)
    )


def init_stability(
    ratings: np.ndarray[Any, Any], params: tuple[float, ...]
) -> np.ndarray[Any, Any]:
    """Vectorized initial stability using FSRS v6 formulas."""
    # ratings are 1, 2, 3, 4 (Again, Hard, Good, Easy)
    # params[0..3] are initial stabilities
    params_arr = np.array(params)
    return cast(
        np.ndarray[Any, Any],
        np.maximum(params_arr[ratings - 1], STABILITY_MIN).astype(np.float64),
    )


def init_difficulty(
    ratings: np.ndarray[Any, Any], params: tuple[float, ...]
) -> np.ndarray[Any, Any]:
    """Vectorized initial difficulty using FSRS v6 formulas."""
    # D0(r) = w4 - exp(w5 * (r-1)) + 1
    d0 = params[4] - np.exp(params[5] * (ratings - 1.0)) + 1.0
    return cast(
        np.ndarray[Any, Any],
        np.clip(d0, MIN_DIFFICULTY, MAX_DIFFICULTY).astype(np.float64),
    )


def _next_difficulty(
    difficulties: np.ndarray[Any, Any],
    ratings: np.ndarray[Any, Any],
    params: tuple[float, ...],
) -> np.ndarray[Any, Any]:
    """Helper for difficulty updates."""
    # arg_1 = initial_difficulty(Easy, clamp=False)
    # Easy rating is 4
    arg_1 = params[4] - np.exp(params[5] * (4.0 - 1.0)) + 1.0

    delta_difficulty = -(params[6] * (ratings - 3.0))
    # arg_2 = difficulty + (10 - difficulty) * delta_difficulty / 9
    arg_2 = difficulties + (10.0 - difficulties) * delta_difficulty / 9.0

    # next_difficulty = w7 * arg_1 + (1 - w7) * arg_2
    next_d = params[7] * arg_1 + (1.0 - params[7]) * arg_2
    return cast(
        np.ndarray[Any, Any],
        np.clip(next_d, MIN_DIFFICULTY, MAX_DIFFICULTY).astype(np.float64),
    )


def update_state_recall(
    stabilities: np.ndarray[Any, Any],
    difficulties: np.ndarray[Any, Any],
    ratings: np.ndarray[Any, Any],
    retrievabilities: np.ndarray[Any, Any],
    params: tuple[float, ...],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Vectorized recall update using FSRS v6 formulas."""
    # S' = S * [1 + exp(w8) * (11-D) * S^-w9 * (exp(w10 * (1-R)) - 1) * penalty * bonus]
    hard_penalty = np.where(ratings == 2, params[15], 1.0)  # Rating.Hard = 2
    easy_bonus = np.where(ratings == 4, params[16], 1.0)  # Rating.Easy = 4

    s_inc = (
        1.0
        + np.exp(params[8])
        * (11.0 - difficulties)
        * (stabilities ** -params[9])
        * (np.exp(params[10] * (1.0 - retrievabilities)) - 1.0)
        * hard_penalty
        * easy_bonus
    )

    new_s = stabilities * s_inc
    new_d = _next_difficulty(difficulties, ratings, params)

    return (
        cast(np.ndarray[Any, Any], np.maximum(new_s, STABILITY_MIN).astype(np.float64)),
        cast(np.ndarray[Any, Any], new_d.astype(np.float64)),
    )


def update_state_forget(
    stabilities: np.ndarray[Any, Any],
    difficulties: np.ndarray[Any, Any],
    retrievabilities: np.ndarray[Any, Any],
    params: tuple[float, ...],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Vectorized forget update using FSRS v6 formulas."""
    # S'_long = w11 * D^-w12 * ((S+1)^w13 - 1) * exp(w14 * (1-R))
    s_long = (
        params[11]
        * (difficulties ** -params[12])
        * ((stabilities + 1.0) ** params[13] - 1.0)
        * np.exp(params[14] * (1.0 - retrievabilities))
    )

    # S'_short = S / exp(w17 * w18)
    s_short = stabilities / np.exp(params[17] * params[18])

    new_s = np.minimum(s_long, s_short)
    # Rating.Again = 1
    new_d = _next_difficulty(difficulties, np.array([1.0]), params)

    return (
        cast(np.ndarray[Any, Any], np.maximum(new_s, STABILITY_MIN).astype(np.float64)),
        cast(np.ndarray[Any, Any], new_d.astype(np.float64)),
    )


def next_interval(
    stabilities: np.ndarray[Any, Any],
    desired_retention: float,
    params: tuple[float, ...],
    max_interval: int = 36500,
) -> np.ndarray[Any, Any]:
    """Vectorized next interval calculation."""
    decay = -params[20]
    factor = 0.9 ** (1.0 / decay) - 1.0

    intervals = (stabilities / factor) * (desired_retention ** (1.0 / decay) - 1.0)
    return cast(
        np.ndarray[Any, Any],
        np.clip(np.round(intervals), 1, max_interval).astype(np.int32),
    )
