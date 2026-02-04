from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from deck import Deck

# Type for FSRS weight vector (usually 21 parameters)
FSRSParameters: TypeAlias = tuple[float, ...]


@dataclass
class LogData:
    """Simulation logs stored as NumPy arrays for high-performance."""

    card_ids: npt.NDArray[np.int64]
    ratings: npt.NDArray[np.int8]
    review_timestamps: npt.NDArray[np.datetime64]
    review_durations: npt.NDArray[np.float32]

    @classmethod
    def concatenate(cls, logs: Sequence["LogData"]) -> "LogData":
        """Efficiently concatenates multiple LogData instances."""
        if not logs:
            return cls(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int8),
                np.array([], dtype="datetime64[ns]"),
                np.array([], dtype=np.float32),
            )
        return cls(
            np.concatenate([log.card_ids for log in logs]),
            np.concatenate([log.ratings for log in logs]),
            np.concatenate([log.review_timestamps for log in logs]),
            np.concatenate([log.review_durations for log in logs]),
        )

    def __len__(self) -> int:
        return len(self.card_ids)

    @property
    def is_empty(self) -> bool:
        return len(self.card_ids) == 0


# Simplified Vectorized Estimator types
RatingEstimator: TypeAlias = Callable[
    [Deck, npt.NDArray[np.intp], datetime, FSRSParameters],
    npt.NDArray[np.int8],
]
TimeEstimator: TypeAlias = Callable[
    [Deck, npt.NDArray[np.intp], datetime, FSRSParameters, npt.NDArray[np.int8]],
    npt.NDArray[np.float32],
]


@dataclass
class SeededData:
    """Pre-loaded simulation state."""

    logs: LogData
    last_rev: datetime
    true_cards: Deck
    sys_cards: Deck


@dataclass
class SimulationConfig:
    n_days: int = 365
    burn_in_days: int = 0
    review_limit: int | None = 200
    new_limit: int | None = 10
    retention: str = "0.9"
    seed: int = 42
    time_limit: float | None = None
    time_estimator: TimeEstimator | None = None
    rating_estimator: RatingEstimator | None = None
    compute_final_params: bool = True
    return_logs: bool = True
    verbose: bool = True
