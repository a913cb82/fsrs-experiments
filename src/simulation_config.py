from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TypeAlias

from fsrs import Card, ReviewLog, Scheduler

# Simple Callable types for estimators
RatingEstimator: TypeAlias = Callable[[Card, datetime, Scheduler], int]
TimeEstimator: TypeAlias = Callable[[Card, datetime, int, Scheduler], float]


@dataclass
class SeededData:
    logs: dict[int, list[ReviewLog]]
    last_rev: datetime
    true_cards: dict[int, Card]
    sys_cards: dict[int, Card]


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
    verbose: bool = True
