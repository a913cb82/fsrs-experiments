from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from fsrs import Card, ReviewLog


@dataclass
class RatingWeights:
    first: list[float] = field(default_factory=lambda: [0.5, 0.1, 0.3, 0.1])
    success: list[float] = field(default_factory=lambda: [0.1, 0.8, 0.1])


@dataclass
class SeededData:
    logs: dict[int, list[ReviewLog]]
    last_rev: datetime
    true_cards: dict[int, Card]
    sys_cards: dict[int, Card]
    weights: RatingWeights | None = None


@dataclass
class SimulationConfig:
    n_days: int = 365
    burn_in_days: int = 0
    review_limit: int | None = 200
    new_limit: int | None = 10
    retention: str = "0.9"
    seed: int = 42
    time_limit: float | None = None
    time_estimator: Callable[[Card, int, datetime], float] | None = None
    rating_estimator: Callable[[Card, datetime], int] | None = None
    weights: RatingWeights = field(default_factory=RatingWeights)
    verbose: bool = True
