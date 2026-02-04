from typing import Any

import numpy as np

from anki_utils import START_DATE


class Deck:
    def __init__(
        self,
        card_ids: np.ndarray[Any, Any] | None = None,
        stabilities: np.ndarray[Any, Any] | None = None,
        difficulties: np.ndarray[Any, Any] | None = None,
        dues: np.ndarray[Any, Any] | None = None,
        last_reviews: np.ndarray[Any, Any] | None = None,
        capacity: int = 100000,
    ) -> None:
        if card_ids is not None:
            n = len(card_ids)
            self.count = n
            cap = max(capacity, n * 2)
            self.card_ids = np.zeros(cap, dtype=np.int64)
            self.stabilities = np.zeros(cap, dtype=np.float64)
            self.difficulties = np.zeros(cap, dtype=np.float64)
            self.dues = np.zeros(cap, dtype=np.int32)
            self.last_reviews = np.zeros(cap, dtype=np.int32)

            self.card_ids[:n] = card_ids
            self.stabilities[:n] = stabilities if stabilities is not None else 0.0
            self.difficulties[:n] = difficulties if difficulties is not None else 0.0
            self.dues[:n] = dues if dues is not None else 0
            self.last_reviews[:n] = last_reviews if last_reviews is not None else -1
        else:
            self.count = 0
            self.card_ids = np.zeros(capacity, dtype=np.int64)
            self.stabilities = np.zeros(capacity, dtype=np.float64)
            self.difficulties = np.zeros(capacity, dtype=np.float64)
            self.dues = np.zeros(capacity, dtype=np.int32)
            self.last_reviews = np.zeros(capacity, dtype=np.int32)

    @classmethod
    def from_cards(cls, cards: list[Any], start_date: Any = START_DATE) -> "Deck":
        if not cards:
            return cls()
        card_ids = np.array([c.card_id for c in cards], dtype=np.int64)
        stabilities = np.array(
            [c.stability if c.stability is not None else 0.0 for c in cards],
            dtype=np.float64,
        )
        difficulties = np.array(
            [c.difficulty if c.difficulty is not None else 0.0 for c in cards],
            dtype=np.float64,
        )
        dues = np.array([(c.due - start_date).days for c in cards], dtype=np.int32)
        last_reviews = np.array(
            [
                (c.last_review - start_date).days if c.last_review is not None else -1
                for c in cards
            ],
            dtype=np.int32,
        )
        return cls(card_ids, stabilities, difficulties, dues, last_reviews)

    def add_cards(
        self,
        card_ids: np.ndarray[Any, Any],
        stabilities: np.ndarray[Any, Any],
        difficulties: np.ndarray[Any, Any],
        dues: np.ndarray[Any, Any],
        last_reviews: np.ndarray[Any, Any],
    ) -> None:
        n = len(card_ids)
        if self.count + n > len(self.card_ids):
            new_cap = len(self.card_ids) * 2
            self.card_ids = np.resize(self.card_ids, new_cap)
            self.stabilities = np.resize(self.stabilities, new_cap)
            self.difficulties = np.resize(self.difficulties, new_cap)
            self.dues = np.resize(self.dues, new_cap)
            self.last_reviews = np.resize(self.last_reviews, new_cap)

        self.card_ids[self.count : self.count + n] = card_ids
        self.stabilities[self.count : self.count + n] = stabilities
        self.difficulties[self.count : self.count + n] = difficulties
        self.dues[self.count : self.count + n] = dues
        self.last_reviews[self.count : self.count + n] = last_reviews
        self.count += n

    def __len__(self) -> int:
        return self.count

    @property
    def current_card_ids(self) -> np.ndarray[Any, Any]:
        return self.card_ids[: self.count]

    @property
    def current_stabilities(self) -> np.ndarray[Any, Any]:
        return self.stabilities[: self.count]

    @property
    def current_difficulties(self) -> np.ndarray[Any, Any]:
        return self.difficulties[: self.count]

    @property
    def current_dues(self) -> np.ndarray[Any, Any]:
        return self.dues[: self.count]

    @property
    def current_last_reviews(self) -> np.ndarray[Any, Any]:
        return self.last_reviews[: self.count]
