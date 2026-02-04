from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

T = TypeVar("T", bound=np.generic)


class Deck:
    def __init__(
        self,
        card_ids: npt.NDArray[np.int64] | None = None,
        stabilities: npt.NDArray[np.float64] | None = None,
        difficulties: npt.NDArray[np.float64] | None = None,
        dues: npt.NDArray[np.datetime64] | None = None,
        last_reviews: npt.NDArray[np.datetime64] | None = None,
        capacity: int = 100000,
    ) -> None:
        """
        Maintains deck state in NumPy arrays.
        Input arrays are copied into pre-allocated internal buffers.
        """
        n = len(card_ids) if card_ids is not None else 0
        cap = max(capacity, n)
        self.count = 0
        self.card_ids = np.zeros(cap, dtype=np.int64)
        self.stabilities = np.zeros(cap, dtype=np.float64)
        self.difficulties = np.zeros(cap, dtype=np.float64)
        self.dues = np.full(cap, np.datetime64("NaT"), dtype="datetime64[ns]")
        self.last_reviews = np.full(cap, np.datetime64("NaT"), dtype="datetime64[ns]")

        if n > 0 and card_ids is not None:
            self.add_cards(card_ids, stabilities, difficulties, dues, last_reviews)

    def add_cards(
        self,
        card_ids: npt.NDArray[np.int64],
        stabilities: npt.NDArray[np.float64] | None,
        difficulties: npt.NDArray[np.float64] | None,
        dues: npt.NDArray[np.datetime64] | None,
        last_reviews: npt.NDArray[np.datetime64] | None,
    ) -> None:
        n = len(card_ids)
        if self.count + n > len(self.card_ids):
            new_cap = max(len(self.card_ids) * 2, self.count + n)

            # Helper to grow array and pad with fill value
            def grow(arr: npt.NDArray[T], cap: int, fill: Any) -> npt.NDArray[T]:
                new_arr = np.full(cap, fill, dtype=arr.dtype)
                new_arr[: len(arr)] = arr
                return new_arr

            self.card_ids = grow(self.card_ids, new_cap, 0)  # type: ignore[assignment]
            self.stabilities = grow(self.stabilities, new_cap, 0.0)  # type: ignore[assignment]
            self.difficulties = grow(self.difficulties, new_cap, 0.0)  # type: ignore[assignment]
            self.dues = grow(self.dues, new_cap, np.datetime64("NaT"))  # type: ignore[assignment]
            self.last_reviews = grow(self.last_reviews, new_cap, np.datetime64("NaT"))  # type: ignore[assignment]

        self.card_ids[self.count : self.count + n] = card_ids
        self.stabilities[self.count : self.count + n] = (
            stabilities if stabilities is not None else 0.0
        )
        self.difficulties[self.count : self.count + n] = (
            difficulties if difficulties is not None else 0.0
        )
        self.dues[self.count : self.count + n] = (
            dues if dues is not None else np.datetime64("NaT")
        )
        self.last_reviews[self.count : self.count + n] = (
            last_reviews if last_reviews is not None else np.datetime64("NaT")
        )
        self.count += n

    def __len__(self) -> int:
        return self.count

    @property
    def current_card_ids(self) -> npt.NDArray[np.int64]:
        return self.card_ids[: self.count]

    @property
    def current_stabilities(self) -> npt.NDArray[np.float64]:
        return self.stabilities[: self.count]

    @property
    def current_difficulties(self) -> npt.NDArray[np.float64]:
        return self.difficulties[: self.count]

    @property
    def current_dues(self) -> npt.NDArray[np.datetime64]:
        return self.dues[: self.count]

    @property
    def current_last_reviews(self) -> npt.NDArray[np.datetime64]:
        return self.last_reviews[: self.count]

    def copy(self) -> "Deck":
        """Returns a copy of the deck with its current state."""
        new_deck = Deck(
            card_ids=self.current_card_ids.copy(),
            stabilities=self.current_stabilities.copy(),
            difficulties=self.current_difficulties.copy(),
            dues=self.current_dues.copy(),
            last_reviews=self.current_last_reviews.copy(),
            capacity=len(self.card_ids),
        )
        return new_deck
