from typing import Any

import fsrs_rs_python
import numpy as np


class RustOptimizer:
    """A wrapper for fsrs-rs-python to compute optimal FSRS parameters."""

    def __init__(
        self,
        card_ids: np.ndarray[Any, Any],
        ratings: np.ndarray[Any, Any],
        days: np.ndarray[Any, Any],
    ) -> None:
        self.card_ids = card_ids
        self.ratings = ratings
        self.days = days

    def compute_optimal_parameters(self, verbose: bool = False) -> list[float] | None:
        """Runs the Rust optimizer on the provided data."""
        if len(self.card_ids) == 0:
            return None

        # Sort and group by card_id to build histories
        idx = np.lexsort((self.days, self.card_ids))
        s_card_ids = self.card_ids[idx]
        s_ratings = self.ratings[idx]
        s_days = self.days[idx]

        diff = np.diff(s_card_ids)
        boundaries = np.where(diff != 0)[0] + 1
        boundaries = np.concatenate(([0], boundaries, [len(s_card_ids)]))

        rust_items = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            if end - start < 2:
                continue

            c_ratings = s_ratings[start:end]
            c_days = s_days[start:end]

            # FSRS backend expects delta_t. First review has delta_t=0
            reviews = [fsrs_rs_python.FSRSReview(int(c_ratings[0]), 0)]
            for j in range(1, len(c_ratings)):
                delta_t = int(c_days[j] - c_days[j - 1])
                reviews.append(fsrs_rs_python.FSRSReview(int(c_ratings[j]), delta_t))

            # Add each prefix of length >= 2 as a training item
            for j in range(2, len(reviews) + 1):
                rust_items.append(fsrs_rs_python.FSRSItem(reviews[:j]))

        # Filter items: require at least one long-term review
        filtered_items = [
            item for item in rust_items if item.long_term_review_cnt() > 0
        ]
        if not filtered_items:
            return None

        fsrs_core = fsrs_rs_python.FSRS(fsrs_rs_python.DEFAULT_PARAMETERS)
        optimized_w = fsrs_core.compute_parameters(filtered_items)
        return [float(x) for x in optimized_w]
