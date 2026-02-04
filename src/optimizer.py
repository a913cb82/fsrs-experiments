from collections import defaultdict
from typing import Any

import numpy as np
from fsrs import ReviewLog

# Use the Rust-powered optimizer if available
try:
    import fsrs_rs_python

    HAS_RUST_OPTIMIZER = True
except ImportError:
    HAS_RUST_OPTIMIZER = False


class RustOptimizer:
    """A wrapper for fsrs-rs-python to match our existing Optimizer interface."""

    def __init__(
        self,
        review_logs: list[ReviewLog] | None = None,
        pre_constructed_items: list[Any] | None = None,
    ) -> None:
        self.review_logs = review_logs or []
        self.pre_constructed_items = pre_constructed_items or []

    def compute_optimal_parameters(self, verbose: bool = False) -> list[float] | None:
        items = self.get_items()
        return self._run_rust_optimizer(items)

    def get_items(self) -> list[Any]:
        if not HAS_RUST_OPTIMIZER:
            return []

        items_map: dict[int, list[ReviewLog]] = defaultdict(list)
        for log in self.review_logs:
            items_map[log.card_id].append(log)

        rust_items = list(self.pre_constructed_items)
        for card_id in items_map:
            logs = sorted(items_map[card_id], key=lambda x: x.review_datetime)
            if len(logs) < 2:
                continue

            all_reviews = []
            last_date = None
            for log in logs:
                delta_t = (log.review_datetime - last_date).days if last_date else 0
                all_reviews.append(
                    fsrs_rs_python.FSRSReview(int(log.rating), int(delta_t))
                )
                last_date = log.review_datetime

            for i in range(2, len(all_reviews) + 1):
                rust_items.append(fsrs_rs_python.FSRSItem(all_reviews[:i]))
        return rust_items

    def compute_optimal_parameters_from_arrays(
        self,
        card_ids: np.ndarray[Any, Any],
        ratings: np.ndarray[Any, Any],
        days: np.ndarray[Any, Any],
        verbose: bool = False,
    ) -> list[float] | None:
        items = self.get_items_from_arrays(card_ids, ratings, days)
        return self._run_rust_optimizer(items)

    def get_items_from_arrays(
        self,
        card_ids: np.ndarray[Any, Any],
        ratings: np.ndarray[Any, Any],
        days: np.ndarray[Any, Any],
    ) -> list[Any]:
        if not HAS_RUST_OPTIMIZER:
            return []
        if len(card_ids) == 0:
            return list(self.pre_constructed_items)

        idx = np.lexsort((days, card_ids))
        s_card_ids = card_ids[idx]
        s_ratings = ratings[idx]
        s_days = days[idx]

        diff = np.diff(s_card_ids)
        boundaries = np.where(diff != 0)[0] + 1
        boundaries = np.concatenate(([0], boundaries, [len(s_card_ids)]))

        rust_items = list(self.pre_constructed_items)
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            if end - start < 2:
                continue

            c_ratings = s_ratings[start:end]
            c_days = s_days[start:end]

            all_reviews = []
            all_reviews.append(fsrs_rs_python.FSRSReview(int(c_ratings[0]), 0))
            for j in range(1, len(c_ratings)):
                delta_t = int(c_days[j] - c_days[j - 1])
                all_reviews.append(
                    fsrs_rs_python.FSRSReview(int(c_ratings[j]), delta_t)
                )

            for j in range(2, len(all_reviews) + 1):
                rust_items.append(fsrs_rs_python.FSRSItem(all_reviews[:j]))
        return rust_items

    def _run_rust_optimizer(self, rust_items: list[Any]) -> list[float] | None:
        filtered_items = [
            item for item in rust_items if item.long_term_review_cnt() > 0
        ]
        if not filtered_items:
            return None
        fsrs_core = fsrs_rs_python.FSRS(fsrs_rs_python.DEFAULT_PARAMETERS)
        optimized_w = fsrs_core.compute_parameters(filtered_items)
        return [float(x) for x in optimized_w]
