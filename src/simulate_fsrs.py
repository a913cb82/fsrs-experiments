import random
from datetime import datetime, timedelta
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

import fsrs_engine
from anki_utils import (
    START_DATE,
    load_anki_history,
)
from deck import Deck
from optimizer import RustOptimizer
from simulation_config import (
    FSRSParameters,
    LogData,
    RatingEstimator,
    SeededData,
    SimulationConfig,
    TimeEstimator,
)
from utils import (
    DEFAULT_PARAMETERS,
    get_retention_for_day,
    parse_retention_schedule,
)


def _batch_process_reviews_numpy(
    deck_sys: Deck,
    deck_true: Deck,
    indices: npt.NDArray[np.intp],
    nature_params: FSRSParameters,
    algo_params: FSRSParameters,
    current_date: datetime,
    desired_retention: float,
    rating_estimator: RatingEstimator,
    time_estimator: TimeEstimator,
    card_logs_acc: list[LogData],
) -> int:
    if len(indices) == 0:
        return 0

    cur_ts = np.datetime64(current_date)

    ratings_vals = rating_estimator(
        deck_true,
        indices,
        current_date,
        nature_params,
    )

    times_vals = time_estimator(
        deck_true,
        indices,
        current_date,
        nature_params,
        ratings_vals,
    )

    actual_indices = indices
    actual_ratings = ratings_vals
    actual_times = times_vals

    last_revs = deck_true.current_last_reviews[indices]
    elapsed_days = (cur_ts - last_revs) / np.timedelta64(1, "D")
    elapsed_days = np.nan_to_num(elapsed_days, nan=0.0)

    forget_mask = actual_ratings == 1
    recall_mask = ~forget_mask

    if np.any(recall_mask):
        idx_recall = actual_indices[recall_mask]
        rat_recall = actual_ratings[recall_mask]
        elap_recall = elapsed_days[recall_mask]

        ret_true = fsrs_engine.predict_retrievability(
            deck_true.stabilities[idx_recall], elap_recall, nature_params
        )
        new_s_true, new_d_true = fsrs_engine.update_state_recall(
            deck_true.stabilities[idx_recall],
            deck_true.difficulties[idx_recall],
            rat_recall,
            ret_true,
            nature_params,
        )
        deck_true.stabilities[idx_recall] = new_s_true
        deck_true.difficulties[idx_recall] = new_d_true

        last_revs_algo = deck_sys.current_last_reviews[idx_recall]
        elap_algo = (cur_ts - last_revs_algo) / np.timedelta64(1, "D")
        elap_algo = np.nan_to_num(elap_algo, nan=0.0)

        ret_sys = fsrs_engine.predict_retrievability(
            deck_sys.stabilities[idx_recall], elap_algo, algo_params
        )
        new_s_sys, new_d_sys = fsrs_engine.update_state_recall(
            deck_sys.stabilities[idx_recall],
            deck_sys.difficulties[idx_recall],
            rat_recall,
            ret_sys,
            algo_params,
        )
        deck_sys.stabilities[idx_recall] = new_s_sys
        deck_sys.difficulties[idx_recall] = new_d_sys

        intervals = fsrs_engine.next_interval(new_s_sys, desired_retention, algo_params)
        ivals_td = np.array(intervals, dtype="timedelta64[D]")
        deck_sys.dues[idx_recall] = cur_ts + ivals_td
        deck_sys.last_reviews[idx_recall] = cur_ts
        deck_true.last_reviews[idx_recall] = cur_ts

    if np.any(forget_mask):
        idx_forget = actual_indices[forget_mask]
        elap_forget = elapsed_days[forget_mask]

        ret_true = fsrs_engine.predict_retrievability(
            deck_true.stabilities[idx_forget], elap_forget, nature_params
        )
        new_s_true, new_d_true = fsrs_engine.update_state_forget(
            deck_true.stabilities[idx_forget],
            deck_true.difficulties[idx_forget],
            ret_true,
            nature_params,
        )
        deck_true.stabilities[idx_forget] = new_s_true
        deck_true.difficulties[idx_forget] = new_d_true

        last_revs_algo_f = deck_sys.current_last_reviews[idx_forget]
        elap_algo_f = (cur_ts - last_revs_algo_f) / np.timedelta64(1, "D")
        elap_algo_f = np.nan_to_num(elap_algo_f, nan=0.0)

        ret_sys = fsrs_engine.predict_retrievability(
            deck_sys.stabilities[idx_forget], elap_algo_f, algo_params
        )
        new_s_sys, new_d_sys = fsrs_engine.update_state_forget(
            deck_sys.stabilities[idx_forget],
            deck_sys.difficulties[idx_forget],
            ret_sys,
            algo_params,
        )
        deck_sys.stabilities[idx_forget] = new_s_sys
        deck_sys.difficulties[idx_forget] = new_d_sys

        deck_sys.dues[idx_forget] = cur_ts + np.timedelta64(1, "D")
        deck_sys.last_reviews[idx_forget] = cur_ts
        deck_true.last_reviews[idx_forget] = cur_ts

    card_logs_acc.append(
        LogData(
            card_ids=deck_sys.card_ids[actual_indices],
            ratings=actual_ratings,
            review_timestamps=np.full(len(actual_indices), cur_ts),
            review_durations=actual_times,
        )
    )

    return len(actual_indices)


def simulate_day_numpy(
    nature_params: FSRSParameters,
    algo_params: FSRSParameters,
    deck_true: Deck,
    deck_sys: Deck,
    card_logs_acc: list[LogData],
    current_date: datetime,
    config: SimulationConfig,
    desired_retention: float,
    rating_estimator: RatingEstimator,
    time_estimator: TimeEstimator,
    next_card_id: int,
) -> int:
    cur_ts = np.datetime64(current_date)
    due_mask = deck_sys.current_dues <= cur_ts
    due_indices = np.where(due_mask)[0]

    time_accumulated = 0.0
    reviews_done = 0

    if len(due_indices) > 0:
        np.random.shuffle(due_indices)
        if config.review_limit is not None:
            due_indices = due_indices[: config.review_limit]

        count = _batch_process_reviews_numpy(
            deck_sys,
            deck_true,
            due_indices,
            nature_params,
            algo_params,
            current_date,
            desired_retention,
            rating_estimator,
            time_estimator,
            card_logs_acc,
        )
        reviews_done += count
        if count > 0:
            time_accumulated += float(np.sum(card_logs_acc[-1].review_durations))

    new_done = 0
    new_batch_size = 50
    while True:
        if config.time_limit is not None and time_accumulated >= config.time_limit:
            break

        remaining_total = (
            (config.review_limit - reviews_done)
            if config.review_limit is not None
            else 999999
        )
        remaining_new = (
            (config.new_limit - new_done) if config.new_limit is not None else 999999
        )

        if remaining_total <= 0 or remaining_new <= 0:
            break

        batch_size = min(new_batch_size, remaining_total, remaining_new)
        new_ids_raw = np.arange(next_card_id, next_card_id + batch_size, dtype=np.int64)
        new_ids = cast(npt.NDArray[np.int64], new_ids_raw)
        next_card_id += int(batch_size)

        nat_dtype = "datetime64[ns]"
        tmp_deck = Deck(
            new_ids,
            np.zeros(batch_size),
            np.zeros(batch_size),
            np.full(batch_size, np.datetime64("NaT"), dtype=nat_dtype),
            np.full(batch_size, np.datetime64("NaT"), dtype=nat_dtype),
        )

        init_ratings = rating_estimator(
            tmp_deck,
            np.arange(batch_size, dtype=np.intp),
            current_date,
            nature_params,
        )

        s_true = fsrs_engine.init_stability(init_ratings, nature_params)
        d_true = fsrs_engine.init_difficulty(init_ratings, nature_params)
        s_sys = fsrs_engine.init_stability(init_ratings, algo_params)
        d_sys = fsrs_engine.init_difficulty(init_ratings, algo_params)

        intervals = fsrs_engine.next_interval(s_sys, desired_retention, algo_params)
        ivals_td = np.array(intervals, dtype="timedelta64[D]")
        dues = cur_ts + ivals_td
        dues[init_ratings == 1] = cur_ts + np.timedelta64(1, "D")

        times = time_estimator(
            tmp_deck,
            np.arange(batch_size, dtype=np.intp),
            current_date,
            nature_params,
            init_ratings,
        )

        batch_completed = batch_size
        cumulative_times = np.cumsum(times)
        if config.time_limit is not None:
            batch_completed = int(
                np.searchsorted(
                    cumulative_times + time_accumulated,
                    config.time_limit,
                    side="right",
                )
            )

        if batch_completed == 0:
            break

        init_ratings = init_ratings[:batch_completed]
        new_ids = cast(npt.NDArray[np.int64], new_ids[:batch_completed])
        times = times[:batch_completed]
        s_true, d_true = s_true[:batch_completed], d_true[:batch_completed]
        s_sys, d_sys = s_sys[:batch_completed], d_sys[:batch_completed]
        dues = dues[:batch_completed]

        nat_const = np.datetime64("NaT")
        deck_true.add_cards(
            new_ids,
            s_true,
            d_true,
            np.full(batch_completed, nat_const, dtype=nat_dtype),
            np.full(batch_completed, cur_ts, dtype=nat_dtype),
        )
        deck_sys.add_cards(
            new_ids,
            s_sys,
            d_sys,
            dues,
            np.full(batch_completed, cur_ts, dtype=nat_dtype),
        )

        card_logs_acc.append(
            LogData(
                card_ids=new_ids,
                ratings=init_ratings,
                review_timestamps=np.full(batch_completed, cur_ts),
                review_durations=times,
            )
        )

        time_accumulated += float(np.sum(times))
        reviews_done += batch_completed
        new_done += batch_completed

        if batch_completed < batch_size:
            break

    return next_card_id


def _get_estimators_numpy(
    config: SimulationConfig,
    card_logs: LogData,
    nature_params: FSRSParameters,
) -> tuple[RatingEstimator, TimeEstimator]:
    from anki_utils import infer_review_weights

    weights = infer_review_weights(card_logs)
    avg_durations = np.zeros(5)
    if len(card_logs.review_durations) > 0:
        for r in [1, 2, 3, 4]:
            durs = card_logs.review_durations[card_logs.ratings == r]
            avg_durations[r] = float(np.mean(durs)) if len(durs) > 0 else 0.0

    if config.rating_estimator:
        rating_est = config.rating_estimator
    else:

        def rating_est(
            deck: Deck,
            indices: npt.NDArray[np.intp],
            date: datetime,
            params: FSRSParameters,
        ) -> npt.NDArray[np.int8]:
            stabilities = deck.current_stabilities[indices]
            last_reviews = deck.current_last_reviews[indices]
            count = len(indices)
            results = np.zeros(count, dtype=np.int8)
            new_mask = np.isnat(last_reviews)
            review_mask = ~new_mask
            if np.any(new_mask):
                results[new_mask] = np.random.choice(
                    [1, 2, 3, 4], size=np.sum(new_mask), p=weights.first
                )
            if np.any(review_mask):
                n_rev = np.sum(review_mask)
                # elapsed in days
                diff = np.datetime64(date) - last_reviews[review_mask]
                elapsed = diff / np.timedelta64(1, "D")
                rets = fsrs_engine.predict_retrievability(
                    stabilities[review_mask],
                    elapsed,
                    params,
                )
                success = np.random.random(n_rev) < rets
                results_rev = np.ones(n_rev, dtype=np.int8)
                n_success = np.sum(success)
                if n_success > 0:
                    results_rev[success] = np.random.choice(
                        [2, 3, 4], size=n_success, p=weights.success
                    )
                results[review_mask] = results_rev
            return results

    if config.time_estimator:
        time_est = config.time_estimator
    else:

        def time_est(
            deck: Deck,
            indices: npt.NDArray[np.intp],
            date: datetime,
            params: FSRSParameters,
            ratings: npt.NDArray[np.int8],
        ) -> npt.NDArray[np.float32]:
            return avg_durations[ratings].astype(np.float32)

    return rating_est, time_est


def run_simulation(
    config: SimulationConfig,
    ground_truth: FSRSParameters | None = None,
    initial_params: FSRSParameters | None = None,
    seeded_data: SeededData | None = None,
    seed_history: str | None = None,
    deck_config: str | None = None,
    deck_name: str | None = None,
    tqdm_pos: int = 0,
) -> tuple[list[float] | None, FSRSParameters, dict[str, Any]]:
    parsed_schedule = parse_retention_schedule(config.retention)
    np.random.seed(config.seed)
    random.seed(config.seed)

    gt_params = ground_truth if ground_truth is not None else DEFAULT_PARAMETERS
    algo_params = initial_params or DEFAULT_PARAMETERS

    try:
        deck_true, deck_sys, initial_logs, current_date = _load_initial_state(
            gt_params,
            algo_params,
            seeded_data,
            seed_history,
            deck_config,
            deck_name,
        )

        rating_est, time_est = _get_estimators_numpy(config, initial_logs, gt_params)
        card_logs_acc: list[LogData] = []

        next_card_id = (
            int(np.max(deck_sys.current_card_ids)) + 1
            if len(deck_sys) > 0
            else int(datetime.now().timestamp() * 1000)
        )

        phases = [
            (
                "P1",
                config.burn_in_days
                if 0 < config.burn_in_days < config.n_days
                else config.n_days,
                0,
            )
        ]
        if 0 < config.burn_in_days < config.n_days:
            phases.append(("P2", config.n_days, config.burn_in_days))

        for phase_name, limit, start_day in phases:
            pbar = tqdm(
                range(start_day, limit),
                desc=f"Simulating ({phase_name})",
                position=tqdm_pos,
                leave=False,
                disable=not config.verbose,
            )
            for _ in pbar:
                tz_naive = START_DATE.replace(tzinfo=None)
                day_offset = int((current_date - tz_naive).total_seconds() / 86400)
                ret = get_retention_for_day(day_offset, parsed_schedule)
                next_card_id = simulate_day_numpy(
                    gt_params,
                    algo_params,
                    deck_true,
                    deck_sys,
                    card_logs_acc,
                    current_date,
                    config,
                    ret,
                    rating_est,
                    time_est,
                    next_card_id,
                )
                current_date += timedelta(days=1)

            if phase_name == "P1" and 0 < config.burn_in_days < config.n_days:
                if card_logs_acc:
                    sim_logs = LogData.concatenate(card_logs_acc)
                    all_ids = np.concatenate([initial_logs.card_ids, sim_logs.card_ids])
                    all_rats = np.concatenate([initial_logs.ratings, sim_logs.ratings])
                    all_ts = np.concatenate(
                        [initial_logs.review_timestamps, sim_logs.review_timestamps]
                    )
                    all_days = (
                        (all_ts - np.datetime64(START_DATE)) / np.timedelta64(1, "D")
                    ).astype(np.int32)

                    optimizer = RustOptimizer(all_ids, all_rats, all_days)
                    new_params = optimizer.compute_optimal_parameters(
                        verbose=config.verbose
                    )
                    if new_params:
                        algo_params = tuple(new_params)

        sim_logs_final = LogData.concatenate(card_logs_acc)
        total_simulated_reviews = len(sim_logs_final)

        if config.compute_final_params:
            if total_simulated_reviews > 0:
                all_ids_final = np.concatenate(
                    [initial_logs.card_ids, sim_logs_final.card_ids]
                )
                all_rats_final = np.concatenate(
                    [initial_logs.ratings, sim_logs_final.ratings]
                )
                all_ts_final = np.concatenate(
                    [initial_logs.review_timestamps, sim_logs_final.review_timestamps]
                )
                diff_final = all_ts_final - np.datetime64(START_DATE)
                all_days_final = (diff_final / np.timedelta64(1, "D")).astype(np.int32)

                optimizer = RustOptimizer(all_ids_final, all_rats_final, all_days_final)
                fitted_params = optimizer.compute_optimal_parameters(
                    verbose=config.verbose
                )
            else:
                diff_init = initial_logs.review_timestamps - np.datetime64(START_DATE)
                all_days_final = (diff_init / np.timedelta64(1, "D")).astype(np.int32)
                optimizer = RustOptimizer(
                    initial_logs.card_ids, initial_logs.ratings, all_days_final
                )
                fitted_params = optimizer.compute_optimal_parameters(
                    verbose=config.verbose
                )
        else:
            fitted_params = None

        final_logs = (
            LogData.concatenate([initial_logs, sim_logs_final])
            if config.return_logs
            else LogData.concatenate([])
        )

        diff_cur = np.datetime64(current_date) - deck_true.current_last_reviews
        elapsed = diff_cur / np.timedelta64(1, "D")
        elapsed = np.nan_to_num(elapsed, nan=0.0)
        retention = np.sum(
            fsrs_engine.predict_retrievability(
                deck_true.current_stabilities, elapsed, gt_params
            )
        )

        metrics = {
            "review_count": len(final_logs)
            if config.return_logs
            else total_simulated_reviews + len(initial_logs),
            "card_count": len(deck_true),
            "stabilities": (
                deck_true.current_stabilities,
                deck_sys.current_stabilities,
            ),
            "total_retention": float(retention),
            "logs": final_logs,
        }
        return fitted_params, gt_params, metrics
    finally:
        pass


def _load_initial_state(
    nature_params: FSRSParameters,
    algo_params: FSRSParameters,
    seeded_data: SeededData | None,
    seed_history: str | None,
    deck_config: str | None,
    deck_name: str | None,
) -> tuple[Deck, Deck, LogData, datetime]:
    current_date = START_DATE

    if seeded_data:
        current_date = seeded_data.last_rev + timedelta(days=1)
        return (
            seeded_data.true_cards.copy(),
            seeded_data.sys_cards.copy(),
            seeded_data.logs.copy(),
            current_date,
        )

    if seed_history:
        logs_data, last_rev = load_anki_history(seed_history, deck_config, deck_name)
        current_date = last_rev + timedelta(days=1)

        df = pd.DataFrame(
            {
                "card_id": logs_data.card_ids,
                "rating": logs_data.ratings,
                "timestamp": logs_data.review_timestamps,
            }
        )

        states_nat = {}
        states_alg = {}

        for cid, group in df.groupby("card_id"):
            sorted_group = group.sort_values("timestamp")
            nat_const = cast(np.datetime64, np.datetime64("NaT"))
            s_nat, d_nat, last_rev_nat = 0.0, 0.0, nat_const
            s_alg, d_alg, last_rev_alg = 0.0, 0.0, nat_const

            for i, row in enumerate(sorted_group.itertuples()):
                rat = int(cast(Any, row).rating)
                ts = cast(np.datetime64, row.timestamp)
                if i == 0:
                    s_nat = float(
                        fsrs_engine.init_stability(
                            np.array([rat], dtype=np.int8), nature_params
                        )[0]
                    )
                    d_nat = float(
                        fsrs_engine.init_difficulty(
                            np.array([rat], dtype=np.int8), nature_params
                        )[0]
                    )
                    s_alg = float(
                        fsrs_engine.init_stability(
                            np.array([rat], dtype=np.int8), algo_params
                        )[0]
                    )
                    d_alg = float(
                        fsrs_engine.init_difficulty(
                            np.array([rat], dtype=np.int8), algo_params
                        )[0]
                    )
                else:
                    elapsed = float((ts - last_rev_nat) / np.timedelta64(1, "D"))
                    ret_nat = float(
                        fsrs_engine.predict_retrievability(
                            np.array([s_nat]), np.array([elapsed]), nature_params
                        )[0]
                    )
                    if rat == 1:
                        s_nat_arr, d_nat_arr = fsrs_engine.update_state_forget(
                            np.array([s_nat]),
                            np.array([d_nat]),
                            np.array([ret_nat]),
                            nature_params,
                        )
                    else:
                        s_nat_arr, d_nat_arr = fsrs_engine.update_state_recall(
                            np.array([s_nat]),
                            np.array([d_nat]),
                            np.array([rat], dtype=np.int8),
                            np.array([ret_nat]),
                            nature_params,
                        )
                    s_nat, d_nat = float(s_nat_arr[0]), float(d_nat_arr[0])

                    elapsed_alg = float((ts - last_rev_alg) / np.timedelta64(1, "D"))
                    ret_alg = float(
                        fsrs_engine.predict_retrievability(
                            np.array([s_alg]), np.array([elapsed_alg]), algo_params
                        )[0]
                    )
                    if rat == 1:
                        s_alg_arr, d_alg_arr = fsrs_engine.update_state_forget(
                            np.array([s_alg]),
                            np.array([d_alg]),
                            np.array([ret_alg]),
                            algo_params,
                        )
                    else:
                        s_alg_arr, d_alg_arr = fsrs_engine.update_state_recall(
                            np.array([s_alg]),
                            np.array([d_alg]),
                            np.array([rat], dtype=np.int8),
                            np.array([ret_alg]),
                            algo_params,
                        )
                    s_alg, d_alg = float(s_alg_arr[0]), float(d_alg_arr[0])

                last_rev_nat = ts
                last_rev_alg = ts

            ret_target = 0.9
            interval = int(
                fsrs_engine.next_interval(np.array([s_alg]), ret_target, algo_params)[0]
            )
            due = last_rev_alg + np.timedelta64(interval, "D")

            states_nat[cid] = (s_nat, d_nat, last_rev_nat)
            states_alg[cid] = (s_alg, d_alg, due, last_rev_alg)

        cids = np.array(list(states_nat.keys()), dtype=np.int64)

        deck_t = Deck(
            cids,
            np.array([s[0] for s in states_nat.values()], dtype=np.float64),
            np.array([s[1] for s in states_nat.values()], dtype=np.float64),
            np.full(len(cids), np.datetime64("NaT"), dtype="datetime64[ns]"),
            cast(
                npt.NDArray[np.datetime64],
                np.array([s[2] for s in states_nat.values()], dtype="datetime64[ns]"),
            ),
        )
        deck_s = Deck(
            cids,
            np.array([s[0] for s in states_alg.values()], dtype=np.float64),
            np.array([s[1] for s in states_alg.values()], dtype=np.float64),
            cast(
                npt.NDArray[np.datetime64],
                np.array([s[2] for s in states_alg.values()], dtype="datetime64[ns]"),
            ),
            cast(
                npt.NDArray[np.datetime64],
                np.array([s[3] for s in states_alg.values()], dtype="datetime64[ns]"),
            ),
        )

        return deck_t, deck_s, logs_data, current_date

    empty_logs = LogData.concatenate([])
    empty_ids = np.array([], dtype=np.int64)
    empty_f64 = np.array([], dtype=np.float64)
    empty_ts = np.array([], dtype="datetime64[ns]")
    return (
        Deck(empty_ids, empty_f64, empty_f64, empty_ts, empty_ts),
        Deck(empty_ids, empty_f64, empty_f64, empty_ts, empty_ts),
        empty_logs,
        current_date,
    )
