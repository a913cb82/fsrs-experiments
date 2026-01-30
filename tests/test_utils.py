import numpy as np
from fsrs.scheduler import DEFAULT_PARAMETERS

from src.anki_utils import (
    decode_varint,
    get_deck_config_id,
    get_field_from_proto,
    infer_review_weights,
)
from src.utils import (
    calculate_metrics,
    calculate_population_retrievability,
    get_retention_for_day,
    parse_parameters,
    parse_retention_schedule,
)


def test_decode_varint() -> None:
    # 1 byte
    assert decode_varint(b"\x01", 0) == (1, 1)
    assert decode_varint(b"\x7f", 0) == (127, 1)
    # 2 bytes
    assert decode_varint(b"\x80\x01", 0) == (128, 2)
    assert decode_varint(b"\xac\x02", 0) == (300, 2)
    # Large value
    data = b"\xfd\xff\xff\xff\x07"
    val, pos = decode_varint(data, 0)
    assert val == 2147483645
    assert pos == 5


def test_get_deck_config_id() -> None:
    # 1. From common_blob (field 1)
    assert get_deck_config_id(b"\x08\x01", b"") == 1
    assert get_deck_config_id(b"\x08\xac\x02", b"") == 300

    # 2. From kind_blob (field 1 -> field 1)
    # 0a 0d 08 fa e3 9c eb d1 32 ... (NormalDeck wrapping config_id)
    # 0a: tag 1, type 2. 0d: length 13. 08: tag 1, type 0.
    kind_with_cid = b"\x0a\x0d\x08\xfa\xe3\x9c\xeb\xd1\x32\x10\x01\x18\x0b\x30\x02"
    assert get_deck_config_id(b"", kind_with_cid) == 1739955057146

    # 3. Default to 1
    assert get_deck_config_id(b"", b"") == 1
    assert get_deck_config_id(b"\x10\x01", b"\x10\x01") == 1


def test_get_field_from_proto() -> None:
    # Field 1, value 1
    assert get_field_from_proto(b"\x08\x01", 1) == 1
    # Field 2, skip it and find field 1
    assert get_field_from_proto(b"\x10\x05\x08\x01", 1) == 1
    # Field not found
    assert get_field_from_proto(b"\x10\x05", 1) is None
    # Empty
    assert get_field_from_proto(b"", 1) is None
    # Wire type skip tests (fixed64 = 1, length delimited = 2, fixed32 = 5)
    # fixed64 field 2 (tag 0x11), then field 1 (tag 0x08)
    data_fixed64 = b"\x11\x00\x00\x00\x00\x00\x00\x00\x00\x08\x01"
    assert get_field_from_proto(data_fixed64, 1) == 1
    # fixed32 field 2 (tag 0x15), then field 1
    data_fixed32 = b"\x15\x00\x00\x00\x00\x08\x01"
    assert get_field_from_proto(data_fixed32, 1) == 1


def test_parse_retention_schedule() -> None:
    assert parse_retention_schedule("0.9") == [(1, 0.9)]
    assert parse_retention_schedule("2:0.7,1:0.9") == [(2, 0.7), (1, 0.9)]
    # Invalid
    assert parse_retention_schedule("invalid") == [(1, 0.9)]
    assert parse_retention_schedule("2:invalid") == [(1, 0.9)]


def test_get_retention_for_day() -> None:
    sched = [(2, 0.7), (1, 0.9)]  # Total 3 days
    assert get_retention_for_day(0, sched) == 0.7
    assert get_retention_for_day(1, sched) == 0.7
    assert get_retention_for_day(2, sched) == 0.9
    # Cycle
    assert get_retention_for_day(3, sched) == 0.7
    assert get_retention_for_day(4, sched) == 0.7
    assert get_retention_for_day(5, sched) == 0.9

    # Single element
    assert get_retention_for_day(10, [(1, 0.8)]) == 0.8


def test_parse_parameters() -> None:
    p_str = ",".join(["0.1"] * 21)
    parsed = parse_parameters(p_str)
    assert len(parsed) == 21
    assert parsed[0] == 0.1

    # Invalid length
    assert parse_parameters("0.1,0.2") == tuple(DEFAULT_PARAMETERS)
    # Invalid values
    assert parse_parameters("a,b,c") == tuple(DEFAULT_PARAMETERS)
    # Empty
    assert parse_parameters("") == tuple(DEFAULT_PARAMETERS)


def test_infer_review_weights() -> None:
    from datetime import datetime, timedelta, timezone

    from fsrs import Rating, ReviewLog

    now = datetime.now(timezone.utc)
    # Card 1: 1st Good, 2nd Good
    # Card 2: 1st Again, 2nd Hard, 3rd Easy
    logs = {
        1: [
            ReviewLog(1, Rating.Good, now - timedelta(days=10), None),
            ReviewLog(1, Rating.Good, now - timedelta(days=5), None),
        ],
        2: [
            ReviewLog(2, Rating.Again, now - timedelta(days=20), None),
            ReviewLog(2, Rating.Hard, now - timedelta(days=15), None),
            ReviewLog(2, Rating.Easy, now - timedelta(days=10), None),
        ],
    }

    weights = infer_review_weights(logs)

    # First ratings: 1 Good (pos 3), 1 Again (pos 1) -> 50% each
    assert weights["first"][0] == 0.5  # Again
    assert weights["first"][1] == 0.0  # Hard
    assert weights["first"][2] == 0.5  # Good
    assert weights["first"][3] == 0.0  # Easy

    # Subsequent Success (recalled) ratings:
    # Card 1: Good (pos 2)
    # Card 2: Hard (pos 1), Easy (pos 3)
    # Total: 1 Hard, 1 Good, 1 Easy -> 33.3% each
    assert abs(weights["success"][0] - 0.3333) < 0.001  # Hard
    assert abs(weights["success"][1] - 0.3333) < 0.001  # Good
    assert abs(weights["success"][2] - 0.3333) < 0.001  # Easy


def test_calculate_population_retrievability() -> None:
    t = np.linspace(0, 10, 11)
    stabilities = np.array([1.0, 5.0, 10.0])
    params = [0.1] * 21

    res = calculate_population_retrievability(t, stabilities, params)
    assert len(res) == 11
    assert res[0] == 1.0
    assert np.all(res <= 1.0)
    assert np.all(res >= 0.0)

    assert np.all(calculate_population_retrievability(t, np.array([]), params) == 1.0)


def test_calculate_metrics() -> None:
    gt_params = [0.1] * 21
    fit_params = [0.1] * 21
    stabilities = [(1.0, 1.0), (5.0, 5.0)]

    rmse, kl = calculate_metrics(gt_params, fit_params, stabilities)
    assert rmse == 0.0
    assert kl == 0.0

    fit_params_diff = list(gt_params)
    fit_params_diff[20] = 0.5
    rmse, kl = calculate_metrics(gt_params, fit_params_diff, stabilities)
    assert rmse > 0
    assert kl > 0

    assert calculate_metrics(gt_params, fit_params, []) == (0.0, 0.0)
