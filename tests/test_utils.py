from src.simulate_fsrs import (
    DEFAULT_PARAMETERS,
    decode_varint,
    get_deck_config_id,
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
    # Tag 0x08 (field 1)
    assert get_deck_config_id(b"\x08\x01") == 1
    assert get_deck_config_id(b"\x08\xac\x02") == 300
    # No tag 0x08 at start
    assert get_deck_config_id(b"\x10\x01") is None
    # Empty or short
    assert get_deck_config_id(b"") is None
    assert get_deck_config_id(b"\x08") is None


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
