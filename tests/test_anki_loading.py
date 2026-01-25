import json
import sqlite3
from datetime import datetime
from typing import Any

from src.simulate_fsrs import load_anki_history

TEST_DB = "tests/test_collection.anki2"


def test_load_all_history() -> None:
    logs, last_rev = load_anki_history(TEST_DB)
    assert len(logs) == 2
    assert len(logs[1]) == 2
    assert len(logs[2]) == 3
    assert isinstance(last_rev, datetime)


def test_load_by_deck_name() -> None:
    logs, _ = load_anki_history(TEST_DB, deck_name="TestDeck")
    assert len(logs) == 1
    assert 2 in logs
    assert len(logs[2]) == 3


def test_load_non_existent_deck() -> None:
    logs, _ = load_anki_history(TEST_DB, deck_name="NonExistent")
    assert len(logs) == 0


def test_load_invalid_db_path() -> None:
    logs, _ = load_anki_history("non_existent.anki2")
    assert len(logs) == 0


def test_load_old_json_schema(tmp_path: Any) -> None:
    db_path = str(tmp_path / "old_collection.anki2")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    sql = (
        "CREATE TABLE col (id integer primary key, ver integer, decks text, dconf text)"
    )
    cur.execute(sql)

    decks = {
        "1": {"id": 1, "name": "Default", "conf": 1},
        "2": {"id": 2, "name": "OldDeck", "conf": 100},
    }
    dconf = {"1": {"id": 1, "name": "Default"}, "100": {"id": 100, "name": "OldConfig"}}

    cur.execute(
        "INSERT INTO col (id, ver, decks, dconf) VALUES (1, 11, ?, ?)",
        (json.dumps(decks), json.dumps(dconf)),
    )

    cur.execute("CREATE TABLE cards (id integer primary key, did integer)")
    cur.execute("INSERT INTO cards (id, did) VALUES (10, 2)")

    sql_rev = (
        "CREATE TABLE revlog (id integer primary key, cid integer, "
        "ease integer, type integer)"
    )
    cur.execute(sql_rev)
    cur.execute("INSERT INTO revlog (id, cid, ease, type) VALUES (1000000, 10, 3, 0)")

    conn.commit()
    conn.close()

    logs, _ = load_anki_history(db_path, deck_config_name="OldConfig")
    assert len(logs) == 1
    assert 10 in logs

    logs, _ = load_anki_history(db_path, deck_name="OldDeck")
    assert len(logs) == 1
    assert 10 in logs


def test_relational_inheritance(tmp_path: Any) -> None:
    db_path = str(tmp_path / "rel_inheritance.anki2")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("CREATE TABLE col (id integer primary key, ver integer)")
    cur.execute("INSERT INTO col (id, ver) VALUES (1, 18)")

    cur.execute("CREATE TABLE decks (id integer primary key, name text, common blob)")
    # Parent has config 100
    cur.execute("INSERT INTO decks (id, name, common) VALUES (1, 'Parent', x'0864')")
    # Child has NO config (should inherit from Parent)
    cur.execute("INSERT INTO decks (id, name, common) VALUES (2, 'Parent::Child', x'')")

    cur.execute("CREATE TABLE deck_config (id integer primary key, name text)")
    cur.execute("INSERT INTO deck_config (id, name) VALUES (100, 'InheritedConfig')")

    cur.execute("CREATE TABLE cards (id integer primary key, did integer)")
    cur.execute("INSERT INTO cards (id, did) VALUES (10, 2)")

    sql_rev = (
        "CREATE TABLE revlog (id integer primary key, cid integer, "
        "ease integer, type integer)"
    )
    cur.execute(sql_rev)
    cur.execute("INSERT INTO revlog (id, cid, ease, type) VALUES (1000000, 10, 3, 0)")

    conn.commit()
    conn.close()

    logs, _ = load_anki_history(db_path, deck_config_name="InheritedConfig")
    assert len(logs) == 1
    assert 10 in logs


def test_load_anki_history_warnings(tmp_path: Any) -> None:
    db_path = str(tmp_path / "warning_collection.anki2")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("CREATE TABLE col (id integer primary key, ver integer)")
    cur.execute("INSERT INTO col (id, ver) VALUES (1, 18)")

    cur.execute("CREATE TABLE decks (id integer primary key, name text, common blob)")
    cur.execute("INSERT INTO decks (id, name, common) VALUES (1, 'TestDeck', x'0801')")

    cur.execute("CREATE TABLE deck_config (id integer primary key, name text)")
    cur.execute("INSERT INTO deck_config (id, name) VALUES (1, 'Default')")

    conn.commit()
    conn.close()

    logs, _ = load_anki_history(db_path, deck_config_name="NonExistentConfig")
    assert logs == {}

    logs, _ = load_anki_history(db_path, deck_name="NonExistentDeck")
    assert logs == {}


def test_load_anki_history_malformed_json(tmp_path: Any) -> None:
    db_path = str(tmp_path / "malformed_collection.anki2")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    sql = (
        "CREATE TABLE col (id integer primary key, ver integer, decks text, dconf text)"
    )
    cur.execute(sql)
    cur.execute(
        "INSERT INTO col (id, ver, decks, dconf) VALUES (1, 11, 'invalid json', '{}}')"
    )
    conn.commit()
    conn.close()

    logs, _ = load_anki_history(db_path)
    assert logs == {}


def test_weight_inference_from_test_db() -> None:
    from src.simulate_fsrs import infer_review_weights

    logs, _ = load_anki_history(TEST_DB)
    weights = infer_review_weights(logs)

    # Based on scripts/create_test_db.py:
    # Card 1: [3, 3] -> First: 3, Success: 3
    # Card 2: [3, 1, 3] -> First: 3, Success: 3 (1 is Again/Failure)
    # Total First: [0, 0, 2, 0] -> Good=1.0
    # Total Success: [0, 2, 0] -> Good=1.0

    assert weights["first"][2] == 1.0
    assert weights["success"][1] == 1.0
    assert weights["first"][0] == 0.0
    assert weights["success"][0] == 0.0
