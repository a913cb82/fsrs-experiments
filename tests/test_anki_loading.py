import json
import sqlite3
from typing import Any

import numpy as np
import pytest

from src.anki_utils import load_anki_history

# Mock Anki DB path
TEST_DB = "tests/test_collection.anki2"


def test_load_all_history() -> None:
    logs, last_rev = load_anki_history(TEST_DB)
    assert not logs.is_empty
    assert len(np.unique(logs.card_ids)) == 2
    assert len(logs.ratings) == 5


def test_load_by_deck_name() -> None:
    logs, _ = load_anki_history(TEST_DB, deck_name="TestDeck")
    assert not logs.is_empty
    assert len(np.unique(logs.card_ids)) == 1
    assert len(logs.ratings) == 3


def test_load_by_config_name() -> None:
    logs, _ = load_anki_history(TEST_DB, deck_config_name="Default")
    assert not logs.is_empty
    assert len(np.unique(logs.card_ids)) == 2
    assert len(logs.ratings) == 5


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

    cur.execute(
        "CREATE TABLE revlog (id integer primary key, cid integer, "
        "ease integer, type integer, time integer)"
    )
    cur.execute(
        "INSERT INTO revlog (id, cid, ease, type, time) "
        "VALUES (1000000, 10, 3, 0, 5000)"
    )

    conn.commit()
    conn.close()

    logs, _ = load_anki_history(db_path, deck_config_name="OldConfig")
    assert not logs.is_empty
    assert len(np.unique(logs.card_ids)) == 1
    assert len(logs.ratings) == 1


def test_relational_inheritance(tmp_path: Any) -> None:
    db_path = str(tmp_path / "rel_inheritance.anki2")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("CREATE TABLE col (id integer primary key, ver integer)")
    cur.execute("INSERT INTO col (id, ver) VALUES (1, 18)")

    cur.execute(
        "CREATE TABLE decks (id integer primary key, name text, common blob, kind blob)"
    )
    # Parent has config 100 in common blob (field 1)
    cur.execute(
        "INSERT INTO decks (id, name, common, kind) VALUES (1, 'Parent', x'0864', x'')"
    )
    # Child has NO config (should inherit from Parent)
    cur.execute(
        "INSERT INTO decks (id, name, common, kind) "
        "VALUES (2, 'Parent::Child', x'', x'')"
    )

    cur.execute("CREATE TABLE deck_config (id integer primary key, name text)")
    cur.execute("INSERT INTO deck_config (id, name) VALUES (100, 'InheritedConfig')")

    cur.execute("CREATE TABLE cards (id integer primary key, did integer)")
    cur.execute("INSERT INTO cards (id, did) VALUES (10, 2)")

    sql_rev = (
        "CREATE TABLE revlog (id integer primary key, cid integer, "
        "ease integer, type integer, time integer)"
    )
    cur.execute(sql_rev)
    cur.execute(
        "INSERT INTO revlog (id, cid, ease, type, time) "
        "VALUES (1000000, 10, 3, 0, 5000)"
    )

    conn.commit()
    conn.close()

    logs, _ = load_anki_history(db_path, deck_config_name="InheritedConfig")
    assert not logs.is_empty
    assert len(np.unique(logs.card_ids)) == 1
    assert len(logs.ratings) == 1


def test_load_anki_history_warnings(tmp_path: Any) -> None:
    db_path = str(tmp_path / "warning_collection.anki2")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("CREATE TABLE col (id integer primary key, ver integer)")
    cur.execute("INSERT INTO col (id, ver) VALUES (1, 18)")

    cur.execute(
        "CREATE TABLE decks (id integer primary key, name text, common blob, kind blob)"
    )
    cur.execute(
        "INSERT INTO decks (id, name, common, kind) "
        "VALUES (1, 'TestDeck', x'0801', x'')"
    )

    cur.execute("CREATE TABLE deck_config (id integer primary key, name text)")
    cur.execute("INSERT INTO deck_config (id, name) VALUES (1, 'Default')")

    cur.execute("CREATE TABLE cards (id integer primary key, did integer)")
    cur.execute("INSERT INTO cards (id, did) VALUES (1, 1)")

    sql_rev = (
        "CREATE TABLE revlog (id integer primary key, cid integer, "
        "ease integer, type integer, time integer)"
    )
    cur.execute(sql_rev)

    conn.commit()
    conn.close()

    with pytest.raises(SystemExit) as cm:
        load_anki_history(db_path, deck_config_name="NonExistentConfig")
    assert cm.value.code == 1

    logs, _ = load_anki_history(db_path, deck_name="NonExistentDeck")
    assert logs.is_empty


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
    assert logs.is_empty
