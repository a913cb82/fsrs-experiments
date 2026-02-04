import json
import sqlite3
from typing import Any

import pytest

from src.anki_utils import START_DATE, load_anki_history

TEST_DB = "tests/test_collection.anki2"


def test_load_all_history() -> None:
    logs, last_rev = load_anki_history(TEST_DB)
    assert not logs.empty
    assert logs["card_id"].nunique() == 2
    assert last_rev > START_DATE


def test_load_by_deck_name() -> None:
    logs, _ = load_anki_history(TEST_DB, deck_name="TestDeck")
    assert not logs.empty
    assert logs["card_id"].nunique() == 1


def test_load_by_config_name() -> None:
    logs, _ = load_anki_history(TEST_DB, deck_config_name="Default")
    assert not logs.empty
    assert logs["card_id"].nunique() == 2


def test_load_non_existent_config() -> None:
    # This should raise SystemExit because of _raise_informative_config_error
    with pytest.raises(SystemExit):
        load_anki_history(TEST_DB, deck_config_name="NonExistent")


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
    assert not logs.empty
    assert logs["card_id"].nunique() == 1
    assert 10 in logs["card_id"].values


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
    # 0x08 0x64 is protobuf for field 1 = 100
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
    assert not logs.empty
    assert logs["card_id"].nunique() == 1
    assert 10 in logs["card_id"].values


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

    # MUST create cards table for informative error logic
    cur.execute("CREATE TABLE cards (id integer primary key, did integer)")
    cur.execute("INSERT INTO cards (id, did) VALUES (1, 1)")

    # MUST create revlog table for other branches
    sql_rev = (
        "CREATE TABLE revlog (id integer primary key, cid integer, "
        "ease integer, type integer, time integer)"
    )
    cur.execute(sql_rev)

    conn.commit()
    conn.close()

    # Trigger config not found error (SystemExit)
    with pytest.raises(SystemExit) as cm:
        load_anki_history(db_path, deck_config_name="NonExistentConfig")
    assert cm.value.code == 1

    # Trigger deck not found warning (should still return empty logs, but it warns)
    logs, _ = load_anki_history(db_path, deck_name="NonExistentDeck")
    assert logs.empty


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
    assert logs.empty
