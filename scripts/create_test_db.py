import os
import sqlite3
from datetime import datetime, timezone


def create_test_db(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)

    conn = sqlite3.connect(path)
    cur = conn.cursor()

    # col table
    cur.execute(
        "CREATE TABLE col (id integer primary key, crt integer, "
        "mod integer, scm integer, ver integer, dty integer, usn integer, "
        "ls integer, conf text, models text, decks text, dconf text, tags text)"
    )
    cur.execute(
        "INSERT INTO col (id, ver, dty, usn, ls, conf, models, "
        "decks, dconf, tags) VALUES (1, 18, 0, 0, 0, '', '', '', '', '')"
    )

    # decks table (new schema)
    cur.execute(
        "CREATE TABLE decks (id integer primary key, name text, "
        "mtime_secs integer, usn integer, common blob, kind blob)"
    )
    cur.execute(
        "INSERT INTO decks (id, name, mtime_secs, usn, common, kind) "
        "VALUES (1, 'Default', 0, 0, x'', x'')"
    )
    cur.execute(
        "INSERT INTO decks (id, name, mtime_secs, usn, common, kind) "
        "VALUES (2, 'TestDeck', 0, 0, x'', x'')"
    )

    # deck_config table (new schema)
    cur.execute(
        "CREATE TABLE deck_config (id integer primary key, name text, "
        "mtime_secs integer, usn integer, config blob)"
    )
    cur.execute(
        "INSERT INTO deck_config (id, name, mtime_secs, usn, config) "
        "VALUES (1, 'Default', 0, 0, x'')"
    )
    cur.execute(
        "INSERT INTO deck_config (id, name, mtime_secs, usn, config) "
        "VALUES (100, 'TestConfig', 0, 0, x'')"
    )

    # cards table
    cur.execute(
        "CREATE TABLE cards (id integer primary key, nid integer, "
        "did integer, ord integer, mod integer, usn integer, type integer, "
        "queue integer, due integer, ivl integer, factor integer, "
        "reps integer, lapses integer, left integer, odue integer, "
        "odid integer, flags integer, data text)"
    )
    cur.execute(
        "INSERT INTO cards (id, did, nid, ord, mod, usn, type, queue, "
        "due, ivl, factor, reps, lapses, left, odue, odid, flags, data) "
        "VALUES (1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '')"
    )
    cur.execute(
        "INSERT INTO cards (id, did, nid, ord, mod, usn, type, queue, "
        "due, ivl, factor, reps, lapses, left, odue, odid, flags, data) "
        "VALUES (2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '')"
    )

    # revlog table
    cur.execute(
        "CREATE TABLE revlog (id integer primary key, cid integer, "
        "usn integer, ease integer, ivl integer, lastIvl integer, "
        "factor integer, time integer, type integer)"
    )

    # Add some reviews
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    # Card 1 (Default deck) reviews
    cur.execute(
        "INSERT INTO revlog (id, cid, usn, ease, ivl, lastIvl, factor, "
        "time, type) VALUES (?, 1, 0, 3, 0, 0, 0, 5000, 0)",
        (now_ms - 86400000 * 10,),
    )
    cur.execute(
        "INSERT INTO revlog (id, cid, usn, ease, ivl, lastIvl, factor, "
        "time, type) VALUES (?, 1, 0, 3, 0, 0, 0, 6000, 2)",
        (now_ms - 86400000 * 5,),
    )

    # Card 2 (TestDeck) reviews
    cur.execute(
        "INSERT INTO revlog (id, cid, usn, ease, ivl, lastIvl, factor, "
        "time, type) VALUES (?, 2, 0, 3, 0, 0, 0, 7000, 0)",
        (now_ms - 86400000 * 20,),
    )
    cur.execute(
        "INSERT INTO revlog (id, cid, usn, ease, ivl, lastIvl, factor, "
        "time, type) VALUES (?, 2, 0, 1, 0, 0, 0, 8000, 2)",
        (now_ms - 86400000 * 15,),
    )
    cur.execute(
        "INSERT INTO revlog (id, cid, usn, ease, ivl, lastIvl, factor, "
        "time, type) VALUES (?, 2, 0, 3, 0, 0, 0, 9000, 3)",
        (now_ms - 86400000 * 10 + 1000,),
    )

    conn.commit()
    conn.close()
    print(f"Created test database at {path}")


if __name__ == "__main__":
    create_test_db("tests/test_collection.anki2")
