from datetime import datetime

from src.simulate_fsrs import load_anki_history

TEST_DB = "tests/test_collection.anki2"


def test_load_all_history() -> None:
    # Load everything (no filters)
    logs, last_rev = load_anki_history(TEST_DB)

    # Check that we loaded both cards
    assert len(logs) == 2
    # Card 1 has 2 reviews, Card 2 has 3 reviews
    assert len(logs[1]) == 2
    assert len(logs[2]) == 3
    assert isinstance(last_rev, datetime)


def test_load_by_deck_name() -> None:
    # Load only 'TestDeck' (Card 2)
    logs, _ = load_anki_history(TEST_DB, deck_name="TestDeck")

    assert len(logs) == 1
    assert 2 in logs
    assert 1 not in logs
    assert len(logs[2]) == 3


def test_load_non_existent_deck() -> None:
    logs, _ = load_anki_history(TEST_DB, deck_name="NonExistent")
    assert len(logs) == 0


def test_load_invalid_db_path() -> None:
    logs, _ = load_anki_history("non_existent.anki2")
    assert len(logs) == 0
