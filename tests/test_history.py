from pathlib import Path

from voxtray.history import HistoryStore


def test_history_keeps_latest_max_entries(tmp_path: Path):
    path = tmp_path / "history.json"
    store = HistoryStore(max_items=5, path=path)

    for i in range(7):
        store.add_entry(f"item-{i}")

    entries = store.list_entries()
    assert len(entries) == 5
    assert entries[0]["text"] == "item-6"
    assert entries[-1]["text"] == "item-2"


def test_get_by_index(tmp_path: Path):
    path = tmp_path / "history.json"
    store = HistoryStore(max_items=5, path=path)
    store.add_entry("one")
    store.add_entry("two")

    assert store.get_by_index(1)["text"] == "two"
    assert store.get_by_index(2)["text"] == "one"
