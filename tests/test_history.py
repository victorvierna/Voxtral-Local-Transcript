from pathlib import Path

from voxtray.history import HistoryStore, preview_text


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


def test_preview_text_prefers_first_words() -> None:
    text = "Vale tiene muy buena pinta pero quiero mejorar bastante esto ahora mismo"

    preview = preview_text(text, max_words=6, max_chars=200)

    assert preview == "Vale tiene muy buena pinta pero..."


def test_preview_text_normalizes_whitespace_and_caps_length() -> None:
    text = "  uno   dos \n tres   cuatro cinco seis siete ocho nueve diez once  "

    preview = preview_text(text, max_words=20, max_chars=24)

    assert preview == "uno dos tres cuatro..."
