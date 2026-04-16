from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from .paths import HISTORY_FILE, ensure_app_dirs


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def preview_text(text: str, *, max_words: int = 10, max_chars: int = 72) -> str:
    compact = " ".join(text.split())
    if not compact:
        return "(empty)"

    words = compact.split(" ")
    preview = " ".join(words[:max_words])
    truncated = len(words) > max_words or len(preview) < len(compact)

    if len(preview) > max_chars:
        shortened = preview[:max_chars].rstrip()
        if " " in shortened:
            shortened = shortened.rsplit(" ", 1)[0]
        preview = shortened or preview[:max_chars]
        truncated = True

    if truncated:
        preview = preview.rstrip(" .,;:") + "..."
    return preview


class HistoryStore:
    def __init__(self, max_items: int = 5, path: Path | None = None) -> None:
        self.max_items = max_items
        self.path = path or HISTORY_FILE
        ensure_app_dirs()
        if not self.path.exists():
            self._write([])

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            try:
                data = json.load(handle)
            except json.JSONDecodeError:
                return []
        if not isinstance(data, list):
            return []
        return [x for x in data if isinstance(x, dict)]

    def _write(self, entries: list[dict[str, Any]]) -> None:
        tmp_path = self.path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(entries, handle, ensure_ascii=False, indent=2)
        tmp_path.replace(self.path)

    def list_entries(self) -> list[dict[str, Any]]:
        return self._read()

    def add_entry(self, text: str, language: str = "") -> dict[str, Any]:
        entry = {
            "id": str(uuid4()),
            "text": text,
            "language": language,
            "created_at": _now_iso(),
        }
        entries = self._read()
        entries.insert(0, entry)
        entries = entries[: self.max_items]
        self._write(entries)
        return entry

    def get_by_index(self, index_1_based: int) -> dict[str, Any]:
        entries = self._read()
        if index_1_based < 1 or index_1_based > len(entries):
            raise IndexError("history index out of range")
        return entries[index_1_based - 1]
