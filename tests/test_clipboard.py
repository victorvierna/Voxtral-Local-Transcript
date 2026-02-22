import pytest

import voxtray.clipboard as clipboard


def test_auto_prefers_clip_exe_on_wsl(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[list[str], str]] = []

    monkeypatch.setattr(clipboard, "_copy_with_qt", lambda text: False)
    monkeypatch.setattr(clipboard, "_running_in_wsl", lambda: True)
    monkeypatch.setattr(
        clipboard,
        "which",
        lambda cmd: "/usr/bin/clip.exe" if cmd == "clip.exe" else None,
    )
    monkeypatch.setattr(
        clipboard, "_copy_with_cmd", lambda cmd, text: calls.append((cmd, text))
    )

    backend = clipboard.copy_to_clipboard("hola", backend="auto")

    assert backend == "clip.exe"
    assert calls == [(["clip.exe"], "hola")]


def test_auto_falls_back_to_xclip_when_not_wsl(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[list[str], str]] = []

    monkeypatch.setattr(clipboard, "_copy_with_qt", lambda text: False)
    monkeypatch.setattr(clipboard, "_running_in_wsl", lambda: False)
    monkeypatch.setattr(
        clipboard,
        "which",
        lambda cmd: "/usr/bin/xclip" if cmd == "xclip" else None,
    )
    monkeypatch.setattr(
        clipboard, "_copy_with_cmd", lambda cmd, text: calls.append((cmd, text))
    )

    backend = clipboard.copy_to_clipboard("hello", backend="auto")

    assert backend == "xclip"
    assert calls == [(["xclip", "-selection", "clipboard"], "hello")]


def test_unsupported_backend_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(clipboard, "_copy_with_qt", lambda text: False)
    monkeypatch.setattr(clipboard, "_running_in_wsl", lambda: False)
    monkeypatch.setattr(clipboard, "which", lambda cmd: None)

    with pytest.raises(clipboard.ClipboardError, match="unsupported clipboard backend"):
        clipboard.copy_to_clipboard("x", backend="made-up-backend")
