import pytest
import subprocess

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
    calls: list[str] = []

    monkeypatch.setattr(clipboard, "_copy_with_qt", lambda text: False)
    monkeypatch.setattr(clipboard, "_running_in_wsl", lambda: False)
    monkeypatch.setattr(
        clipboard,
        "which",
        lambda cmd: "/usr/bin/xclip" if cmd == "xclip" else None,
    )
    monkeypatch.setattr(clipboard, "_copy_with_xclip", lambda text: calls.append(text))

    backend = clipboard.copy_to_clipboard("hello", backend="auto")

    assert backend == "xclip"
    assert calls == ["hello"]


def test_auto_falls_back_to_xsel_when_xclip_fails(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(clipboard, "_copy_with_qt", lambda text: False)
    monkeypatch.setattr(clipboard, "_running_in_wsl", lambda: False)
    monkeypatch.setattr(
        clipboard,
        "which",
        lambda cmd: f"/usr/bin/{cmd}" if cmd in {"xclip", "xsel"} else None,
    )

    def fail_xclip(text: str) -> None:
        calls.append(("xclip", text))
        raise clipboard.ClipboardError("xclip exited with status 1")

    def fake_copy_with_cmd(command, text, **kwargs):
        del kwargs
        calls.append(("cmd", command, text))

    monkeypatch.setattr(clipboard, "_copy_with_xclip", fail_xclip)
    monkeypatch.setattr(clipboard, "_copy_with_cmd", fake_copy_with_cmd)

    backend = clipboard.copy_to_clipboard("hello", backend="auto")

    assert backend == "xsel"
    assert calls == [
        ("xclip", "hello"),
        ("cmd", ["xsel", "--clipboard", "--input"], "hello"),
    ]


def test_auto_reports_backend_failures(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(clipboard, "_copy_with_qt", lambda text: False)
    monkeypatch.setattr(clipboard, "_running_in_wsl", lambda: False)
    monkeypatch.setattr(
        clipboard,
        "which",
        lambda cmd: f"/usr/bin/{cmd}" if cmd in {"xclip", "xsel"} else None,
    )
    monkeypatch.setattr(
        clipboard,
        "_copy_with_xclip",
        lambda text: (_ for _ in ()).throw(clipboard.ClipboardError("xclip failed")),
    )

    def fail_copy_with_cmd(command, text, **kwargs):
        del command, text, kwargs
        raise clipboard.ClipboardError("xsel failed")

    monkeypatch.setattr(clipboard, "_copy_with_cmd", fail_copy_with_cmd)

    with pytest.raises(clipboard.ClipboardError, match="clipboard backends failed"):
        clipboard.copy_to_clipboard("hello", backend="auto")


def test_copy_with_cmd_times_out(monkeypatch: pytest.MonkeyPatch):
    def timeout_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(clipboard.subprocess, "run", timeout_run)

    with pytest.raises(clipboard.ClipboardError, match="timed out"):
        clipboard._copy_with_cmd(["tool"], "text", timeout_seconds=0.1)


def test_copy_with_xclip_returns_when_owner_keeps_running(monkeypatch: pytest.MonkeyPatch):
    calls: list[object] = []

    class FakeStdin:
        def write(self, data: bytes) -> None:
            calls.append(data)

        def close(self) -> None:
            calls.append("closed")

    class FakeProcess:
        stdin = FakeStdin()

        def wait(self, timeout: float) -> int:
            calls.append(("wait", timeout))
            raise subprocess.TimeoutExpired(cmd="xclip", timeout=timeout)

        def poll(self):
            return None

        def kill(self) -> None:
            calls.append("killed")

    def fake_popen(command, **kwargs):
        calls.append(command)
        calls.append(kwargs["start_new_session"])
        return FakeProcess()

    monkeypatch.setattr(clipboard.subprocess, "Popen", fake_popen)

    clipboard._copy_with_xclip("hola")

    assert calls == [
        ["xclip", "-selection", "clipboard"],
        True,
        b"hola",
        "closed",
        ("wait", clipboard.XCLIP_START_TIMEOUT_SECONDS),
    ]


def test_copy_with_xclip_wraps_pipe_errors(monkeypatch: pytest.MonkeyPatch):
    calls: list[object] = []

    class FakeStdin:
        def write(self, data: bytes) -> None:
            calls.append(data)
            raise BrokenPipeError("no usable display")

        def close(self) -> None:
            calls.append("closed")

    class FakeProcess:
        stdin = FakeStdin()
        stderr = None

        def poll(self):
            return None

        def kill(self) -> None:
            calls.append("killed")

    def fake_popen(command, **kwargs):
        calls.append(command)
        calls.append(kwargs["start_new_session"])
        return FakeProcess()

    monkeypatch.setattr(clipboard.subprocess, "Popen", fake_popen)

    with pytest.raises(clipboard.ClipboardError, match="xclip pipe failed"):
        clipboard._copy_with_xclip("hola")

    assert calls == [
        ["xclip", "-selection", "clipboard"],
        True,
        b"hola",
        "killed",
    ]


def test_verify_clipboard_text_reads_xclip(monkeypatch: pytest.MonkeyPatch):
    calls: list[list[str]] = []

    monkeypatch.setattr(
        clipboard,
        "which",
        lambda cmd: "/usr/bin/xclip" if cmd == "xclip" else None,
    )

    def fake_read(command, **kwargs):
        del kwargs
        calls.append(command)
        return "hola"

    monkeypatch.setattr(clipboard, "_read_with_cmd", fake_read)

    assert clipboard.verify_clipboard_text("hola", "xclip") is True
    assert calls == [["xclip", "-selection", "clipboard", "-out"]]


def test_verify_clipboard_text_detects_mismatch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(clipboard, "CLIPBOARD_VERIFY_ATTEMPTS", 1)
    monkeypatch.setattr(
        clipboard,
        "which",
        lambda cmd: "/usr/bin/xclip" if cmd == "xclip" else None,
    )
    monkeypatch.setattr(clipboard, "_read_with_cmd", lambda command, **kwargs: "otro")

    assert clipboard.verify_clipboard_text("hola", "xclip") is False


def test_verify_clipboard_text_returns_false_on_read_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        clipboard,
        "which",
        lambda cmd: "/usr/bin/xclip" if cmd == "xclip" else None,
    )

    def fail_read(command, **kwargs):
        del command, kwargs
        raise clipboard.ClipboardError("cannot read clipboard")

    monkeypatch.setattr(clipboard, "_read_with_cmd", fail_read)

    assert clipboard.verify_clipboard_text("hola", "xclip") is False


def test_verify_clipboard_text_returns_none_for_unsupported_backend():
    assert clipboard.verify_clipboard_text("hola", "clip.exe") is None


def test_unsupported_backend_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(clipboard, "_copy_with_qt", lambda text: False)
    monkeypatch.setattr(clipboard, "_running_in_wsl", lambda: False)
    monkeypatch.setattr(clipboard, "which", lambda cmd: None)

    with pytest.raises(clipboard.ClipboardError, match="unsupported clipboard backend"):
        clipboard.copy_to_clipboard("x", backend="made-up-backend")
