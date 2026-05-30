from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from types import SimpleNamespace

from voxtray.assistant_hook import route_text


class RouteHandler(BaseHTTPRequestHandler):
    response = {"action": "agent", "command_id": "cmd_1", "agent_id": "Harvis", "message": "Procesando"}
    seen = {}

    def log_message(self, format, *args):  # type: ignore[no-untyped-def]
        return

    def do_POST(self):  # type: ignore[no-untyped-def]
        size = int(self.headers.get("Content-Length") or "0")
        RouteHandler.seen = {
            "path": self.path,
            "auth": self.headers.get("Authorization"),
            "token": self.headers.get("X-Harvis-Token"),
            "body": json.loads(self.rfile.read(size).decode("utf-8")),
        }
        body = json.dumps(RouteHandler.response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _config(endpoint: str, *, fail_open: bool = True):
    return SimpleNamespace(
        assistant=SimpleNamespace(
            enabled=True,
            endpoint=endpoint,
            token_env="HARVIS_TEST_TOKEN",
            token_file="",
            timeout_seconds=1.0,
            fail_open_to_clipboard=fail_open,
        )
    )


def test_route_text_posts_to_harvis_with_token(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), RouteHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("HARVIS_TEST_TOKEN", "test-token")
    endpoint = f"http://127.0.0.1:{server.server_port}/api/route"

    try:
        route = route_text(
            "Harvis, revisa",
            artifact_path="/tmp/artifact",
            history_id="h1",
            provider="openai_realtime",
            config=_config(endpoint),
        )
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert route.action == "agent"
    assert route.command_id == "cmd_1"
    assert route.agent_id == "Harvis"
    assert RouteHandler.seen["path"] == "/api/route"
    assert RouteHandler.seen["auth"] == "Bearer test-token"
    assert RouteHandler.seen["token"] == "test-token"
    assert RouteHandler.seen["body"]["artifact_path"] == "/tmp/artifact"
    assert RouteHandler.seen["body"]["provider"] == "openai_realtime"


def test_route_text_can_read_token_file(tmp_path, monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), RouteHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.delenv("HARVIS_TEST_TOKEN", raising=False)
    token_path = tmp_path / "harvis.token"
    token_path.write_text("file-token\n", encoding="utf-8")
    endpoint = f"http://127.0.0.1:{server.server_port}/api/route"
    config = _config(endpoint)
    config.assistant.token_file = str(token_path)

    try:
        route = route_text("Harvis, revisa", config=config)
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert route.action == "agent"
    assert RouteHandler.seen["auth"] == "Bearer file-token"


def test_route_text_fails_open_when_harvis_is_down(monkeypatch):
    monkeypatch.setenv("HARVIS_TEST_TOKEN", "test-token")
    route = route_text("texto normal", config=_config("http://127.0.0.1:9/api/route"))

    assert route.action == "clipboard"
    assert "assistant unavailable" in route.error


def test_route_text_does_not_fail_open_for_explicit_harvis_command(monkeypatch):
    monkeypatch.setenv("HARVIS_TEST_TOKEN", "test-token")

    route = route_text(
        "Harvis, manda un correo a user@example.test",
        config=_config("http://127.0.0.1:9/api/route"),
    )

    assert route.action == "error"
    assert "assistant unavailable" in route.error


def test_route_text_does_not_copy_explicit_command_when_disabled():
    config = SimpleNamespace(assistant=SimpleNamespace(enabled=False))

    route = route_text("Jarvis, revisa el correo", config=config)

    assert route.action == "error"
    assert "assistant disabled" in route.error


def test_route_text_can_disable_fail_open(monkeypatch):
    monkeypatch.delenv("HARVIS_TEST_TOKEN", raising=False)
    route = route_text(
        "Harvis, revisa",
        config=_config("http://127.0.0.1:9/api/route", fail_open=False),
    )

    assert route.action == "error"
    assert "HARVIS_TEST_TOKEN" in route.error


def test_route_text_disabled_returns_clipboard():
    config = SimpleNamespace(assistant=SimpleNamespace(enabled=False))

    route = route_text("texto normal", config=config)

    assert route.action == "clipboard"
