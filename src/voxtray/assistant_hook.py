from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib import error, request


@dataclass(frozen=True, slots=True)
class AssistantRoute:
    action: str
    message: str = ""
    command_id: str = ""
    agent_id: str = ""
    context_id: str = ""
    confirmation_id: str = ""
    run_id: str = ""
    risk_level: str = ""
    error: str = ""


def _assistant_config(config: Any) -> Any:
    return getattr(config, "assistant", None)


_ASSISTANT_COMMAND_RE = re.compile(
    r"^\s*(harvis|jarvis|harbis|hardisk|agente|ordenador)\b",
    re.IGNORECASE,
)


def _looks_like_assistant_command(text: str) -> bool:
    return bool(_ASSISTANT_COMMAND_RE.search(text or ""))


def _fallback_or_error(
    assistant: Any,
    message: str,
    *,
    assistant_command: bool = False,
) -> AssistantRoute:
    if not assistant_command and bool(getattr(assistant, "fail_open_to_clipboard", True)):
        return AssistantRoute(action="clipboard", message="clipboard", error=message)
    return AssistantRoute(action="error", message=message, error=message)


def _token_from_file(path: str) -> str:
    if not path:
        return ""
    try:
        return Path(path).expanduser().read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def route_text(
    text: str,
    *,
    artifact_path: str = "",
    history_id: str = "",
    provider: str = "",
    source: str = "voxtray",
    config: Any,
) -> AssistantRoute:
    assistant_command = _looks_like_assistant_command(text)
    assistant = _assistant_config(config)
    if assistant is None or not bool(getattr(assistant, "enabled", False)):
        if assistant_command:
            return AssistantRoute(
                action="error",
                message="assistant disabled for explicit Harvis command",
                error="assistant disabled for explicit Harvis command",
            )
        return AssistantRoute(action="clipboard", message="assistant disabled")

    token_env = str(getattr(assistant, "token_env", "HARVIS_API_TOKEN") or "HARVIS_API_TOKEN")
    token = os.environ.get(token_env, "")
    if not token:
        token = _token_from_file(str(getattr(assistant, "token_file", "") or ""))
    if not token:
        return _fallback_or_error(
            assistant,
            f"{token_env} is not set",
            assistant_command=assistant_command,
        )

    payload = {
        "source": source,
        "text": text,
        "artifact_path": artifact_path,
        "history_id": history_id,
        "provider": provider,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    endpoint = str(getattr(assistant, "endpoint", "") or "")
    if not endpoint:
        return _fallback_or_error(
            assistant,
            "assistant endpoint is empty",
            assistant_command=assistant_command,
        )

    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "X-Harvis-Token": token,
        },
        method="POST",
    )
    try:
        with request.urlopen(
            req,
            timeout=float(getattr(assistant, "timeout_seconds", 1.5) or 1.5),
        ) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        return _fallback_or_error(
            assistant,
            f"assistant HTTP {exc.code}",
            assistant_command=assistant_command,
        )
    except OSError as exc:
        return _fallback_or_error(
            assistant,
            f"assistant unavailable: {exc}",
            assistant_command=assistant_command,
        )

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _fallback_or_error(
            assistant,
            "assistant returned invalid JSON",
            assistant_command=assistant_command,
        )
    if not isinstance(data, dict):
        return _fallback_or_error(
            assistant,
            "assistant response is not an object",
            assistant_command=assistant_command,
        )

    action = str(data.get("action") or "")
    if action not in {"clipboard", "agent", "confirm", "queued", "blocked"}:
        return _fallback_or_error(
            assistant,
            f"assistant returned unsupported action: {action}",
            assistant_command=assistant_command,
        )

    return AssistantRoute(
        action=action,
        message=str(data.get("message") or action),
        command_id=str(data.get("command_id") or ""),
        agent_id=str(data.get("agent_id") or ""),
        context_id=str(data.get("context_id") or ""),
        confirmation_id=str(data.get("confirmation_id") or ""),
        run_id=str(data.get("run_id") or data.get("agenthub_run_id") or ""),
        risk_level=str(data.get("risk_level") or ""),
    )
