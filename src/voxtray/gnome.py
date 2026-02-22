from __future__ import annotations

import ast
import subprocess


class GnomeShortcutError(RuntimeError):
    pass


BASE_SCHEMA = "org.gnome.settings-daemon.plugins.media-keys"
KEYBINDING_SCHEMA_PREFIX = (
    "org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:"
)


def _run_gsettings(args: list[str]) -> str:
    proc = subprocess.run(
        ["gsettings", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if proc.returncode != 0:
        raise GnomeShortcutError(proc.stderr.strip() or "gsettings failed")
    return proc.stdout.strip()


def _parse_strv(raw: str) -> list[str]:
    value = raw.strip()
    if value.startswith("@as "):
        value = value[4:]
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise GnomeShortcutError(f"unexpected gsettings strv: {raw}")
    return [str(x) for x in parsed]


def install_toggle_shortcut(
    binding: str = "<Super>F9",
    command: str = "voxtray record --toggle",
    name: str = "Voxtray Toggle Record",
) -> dict[str, str]:
    path = "/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/voxtray/"

    raw_current = _run_gsettings(["get", BASE_SCHEMA, "custom-keybindings"])
    current = _parse_strv(raw_current)
    if path not in current:
        current.append(path)
        quoted = "[" + ", ".join(repr(x) for x in current) + "]"
        _run_gsettings(["set", BASE_SCHEMA, "custom-keybindings", quoted])

    schema = KEYBINDING_SCHEMA_PREFIX + path
    _run_gsettings(["set", schema, "name", name])
    _run_gsettings(["set", schema, "command", command])
    _run_gsettings(["set", schema, "binding", binding])

    return {
        "binding": binding,
        "command": command,
        "name": name,
        "path": path,
    }
