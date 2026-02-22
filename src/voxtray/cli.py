from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Literal

import typer

from .config import write_default_config
from .controller import Controller, handle_engine_error
from .engine import EngineError
from .gnome import GnomeShortcutError, install_toggle_shortcut
from .logging_utils import configure_logging
from .paths import APP_LOG_FILE, VLLM_LOG_FILE
from .tray import run_tray
from .worker import run_record_worker

app = typer.Typer(
    help="Voxtray: local Voxtral realtime transcription utility.",
    pretty_exceptions_enable=False,
)
warm_app = typer.Typer(help="Warm engine controls.", pretty_exceptions_enable=False)
model_app = typer.Typer(help="Model load/unload controls.", pretty_exceptions_enable=False)
history_app = typer.Typer(help="History controls.", pretty_exceptions_enable=False)

app.add_typer(warm_app, name="warm")
app.add_typer(model_app, name="model")
app.add_typer(history_app, name="history")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging."),
) -> None:
    configure_logging(verbose=verbose)
    write_default_config()


@app.command("init")
def init_config() -> None:
    path = write_default_config()
    typer.echo(f"Config ready at: {path}")


@app.command("status")
def status() -> None:
    ctl = Controller()
    info = ctl.status()
    for key, value in info.items():
        typer.echo(f"{key}: {value}")


@app.command("logs")
def logs(
    target: Literal["app", "vllm", "all"] = typer.Option(
        "all", "--target", help="Which log to show."
    ),
    lines: int = typer.Option(120, "--lines", min=1, max=2000),
) -> None:
    def _print_tail(path: Path, label: str) -> None:
        typer.echo(f"== {label}: {path} ==")
        if not path.exists():
            typer.echo("(missing)")
            return
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in deque(handle, maxlen=lines):
                typer.echo(line.rstrip("\n"))

    if target in {"app", "all"}:
        _print_tail(APP_LOG_FILE, "app")
    if target in {"vllm", "all"}:
        _print_tail(VLLM_LOG_FILE, "vllm")


@app.command("record")
def record(
    start: bool = typer.Option(False, "--start", help="Start recording."),
    stop: bool = typer.Option(False, "--stop", help="Stop recording."),
    toggle: bool = typer.Option(False, "--toggle", help="Toggle recording state."),
) -> None:
    selected = sum([bool(start), bool(stop), bool(toggle)])
    if selected > 1:
        raise typer.BadParameter("use only one of --start, --stop, --toggle")
    if selected == 0:
        toggle = True

    ctl = Controller()
    try:
        if start:
            message = ctl.start_recording()
        elif stop:
            message = ctl.stop_recording()
        else:
            message = ctl.toggle_recording()
        typer.echo(message)
    except (EngineError, RuntimeError) as exc:
        typer.echo(handle_engine_error(exc), err=True)
        raise typer.Exit(1)


@warm_app.command("on")
def warm_on() -> None:
    ctl = Controller()
    try:
        message = ctl.warm_on()
    except (EngineError, RuntimeError) as exc:
        typer.echo(handle_engine_error(exc), err=True)
        raise typer.Exit(1)
    typer.echo(message)


@warm_app.command("off")
def warm_off() -> None:
    ctl = Controller()
    message = ctl.warm_off()
    typer.echo(message)


@warm_app.command("status")
def warm_status() -> None:
    ctl = Controller()
    info = ctl.warm_status()
    for key, value in info.items():
        typer.echo(f"{key}: {value}")


@model_app.command("load")
def model_load() -> None:
    ctl = Controller()
    try:
        message = ctl.load_model()
    except (EngineError, RuntimeError) as exc:
        typer.echo(handle_engine_error(exc), err=True)
        raise typer.Exit(1)
    typer.echo(message)


@model_app.command("unload")
def model_unload() -> None:
    ctl = Controller()
    try:
        message = ctl.unload_model()
    except (EngineError, RuntimeError) as exc:
        typer.echo(handle_engine_error(exc), err=True)
        raise typer.Exit(1)
    typer.echo(message)


@model_app.command("status")
def model_status() -> None:
    ctl = Controller()
    info = ctl.model_status()
    for key, value in info.items():
        typer.echo(f"{key}: {value}")


@history_app.command("list")
def history_list() -> None:
    ctl = Controller()
    entries = ctl.list_history()
    if not entries:
        typer.echo("History is empty.")
        return
    for idx, entry in enumerate(entries, start=1):
        created_at = entry.get("created_at", "")
        text = str(entry.get("text", "")).replace("\n", " ").strip()
        typer.echo(f"{idx}. [{created_at}] {text}")


@history_app.command("copy")
def history_copy(index: int = typer.Argument(..., min=1)) -> None:
    ctl = Controller()
    try:
        entry, backend = ctl.copy_history_item(index)
    except IndexError:
        typer.echo(f"History item {index} not found.", err=True)
        raise typer.Exit(1)
    except Exception as exc:
        typer.echo(f"Could not copy history item {index}: {exc}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Copied history item {index} with backend={backend}.")
    typer.echo(str(entry.get("text", "")))


@app.command("transcribe-file")
def transcribe_file(
    audio_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    copy: bool = typer.Option(True, "--copy/--no-copy", help="Copy result to clipboard."),
) -> None:
    ctl = Controller()
    try:
        result = ctl.transcribe_file(audio_path=audio_path, copy_result=copy)
    except (EngineError, RuntimeError) as exc:
        typer.echo(handle_engine_error(exc), err=True)
        raise typer.Exit(1)
    except Exception as exc:
        typer.echo(f"transcription failed: {exc}", err=True)
        raise typer.Exit(1)

    text = str(result.get("text", ""))
    typer.echo(text)


@app.command("tray")
def tray() -> None:
    try:
        exit_code = run_tray(Controller())
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)
    raise typer.Exit(exit_code)


@app.command("install-shortcut")
def install_shortcut(
    binding: str = typer.Option("<Super>F9", help="GNOME keybinding string."),
    command: str = typer.Option(
        "voxtray record --toggle", help="Command run by the shortcut."
    ),
    name: str = typer.Option("Voxtray Toggle Record", help="Shortcut display name."),
) -> None:
    try:
        result = install_toggle_shortcut(binding=binding, command=command, name=name)
    except GnomeShortcutError as exc:
        typer.echo(f"Shortcut install failed: {exc}", err=True)
        raise typer.Exit(1)

    typer.echo("Shortcut installed:")
    for key, value in result.items():
        typer.echo(f"  {key}: {value}")


@app.command("_record-worker", hidden=True)
def record_worker() -> None:
    code = run_record_worker()
    raise typer.Exit(code=code)


def entrypoint() -> None:
    app()


if __name__ == "__main__":
    entrypoint()
