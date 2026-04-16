from __future__ import annotations

import ctypes
from datetime import datetime
import logging
import os
import subprocess
import threading

from .clipboard import ClipboardError, copy_to_clipboard
from .controller import Controller
from .history import preview_text


def _check_linux_qt_runtime() -> None:
    # Qt's xcb backend on Ubuntu needs libxcb-cursor0 at runtime.
    if os.name != "posix":
        return
    if os.environ.get("XDG_SESSION_TYPE", "").lower() != "x11":
        return
    try:
        ctypes.CDLL("libxcb-cursor.so.0")
    except OSError as exc:
        raise RuntimeError(
            "Missing system library 'libxcb-cursor0' required by Qt tray on X11. "
            "Install it with: sudo apt-get install -y libxcb-cursor0"
        ) from exc


def run_tray(controller: Controller | None = None) -> int:
    _check_linux_qt_runtime()
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtGui import QAction, QIcon
        from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon
    except ImportError as exc:
        raise RuntimeError(
            "PySide6 is required for tray mode. Install with: pip install 'voxtray[ui]'"
        ) from exc

    logger = logging.getLogger("voxtray.tray")
    ctl = controller or Controller()
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)

    tray = QSystemTrayIcon()
    idle_icon = QIcon.fromTheme("audio-input-microphone")
    recording_icon = QIcon.fromTheme("media-record", idle_icon)
    if not idle_icon.isNull():
        tray.setIcon(idle_icon)

    menu = QMenu()
    action_toggle = QAction("Start Recording")
    menu.addAction(action_toggle)

    action_warm = QAction("Keep engine loaded")
    action_warm.setCheckable(True)
    menu.addAction(action_warm)

    action_model = QAction("Model loaded")
    action_model.setCheckable(True)
    menu.addAction(action_model)

    menu.addSeparator()
    history_menu = menu.addMenu("Recent (5)")

    menu.addSeparator()
    action_status = QAction("Show Status")
    menu.addAction(action_status)

    action_open_config = QAction("Open Config Path")
    menu.addAction(action_open_config)

    menu.addSeparator()
    action_quit = QAction("Quit")
    menu.addAction(action_quit)

    tray.setContextMenu(menu)
    tray.setToolTip("Voxtray")
    menu_is_open = False
    last_is_recording: bool | None = None
    last_notice_id = ""
    refresh_interval_ms = 1500
    min_refresh_interval_ms = 1000
    max_refresh_interval_ms = 7000

    def _notify(
        title: str,
        text: str,
        icon=QSystemTrayIcon.MessageIcon.Information,
        timeout_ms: int = 3500,
        interactive: bool = False,
    ) -> None:
        if QSystemTrayIcon.supportsMessages():
            tray.showMessage(title, text, icon, timeout_ms)
            return
        logger.info("%s: %s", title, text)
        tray.setToolTip(f"Voxtray - {text[:96]}")
        if interactive:
            QMessageBox.information(None, title, text)

    def _run_async(fn):
        threading.Thread(target=fn, daemon=True).start()

    def _set_menu_open(value: bool) -> None:
        nonlocal menu_is_open
        menu_is_open = value

    def _toggle_record() -> None:
        def work() -> None:
            message = ctl.toggle_recording()
            logger.info(message)

        _run_async(work)

    def _toggle_warm(checked: bool) -> None:
        def work() -> None:
            try:
                ctl.apply_warm_preference(bool(checked))
            except Exception:
                logger.exception("failed to toggle warm mode")

        _run_async(work)

    def _toggle_model(checked: bool) -> None:
        def work() -> None:
            try:
                message = ctl.load_model() if checked else ctl.unload_model()
                logger.info(message)
            except Exception:
                logger.exception("failed to toggle model loaded state")

        _run_async(work)

    def _show_status() -> None:
        try:
            status = ctl.status()
            _notify(
                "Voxtray Status",
                (
                    f"recording={status['recording']} warm={status['warm_enabled']} "
                    f"engine_ready={status['engine_ready']} "
                    f"model_loaded={status['model_loaded']}"
                ),
                QSystemTrayIcon.MessageIcon.Information,
                3500,
                interactive=True,
            )
        except Exception:
            logger.exception("failed to show status")
            _notify(
                "Voxtray",
                "Could not read status. Check logs with: voxtray logs --target app --lines 200",
                QSystemTrayIcon.MessageIcon.Warning,
                4000,
                interactive=True,
            )

    def _quit() -> None:
        try:
            summary = ctl.shutdown_for_exit()
            logger.info("tray shutdown: %s", summary)
        except Exception:
            logger.exception("failed shutdown cleanup")
        finally:
            app.quit()

    def _preload_model_startup() -> None:
        try:
            logger.info(ctl.preload_if_warm_enabled())
        except Exception:
            logger.exception("failed startup model preload")

    def _open_config_dir() -> None:
        from .paths import CONFIG_DIR

        try:
            subprocess.Popen(  # noqa: S603
                ["xdg-open", str(CONFIG_DIR)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            _notify(
                "Voxtray",
                f"Opened config path: {CONFIG_DIR}",
                QSystemTrayIcon.MessageIcon.Information,
                3000,
                interactive=True,
            )
        except Exception:
            logger.exception("failed to open config path")
            _notify(
                "Voxtray",
                f"Config path: {CONFIG_DIR}",
                QSystemTrayIcon.MessageIcon.Warning,
                4000,
                interactive=True,
            )

    def _history_entry_label(index: int, entry: dict[str, object]) -> str:
        preview = preview_text(str(entry.get("text", "")))
        created_at = str(entry.get("created_at", "") or "")
        if created_at:
            try:
                timestamp = datetime.fromisoformat(created_at).astimezone().strftime("%H:%M")
                return f"{index}. {preview} ({timestamp})"
            except ValueError:
                pass
        return f"{index}. {preview}"

    def _copy_history_entry(entry: dict[str, object], index: int) -> None:
        text = str(entry.get("text", ""))
        preview = preview_text(text)
        try:
            copy_to_clipboard(text, backend=ctl.config.clipboard.backend)
            _notify(
                "Voxtray",
                f"Copied #{index}: {preview}",
                QSystemTrayIcon.MessageIcon.Information,
                2800,
                interactive=False,
            )
        except ClipboardError:
            _notify(
                "Voxtray",
                f"Could not copy #{index}: {preview}",
                QSystemTrayIcon.MessageIcon.Warning,
                2800,
                interactive=False,
            )
        except Exception:
            logger.exception("unexpected error copying history item %s", index)
            _notify(
                "Voxtray",
                f"Unexpected error while copying #{index}: {preview}",
                QSystemTrayIcon.MessageIcon.Warning,
                3200,
                interactive=False,
            )

    def _rebuild_history_menu() -> None:
        entries = ctl.list_history()
        history_menu.clear()
        history_menu.setTitle(f"Recent ({len(entries)})")
        if not entries:
            empty = QAction("No transcripts yet")
            empty.setEnabled(False)
            history_menu.addAction(empty)
            return

        for idx, entry in enumerate(entries, start=1):
            action = QAction(_history_entry_label(idx, entry))
            full_text = " ".join(str(entry.get("text", "")).split())
            if full_text:
                tooltip = full_text if len(full_text) <= 512 else full_text[:509] + "..."
                action.setToolTip(tooltip)
                action.setStatusTip(tooltip)
            action.triggered.connect(
                lambda checked=False, e=entry, i=idx: _copy_history_entry(e, i)
            )
            history_menu.addAction(action)

    def _refresh() -> None:
        nonlocal menu_is_open
        nonlocal last_is_recording
        nonlocal last_notice_id
        nonlocal refresh_interval_ms
        try:
            status = ctl.status()
            is_recording = bool(status["recording"])
            action_toggle.setText("Stop Recording" if is_recording else "Start Recording")

            if last_is_recording is None or last_is_recording != is_recording:
                last_is_recording = is_recording
                tray.setIcon(recording_icon if is_recording else idle_icon)
                tray.setToolTip("Voxtray (Recording)" if is_recording else "Voxtray")

            action_warm.blockSignals(True)
            action_warm.setChecked(bool(status["warm_enabled"]))
            action_warm.blockSignals(False)

            action_model.blockSignals(True)
            action_model.setChecked(bool(status["model_loaded"]))
            action_model.blockSignals(False)

            # Poll quickly while recording; back off when idle to reduce load/log noise.
            target_interval = min_refresh_interval_ms if is_recording else max_refresh_interval_ms
            if not is_recording and not menu_is_open:
                target_interval = min(max_refresh_interval_ms, refresh_interval_ms + 500)
            elif is_recording:
                target_interval = min_refresh_interval_ms

            if target_interval != refresh_interval_ms:
                refresh_interval_ms = target_interval
                refresh_timer.setInterval(refresh_interval_ms)

            notice_id = str(status.get("last_notice_id", "") or "")
            if notice_id and notice_id != last_notice_id:
                last_notice_id = notice_id
                notice_level = str(status.get("last_notice_level", "info") or "info")
                notice_icon = {
                    "warning": QSystemTrayIcon.MessageIcon.Warning,
                    "error": QSystemTrayIcon.MessageIcon.Critical,
                }.get(notice_level, QSystemTrayIcon.MessageIcon.Information)
                notice_title = str(status.get("last_notice_title", "") or "Voxtray")
                notice_body = str(status.get("last_notice_body", "") or "").strip()
                if notice_body:
                    _notify(
                        notice_title,
                        notice_body,
                        notice_icon,
                        4500,
                        interactive=notice_level == "error",
                    )

        except Exception:
            logger.exception("tray refresh failed")

    action_toggle.triggered.connect(_toggle_record)
    action_warm.triggered.connect(_toggle_warm)
    action_model.triggered.connect(_toggle_model)
    action_status.triggered.connect(_show_status)
    action_open_config.triggered.connect(_open_config_dir)
    action_quit.triggered.connect(_quit)
    menu.aboutToShow.connect(lambda: _set_menu_open(True))
    menu.aboutToHide.connect(lambda: _set_menu_open(False))

    def _on_history_menu_show() -> None:
        _set_menu_open(True)
        try:
            _rebuild_history_menu()
        except Exception:
            logger.exception("failed rebuilding history menu")

    history_menu.aboutToShow.connect(_on_history_menu_show)
    history_menu.aboutToHide.connect(lambda: _set_menu_open(False))

    tray.activated.connect(
        lambda reason: _toggle_record()
        if reason == QSystemTrayIcon.ActivationReason.Trigger
        else None
    )

    refresh_timer = QTimer()
    refresh_timer.setInterval(refresh_interval_ms)
    refresh_timer.timeout.connect(_refresh)
    refresh_timer.start()

    tray.show()
    _refresh()
    _run_async(_preload_model_startup)
    _notify(
        "Voxtray",
        "Tray ready. Click icon or use GNOME shortcut to start/stop recording.",
        QSystemTrayIcon.MessageIcon.Information,
        3500,
        interactive=False,
    )
    return app.exec()
