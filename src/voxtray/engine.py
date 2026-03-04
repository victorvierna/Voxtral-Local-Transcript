from __future__ import annotations

import logging
import os
from pathlib import Path
import signal
import subprocess
import time

import httpx

from .config import VoxtrayConfig
from .paths import VLLM_LOG_FILE, ensure_app_dirs
from .state import StateStore, pid_is_alive


class EngineError(RuntimeError):
    pass


class EngineManager:
    def __init__(self, config: VoxtrayConfig, state_store: StateStore) -> None:
        self.config = config
        self.state_store = state_store
        self.logger = logging.getLogger("voxtray.engine")

    def _is_external(self) -> bool:
        return bool(self.config.server.external_base_url)

    def is_ready(self, timeout_seconds: float = 3.0) -> bool:
        url = f"{self.config.server_base_url}/v1/models"
        try:
            response = httpx.get(url, timeout=timeout_seconds)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    def _build_command(self) -> list[str]:
        command = [
            self.config.engine.command,
            "serve",
            self.config.model_id,
            "--host",
            self.config.server.host,
            "--port",
            str(self.config.server.port),
            "--compilation_config",
            self.config.engine.compilation_config,
        ]
        if self.config.engine.enforce_eager:
            command.append("--enforce-eager")
        command.extend(self.config.engine.extra_args)
        return command

    def ensure_running(self) -> None:
        if self.is_ready(timeout_seconds=2.0):
            return

        if self._is_external():
            raise EngineError(
                "configured external_base_url is not reachable at /v1/models"
            )

        state = self.state_store.read()
        pid = state.get("engine_pid")
        if pid and pid_is_alive(pid):
            if not self._is_expected_engine_pid(pid):
                self.logger.warning(
                    "state engine_pid=%s is alive but does not look like vLLM; resetting",
                    pid,
                )
                self.state_store.set_values(engine_pid=None)
            else:
                # Process exists but is not ready. Wait first, then force a clean restart.
                try:
                    self._wait_until_ready_or_fail(pid)
                    return
                except EngineError as exc:
                    self.logger.warning(
                        "existing vLLM process %s failed readiness check (%s); restarting",
                        pid,
                        exc,
                    )
                    self._stop_process_group(pid, timeout_seconds=5.0)
                    self.state_store.set_values(engine_pid=None)

        self.start_local_engine()

    def start_local_engine(self) -> None:
        if self._is_external():
            raise EngineError("cannot start local engine when external_base_url is set")
        if self.is_ready(timeout_seconds=2.0):
            return

        ensure_app_dirs()
        log_path = Path(VLLM_LOG_FILE)
        command = self._build_command()
        env = os.environ.copy()
        if self.config.engine.disable_compile_cache:
            env["VLLM_DISABLE_COMPILE_CACHE"] = "1"

        self.logger.info("starting vLLM: %s", " ".join(command))

        try:
            with log_path.open("ab") as log_file:
                proc = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=env,
                )
        except OSError as exc:
            raise EngineError(f"failed to start vLLM command '{command[0]}': {exc}") from exc

        self.state_store.set_values(engine_pid=proc.pid)
        self._wait_until_ready_or_fail(proc.pid)

    def _wait_until_ready_or_fail(self, pid: int) -> None:
        timeout = self.config.server.start_timeout_seconds
        start = time.time()
        while time.time() - start < timeout:
            if not pid_is_alive(pid):
                self.state_store.set_values(engine_pid=None)
                raise EngineError(
                    f"vLLM process {pid} exited during startup. See log: {VLLM_LOG_FILE}"
                )
            if self.is_ready(timeout_seconds=2.0):
                return
            time.sleep(1.0)
        self.state_store.set_values(engine_pid=None)
        raise EngineError(
            f"vLLM not ready after {timeout}s. See log: {VLLM_LOG_FILE}"
        )

    @staticmethod
    def _read_process_cmdline(pid: int) -> list[str] | None:
        cmdline_path = Path(f"/proc/{pid}/cmdline")
        try:
            raw = cmdline_path.read_bytes()
        except OSError:
            return None
        if not raw:
            return []
        return [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]

    def _is_expected_engine_pid(self, pid: int) -> bool:
        cmdline = self._read_process_cmdline(pid)
        if cmdline is None:
            # If we cannot inspect /proc, preserve legacy behavior.
            return True
        if not cmdline:
            return False
        parts = [part.strip() for part in cmdline if part.strip()]
        if not parts:
            return False

        expected = Path(self.config.engine.command).name
        names = {Path(part).name for part in parts}
        if expected == "vllm":
            return "vllm" in names and "serve" in parts
        return expected in names

    @staticmethod
    def _process_group_is_alive(pgid: int) -> bool:
        if pgid <= 0:
            return False
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @staticmethod
    def _signal_process_group(pgid: int, sig: int) -> bool:
        try:
            os.killpg(pgid, sig)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # Fallback for restricted environments where group signaling may fail.
            try:
                os.kill(pgid, sig)
                return True
            except ProcessLookupError:
                return False
        return False

    def _stop_process_group(self, pgid: int, timeout_seconds: float) -> None:
        if not pid_is_alive(pgid):
            return
        self.logger.info("stopping vLLM process group pgid=%s", pgid)
        if not self._signal_process_group(pgid, signal.SIGTERM):
            return
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if not self._process_group_is_alive(pgid):
                return
            time.sleep(0.2)
        self._signal_process_group(pgid, signal.SIGKILL)

    def stop_if_running(self, timeout_seconds: float = 15.0) -> None:
        state = self.state_store.read()
        pid = state.get("engine_pid")
        if not pid:
            return
        if not pid_is_alive(pid):
            self.state_store.set_values(engine_pid=None)
            return

        self._stop_process_group(pid, timeout_seconds=timeout_seconds)
        self.state_store.set_values(engine_pid=None)
