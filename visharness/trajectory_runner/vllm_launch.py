"""Launch vLLM unchanged and retain process-bound KV startup information.

This module uses only the standard library so it can run in the vLLM serving
environment, independently of trajectory-runner training dependencies.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_ENGINE = re.compile(r"EngineCore(?:_DP(\d+))?\s+pid=(\d+)")
_TOKENS = re.compile(r"GPU KV cache size:\s*([\d,]+)\s+tokens")
_DTYPE = re.compile(r"(?<!\w)dtype=(?:torch\.)?([\w]+)")


def _registry_path(port: int) -> Path:
    # Shared by both checkouts, but never by different OS users.
    # Do not depend on TMPDIR: deployment and inference shells may differ.
    return Path("/tmp") / f"visharness-vllm-{os.getuid()}" / f"{port}.json"


@lru_cache(maxsize=1)
def _boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def _process_identity(pid: int) -> dict[str, Any] | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start_ticks": fields[19], "boot_id": _boot_id()}
    except (OSError, ValueError, IndexError):
        return None


def _is_descendant(pid: int, parent: int) -> bool:
    visited: set[int] = set()
    while pid > 1 and pid not in visited:
        if pid == parent:
            return True
        visited.add(pid)
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            pid = int(fields[1])
        except (OSError, ValueError, IndexError):
            return False
    return False


def _server_options(arguments: list[str]) -> dict[str, Any]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command", choices=["serve"])
    parser.add_argument("model")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", nargs="+")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", "-pp", type=int, default=1)
    parser.add_argument("--data-parallel-size", "-dp", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    options, _ = parser.parse_known_args(arguments)
    return {
        "host": options.host,
        "port": options.port,
        "model_config_path": str(Path(options.model).expanduser().resolve()),
        "model_names": options.served_model_name or [options.model],
        "tensor_parallel_size": options.tensor_parallel_size,
        "pipeline_parallel_size": options.pipeline_parallel_size,
        "data_parallel_size": options.data_parallel_size,
        "model_dtype": options.dtype if options.dtype != "auto" else None,
    }


class StartupRecorder:
    """Record only real startup lines belonging to the live serving process."""

    def __init__(self, arguments: list[str], pid: int, startup_log: str):
        identity = _process_identity(pid)
        if identity is None:
            raise ValueError("Cannot identify the live vLLM serving process")
        self.record = {
            "schema_version": 1,
            **_server_options(arguments),
            "server_process": identity,
            "startup_log": startup_log,
            "engines": {},
        }
        self.path = _registry_path(self.record["port"])
        self._publish()

    @property
    def complete(self) -> bool:
        return (
            bool(self.record["model_dtype"])
            and len(self.record["engines"]) == self.record["data_parallel_size"]
        )

    def _publish(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as file:
                temporary = file.name
                json.dump(self.record, file)
            os.replace(temporary, self.path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    def observe(self, raw_line: str) -> None:
        line = _ANSI.sub("", raw_line)
        changed = False
        dtype = _DTYPE.search(line)
        if dtype and dtype[1] != self.record["model_dtype"]:
            self.record["model_dtype"] = dtype[1]
            changed = True
        tokens = _TOKENS.search(line)
        engine = _ENGINE.search(line)
        if tokens and engine:
            engine_id = engine[1] or "0"
            pid = int(engine[2])
            identity = _process_identity(pid)
            # Also rejects old tmux scrollback when registering an existing server.
            if identity is not None and _is_descendant(
                pid, self.record["server_process"]["pid"]
            ):
                capacity = int(tokens[1].replace(",", ""))
                if capacity > 0:
                    value = {"capacity_tokens": capacity, "process": identity}
                    if self.record["engines"].get(engine_id) != value:
                        self.record["engines"][engine_id] = value
                        changed = True
        if changed:
            self._publish()


def load_startup_record(
    metrics_url: str | None,
    *,
    model_name: str | None,
    model_config_path: Any,
) -> dict[str, Any] | None:
    """Read local startup metadata only while the exact server is still alive."""

    if not metrics_url:
        return None
    try:
        url = urlsplit(metrics_url)
        host = url.hostname or ""
        try:
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = host in {"localhost", socket.gethostname()}
        if not local:
            return None
        port = url.port or (443 if url.scheme == "https" else 80)
        record = json.loads(_registry_path(port).read_text())
        if record.get("schema_version") != 1 or record["port"] != port:
            return None
        bind_host = record["host"]
        if bind_host not in {"0.0.0.0", "::", "", host}:
            # A server bound only to another interface can share this port.
            # Do not attribute its cache to the requested local endpoint.
            try:
                same_loopback = (
                    ipaddress.ip_address(host).is_loopback
                    and ipaddress.ip_address(bind_host).is_loopback
                )
            except ValueError:
                same_loopback = (
                    host in {"localhost", "127.0.0.1", "::1"}
                    and bind_host in {"localhost", "127.0.0.1", "::1"}
                )
            if not same_loopback:
                return None
        identity = record["server_process"]
        if _process_identity(int(identity["pid"])) != identity:
            return None
        if model_name and model_name not in record["model_names"]:
            return None
        if model_config_path:
            path = Path(str(model_config_path)).expanduser().resolve()
            if path.name == "config.json":
                path = path.parent
            if str(path) != record["model_config_path"]:
                return None
        if (
            not record.get("model_dtype")
            or len(record["engines"]) != record["data_parallel_size"]
        ):
            return None
        for engine in record["engines"].values():
            process = engine["process"]
            if _process_identity(int(process["pid"])) != process:
                return None
        return record
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


def main(arguments: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    executable = shutil.which("vllm")
    if executable is None:
        raise RuntimeError("vllm is not installed in the active environment")
    root = Path(__file__).resolve().parents[2]
    run_dir = root / "outputs" / "vllm" / str(time.time_ns())
    log_path = run_dir / "startup.log"
    startup_log = None
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        startup_log = log_path.open("w", encoding="utf-8")
    except OSError as exc:
        print(f"Startup log unavailable: {exc}", file=sys.stderr)
    process = subprocess.Popen(
        [executable, *arguments], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        start_new_session=True,
    )

    def forward_signal(signum, _frame):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

    old_handlers = {
        signum: signal.signal(signum, forward_signal)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    recorder = None
    try:
        try:
            recorder = StartupRecorder(arguments, process.pid, str(log_path))
        except Exception as exc:
            print(f"KV startup recording unavailable: {exc}", file=sys.stderr)
        forwarding = True
        startup_finished = False
        assert process.stdout is not None
        for line in process.stdout:
            if forwarding:
                try:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                except BrokenPipeError:
                    forwarding = False
            if not startup_finished:
                if startup_log is not None:
                    try:
                        startup_log.write(line)
                        startup_log.flush()
                    except OSError as exc:
                        print(f"Startup log unavailable: {exc}", file=sys.stderr)
                        startup_log.close()
                        startup_log = None
                if recorder is not None:
                    try:
                        recorder.observe(line)
                    except Exception as exc:
                        print(f"KV startup recording unavailable: {exc}", file=sys.stderr)
                        recorder = None
                startup_finished = bool(
                    (recorder is not None and recorder.complete)
                    or "Application startup complete" in line
                )
        return_code = process.wait()
        return return_code if return_code >= 0 else 128 - return_code
    finally:
        if process.poll() is None:
            forward_signal(signal.SIGTERM, None)
            process.wait()
        if process.stdout is not None:
            process.stdout.close()
        if startup_log is not None:
            startup_log.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
