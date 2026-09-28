import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

CURRENT_FILE_PATH = Path(__file__).resolve()
LAUNCH_DIR = CURRENT_FILE_PATH.parent
WORKERS_DIR = LAUNCH_DIR.parents[1]
PROJECT_ROOT = WORKERS_DIR.parents[1]
DEFAULT_LOG_FOLDER = WORKERS_DIR / "logs" / "server_log"
DEFAULT_CONFIG_PATH = LAUNCH_DIR / "config" / "remote_tools.yaml"

project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

import tool_server
import yaml
from box import Box


class RemoteToolsManager:
    """Start and manage tool workers without starting a controller."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = Box(config or {})
        self.logger = self._setup_logger()
        self.base_dir = WORKERS_DIR
        self.log_folder = DEFAULT_LOG_FOLDER
        self.log_folder.mkdir(parents=True, exist_ok=True)
        self.python_paths = {}
        self.processes = []
        self.tool_worker_config = self.config.get("tool_worker_config", [])

        os.environ["OMP_NUM_THREADS"] = "1"
        os.chdir(self.base_dir)

    def _setup_logger(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        return logging.getLogger(__name__)

    def _resolve_python_path(self, conda_env: Optional[str]) -> str:
        if not conda_env:
            return sys.executable

        if conda_env in self.python_paths:
            return self.python_paths[conda_env]

        current_prefix = Path(sys.prefix).resolve()
        if current_prefix.name == conda_env:
            python_path = Path(sys.executable).resolve()
        else:
            conda_executable = os.environ.get("CONDA_EXE") or shutil.which("conda")
            if not conda_executable:
                raise RuntimeError(
                    f"Cannot resolve Conda environment {conda_env!r}: "
                    "conda is not available."
                )

            result = subprocess.run(
                [conda_executable, "info", "--json"],
                check=True,
                capture_output=True,
                text=True,
            )
            conda_info = json.loads(result.stdout)
            if conda_env == "base":
                environment_prefixes = [conda_info.get("root_prefix")]
            else:
                environment_prefixes = [
                    prefix
                    for prefix in conda_info.get("envs", [])
                    if Path(prefix).name == conda_env
                ]
            environment_prefixes = [
                prefix for prefix in environment_prefixes if prefix
            ]
            if len(environment_prefixes) != 1:
                raise RuntimeError(
                    f"Expected exactly one Conda environment named {conda_env!r}, "
                    f"found {len(environment_prefixes)}."
                )
            python_path = Path(environment_prefixes[0]) / "bin" / "python"

        if not python_path.is_file():
            raise RuntimeError(
                f"Python executable for Conda environment {conda_env!r} "
                f"does not exist: {python_path}"
            )

        resolved_path = str(python_path)
        self.python_paths[conda_env] = resolved_path
        return resolved_path

    def run_local_command(
        self,
        job_name: str,
        command: List[str],
        log_file: str,
        conda_env: Optional[str] = None,
        cuda_visible_devices: Optional[str] = None,
    ) -> subprocess.Popen:
        command[0] = self._resolve_python_path(conda_env)
        project_root = os.path.dirname(os.path.dirname(tool_server.__file__))
        env = os.environ.copy()
        env.pop("LD_LIBRARY_PATH", None)
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{project_root}:{current_pythonpath}"
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices or ""

        log_handle = open(log_file, "w")
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=log_handle,
            env=env,
        )
        log_handle.close()
        self.logger.info(
            f"[Remote Tools] Process {job_name} started with PID {process.pid}"
        )
        return process

    def wait_for_process(self, process: subprocess.Popen, job_name: str) -> None:
        self.logger.info(f"Waiting for process to start: {job_name}")
        time.sleep(2)
        if process.poll() is not None:
            raise RuntimeError(
                f"Process {job_name} failed to start with exit code "
                f"{process.returncode}"
            )
        self.logger.info(f"Process {job_name} is running with PID: {process.pid}")

    def start_tool_workers(self) -> None:
        for tool_config in self.tool_worker_config:
            worker_configs = list(tool_config.values())[0]
            script_addr = next(iter(worker_configs.values())).cmd.script_addr
            subprocess.run(
                ["pkill", "-f", os.path.basename(script_addr)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            for worker_name, worker_config in worker_configs.items():
                self.start_worker(worker_name, worker_config)

    def start_worker(self, worker_name: str, worker_config: Box) -> None:
        log_file = self.log_folder / f"{worker_name}_worker.log"
        script_addr = worker_config.cmd.pop("script-addr")
        worker_config.cmd.worker_name = worker_name

        command = ["python", script_addr]
        for key, value in worker_config.cmd.items():
            command.extend([f"--{key}", str(value)])

        process = self.run_local_command(
            worker_name,
            command,
            str(log_file),
            conda_env=worker_config.get("conda_env"),
            cuda_visible_devices=worker_config.get("cuda_visible_devices"),
        )
        self.processes.append({"name": worker_name, "process": process})
        self.wait_for_process(process, worker_name)

    def shutdown_services(self) -> None:
        for process_info in self.processes:
            process = process_info["process"]
            name = process_info["name"]
            if process.poll() is not None:
                self.logger.info(
                    f"Process {name} already finished with exit code "
                    f"{process.returncode}"
                )
                continue

            process.terminate()
            try:
                process.wait(timeout=5)
                self.logger.info(
                    f"Process {name} (PID: {process.pid}) terminated successfully"
                )
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                self.logger.warning(
                    f"Process {name} (PID: {process.pid}) killed forcefully"
                )

        self.processes.clear()
        self.logger.info("All remote tool services have been shut down")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the remote tool configuration file",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    manager = RemoteToolsManager(config)
    try:
        manager.start_tool_workers()
        manager.logger.info(
            "All remote tool services started. Press Ctrl+C to shut down."
        )
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.logger.info("Shutting down remote tool services...")
        manager.shutdown_services()
    except Exception:
        manager.logger.exception("Failed to run remote tool services")
        manager.shutdown_services()
        raise


if __name__ == "__main__":
    main()
