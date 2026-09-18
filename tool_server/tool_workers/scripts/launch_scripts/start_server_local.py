import os
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent.parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
import time
import subprocess
import logging
from pathlib import Path
import requests
import signal
from typing import Optional, List, Dict
from box import Box
import argparse
import yaml
import tool_server
from tool_server.utils.utils import load_json_file, write_json_file

class ServerManager:
    """Server Manager Class for local process management"""
    def __init__(self, config: Optional[Dict] = None):
        # Initialize configuration
        self.config = Box(config)
        self.logger = self._setup_logger()
        self.log_folder = Path(self.config.log_folder)
        self.log_folder.mkdir(parents=True, exist_ok=True)

        # Initialize status
        self.controller_addr = None
        self._clean_environment()
        os.chdir(self.config.base_dir)

        self.controller_config = self.config.controller_config
        self.model_worker_config = self.config.model_worker_config if "model_worker_config" in self.config else []
        self.tool_worker_config = self.config.tool_worker_config if "tool_worker_config" in self.config else []
        self.processes = []  # Track all started processes

    def _setup_logger(self) -> logging.Logger:
        """Set up logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _clean_environment(self) -> None:
        """Clean environment variables"""
        os.environ["OMP_NUM_THREADS"] = "1"

    # def run_local_command(self, job_name: str, command: List[str], log_file: str,
    #                       conda_env: str = None, cuda_visible_devices: str = None) -> subprocess.Popen:
    #     """Run command locally"""
    #     env = os.environ.copy()

    #     if cuda_visible_devices:
    #         env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    #     cmd = []
    #     if conda_env:
    #         # Use conda run instead of source activate
    #         cmd = ["conda", "run", "-n", conda_env]

    #     cmd.extend(command)

    #     self.logger.info(f"Starting process: {job_name} with command: {' '.join(cmd)}")
    #     with open(log_file, 'w') as f:
    #         process = subprocess.Popen(cmd, stdout=f, stderr=f, env=env)
    #         return process

    def run_local_command(self, job_name: str, command: List[str], log_file: str,
                        conda_env: str = None, cuda_visible_devices: str = None) -> subprocess.Popen:
        if conda_env is not None:
            python_path = os.path.join(os.path.dirname(self.config.conda_env), "{}/bin/python".format(conda_env))
        else:
            python_path = os.path.join(self.config.conda_env, "bin/python")
        command[0] = python_path
        project_root = os.path.dirname(os.path.dirname(tool_server.__file__))
        env = os.environ.copy()
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{project_root}:{current_pythonpath}"
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices if cuda_visible_devices else ""
        log_f = open(log_file, "w")
        proc = subprocess.Popen(
        command,
        stdout=log_f,
        stderr=log_f,
        env=env
        )
        self.logger.info(f"[Local Mode] Process {job_name} started with PID {proc.pid}")
        return proc

    def wait_for_process(self, process, job_name: str) -> dict:
        """Wait for process to initialize"""
        self.logger.info(f"Waiting for process to start: {job_name}")
        # Give the process a little time to initialize
        time.sleep(2)

        # Check if process is still running
        if process.poll() is not None:
            self.logger.error(f"Process {job_name} failed to start. Exit code: {process.returncode}")
            raise Exception(f"Process {job_name} failed to start")

        self.logger.info(f"Process {job_name} is running with PID: {process.pid}")
        return {"process": process, "pid": process.pid}

    def wait_for_worker_addr(self, worker_name: str) -> str:
        """Wait for worker address to be available"""
        self.logger.info(f"Waiting for {worker_name} worker...")
        attempt = 0

        while True:
            try:
                attempt += 1
                response = requests.post(
                    f"{self.controller_addr}/get_worker_address",
                    json={"model": worker_name},
                    timeout=self.config.request_timeout
                )
                response.raise_for_status()

                address = response.json().get("address", "")
                if address.strip():
                    self.logger.info(f"Worker {worker_name} is ready at: {address}")
                    return address

                self.logger.warning(f"Attempt {attempt}: worker not ready")

            except Exception as e:
                self.logger.error(f"Attempt {attempt} failed: {e}")

            time.sleep(self.config.retry_interval)

    def wait_for_controller_ready(self):
        self.logger.info("Waiting for controller to become ready...")
        for i in range(10):
            try:
                response = requests.post(f"{self.controller_addr}/list_models", timeout=2)
                if response.status_code == 200:
                    self.logger.info("Controller is ready.")
                    return
                else:
                    self.logger.info(f"Controller not ready yet... retrying ({i+1})")
                    time.sleep(2)
            except:
                self.logger.info(f"Controller not ready yet... retrying ({i+1})")
                time.sleep(2)
        raise RuntimeError("Controller failed to become ready.")


    def start_controller(self) -> str:
        """Start controller"""
        log_file = self.log_folder / f"{self.controller_config.worker_name}.log"
        script_addr = self.controller_config.cmd.pop("script-addr")
        job_name = self.controller_config.job_name
        command = ["python", script_addr]
        for k, v in self.controller_config.cmd.items():
            command.extend([f"--{k}", str(v)])

        subprocess.run(["pkill", "-f", os.path.basename(script_addr)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        process = self.run_local_command(
            job_name,
            command,
            str(log_file),
            conda_env=self.controller_config.get("conda_env", None),
            cuda_visible_devices=self.controller_config.get("cuda_visible_devices", None)
        )

        self.processes.append({"name": job_name, "process": process})
        self.wait_for_process(process, job_name)

        # Controller is running on localhost
        port = self.controller_config.cmd.port
        self.controller_addr = f"http://localhost:{port}"
        self.logger.info(f"Controller is running at: {self.controller_addr}")

        controller_addr_dict = {"controller_addr": self.controller_addr}
        if "controller_addr_location" in self.controller_config:
            self.controller_addr_location = self.controller_config.controller_addr_location
        else:
            current_file_path = os.path.dirname(os.path.abspath(__file__))
            self.controller_addr_location = f"{current_file_path}/../../online_workers/controller_addr/controller_addr.json"

        # Create directory if it doesn't exist
        controller_addr_dir = os.path.dirname(self.controller_addr_location)
        os.makedirs(controller_addr_dir, exist_ok=True)

        write_json_file(controller_addr_dict, self.controller_addr_location)
        self.logger.info(f"Controller address saved to: {self.controller_addr_location}")

        self.wait_for_controller_ready()

        return self.controller_addr

    def start_all_workers(self) -> None:
        """Start all worker services"""
        self.start_model_worker()
        self.start_tool_worker()

    def start_model_worker(self) -> None:
        for config in self.model_worker_config:
            config = list(config.values())[0]
            self.start_worker_by_config(config)

    def start_tool_worker(self) -> None:
        for tool_config in self.tool_worker_config:
            worker_configs = list(tool_config.values())[0]
            script_addr = next(iter(worker_configs.values())).cmd.script_addr
            subprocess.run(["pkill", "-f", os.path.basename(script_addr)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for worker_name, worker_config in worker_configs.items():
                self.start_worker_by_config(worker_name, worker_config)

    def start_worker_by_config(self, worker_name, worker_config) -> None:
        """Start specific worker"""

        log_file = self.log_folder / f"{worker_name}_worker.log"
        script_addr = worker_config.cmd.pop("script-addr")
        command = [
            "python", script_addr
        ]
        worker_config.cmd.worker_name = worker_name
        for k, v in worker_config.cmd.items():
            command.extend([f"--{k}", str(v)])

        process = self.run_local_command(
            worker_name,
            command,
            str(log_file),
            conda_env=worker_config.get("conda_env", None),
            cuda_visible_devices=worker_config.get("cuda_visible_devices", None)
        )

        self.processes.append({"name": worker_name, "process": process})
        self.wait_for_process(process, worker_name)

        if "wait_for_self" in worker_config and worker_config["wait_for_self"]:
            self.wait_for_worker_addr(worker_name)

    def shutdown_services(self) -> None:
        """Shut down all local processes"""
        try:
            if hasattr(self, 'controller_addr_location') and os.path.exists(self.controller_addr_location):
                os.remove(self.controller_addr_location)
                self.logger.info("Controller address file removed")

            for proc_info in self.processes:
                process = proc_info["process"]
                name = proc_info["name"]

                if process.poll() is None:  # Process is still running
                    # Send SIGTERM to gracefully terminate the process
                    process.terminate()
                    try:
                        # Wait for process to terminate
                        process.wait(timeout=5)
                        self.logger.info(f"Process {name} (PID: {process.pid}) terminated successfully")
                    except subprocess.TimeoutExpired:
                        # If process doesn't terminate after timeout, force kill
                        process.kill()
                        self.logger.warning(f"Process {name} (PID: {process.pid}) killed forcefully")
                else:
                    self.logger.info(f"Process {name} already finished with exit code {process.returncode}")

            # Clear the processes list
            self.processes.clear()
            self.logger.info("All services have been shutdown")

        except Exception as e:
            self.logger.error(f"Critical error during shutdown: {e}")
            raise

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, required=True, help="Path to configuration file")

    args = argparser.parse_args()
    config_path = Path(args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    try:
        # Create server manager
        manager = ServerManager(config)
        os.chdir(manager.config.base_dir)
        manager.start_controller()
        manager.start_all_workers()

        logger = logging.getLogger(__name__)
        logger.info("All services started. Press Ctrl+C to shutdown.")

        try:
            # Keep running
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Shutting down services...")
            manager.shutdown_services()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
