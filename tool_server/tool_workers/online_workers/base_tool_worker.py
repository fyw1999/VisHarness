"""
A model worker executes the model.
"""
import argparse
import asyncio
import json
import time
import threading
import uuid
import os
import psutil
from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse
import requests
import torch
import uvicorn
import msgpack
import traceback
from enum import IntEnum

from tool_server.tool_workers.oom_utils import (
    CUDA_OOM_ERROR_CODE,
    CUDA_OOM_ERROR_TYPE,
    message_indicates_cuda_oom,
)
from tool_server.utils.utils import *
from tool_server.utils.server_utils import *

SERVER_ERROR_MSG = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
class ErrorCode(IntEnum):
    """
    https://platform.openai.com/docs/guides/error-codes/api-errors
    """

    VALIDATION_TYPE_ERROR = 40001

    INVALID_AUTH_KEY = 40101
    INCORRECT_AUTH_KEY = 40102
    NO_PERMISSION = 40103

    INVALID_MODEL = 40301
    PARAM_OUT_OF_RANGE = 40302
    CONTEXT_OVERFLOW = 40303
    TIMEOUT_ERROR = 40304

    RATE_LIMIT = 42901
    QUOTA_EXCEEDED = 42902
    ENGINE_OVERLOADED = 42903

    INTERNAL_ERROR = 50001
    CUDA_OUT_OF_MEMORY = 50002
    GRADIO_REQUEST_ERROR = 50003
    GRADIO_STREAM_UNKNOWN_ERROR = 50004
    CONTROLLER_NO_WORKER = 50005
    CONTROLLER_WORKER_TIMEOUT = 50006


GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("tool_worker", f"base_tool_worker_{worker_id}.log")

def is_cuda_oom_error(error):
    cuda_oom_error = getattr(torch.cuda, "OutOfMemoryError", None)
    if cuda_oom_error is not None and isinstance(error, cuda_oom_error):
        return True
    return message_indicates_cuda_oom(error)


def build_worker_error_response(tool_name, error, remote_traceback=None):
    """Serialize an exception without changing whether it is retryable."""
    is_oom = is_cuda_oom_error(error)
    response = {
        "status": "error",
        "error_type": CUDA_OOM_ERROR_TYPE if is_oom else "tool_execution_error",
        "error_code": int(
            CUDA_OOM_ERROR_CODE if is_oom else ErrorCode.INTERNAL_ERROR
        ),
        "exception_type": type(error).__name__,
        "message": str(error),
        "tool_name": tool_name,
    }
    if remote_traceback and not is_oom:
        response["remote_traceback"] = remote_traceback
    return response

class BaseToolWorker:
    def __init__(self,
                 controller_addr,
                 worker_name = "",
                 worker_addr = "auto",
                 no_register = False,
                 tool_name = None,
                 limit_model_concurrency = 1,
                 host = "0.0.0.0",
                 port = None,
                 ):
        self.controller_addr = controller_addr
        assert port is not None, "Port must be specified"
        if worker_addr == "auto":
            node_name = "localhost"
            self.worker_addr = f"http://{node_name}:{port}"
        else:
            self.worker_addr = worker_addr

        assert tool_name is not None, "tool_name must be specified"
        self.tool_name = tool_name

        self.worker_name = worker_name
        self.limit_model_concurrency = limit_model_concurrency
        self.model_semaphore = asyncio.BoundedSemaphore(
            self.limit_model_concurrency
        )
        self.active_requests = 0
        self.queue_lock = threading.Lock()
        self.no_register = no_register


        self.host = host
        self.port = port

        self.global_counter = 0

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=self.heart_beat_worker)
            self.heart_beat_thread.daemon = True
            self.heart_beat_thread.start()

        # Set up the routes
        self.app = FastAPI()
        self.init_model()
        self.setup_routes()




    ## HTTP Methods
    def heart_beat_worker(self):
        while True:
            time.sleep(WORKER_HEART_BEAT_INTERVAL)
            self.send_heart_beat()


    def release_model_semaphore(self, fn=None):
        self.model_semaphore.release()
        with self.queue_lock:
            self.active_requests -= 1
        if fn is not None:
            fn()

    async def acquire_model_semaphore(self):
        self.global_counter += 1
        with self.queue_lock:
            self.active_requests += 1

        try:
            # Deliberately no timeout: queued inference is allowed to wait for
            # an unlimited amount of time.
            await self.model_semaphore.acquire()
        except BaseException:
            with self.queue_lock:
                self.active_requests -= 1
            raise

    def setup_routes(self):
        @self.app.post("/worker_generate")
        async def api_generate(request: Request):
            acquired = False
            try:
                body = await request.body()
                params = msgpack.unpackb(body, raw=False)
                if not isinstance(params, dict):
                    raise TypeError(
                        f"Worker parameters must be a mapping, got {type(params).__name__}"
                    )
                await self.acquire_model_semaphore()
                acquired = True
                output = await self.generate_gate_async(params)
                if not isinstance(output, dict):
                    raise TypeError(
                        f"{self.tool_name} returned {type(output).__name__}, expected dict"
                    )
                if output.get("status") not in {"success", "error"}:
                    raise ValueError(
                        f"{self.tool_name} returned an invalid status: {output!r}"
                    )
                packed_response = msgpack.packb(output, use_bin_type=True)
                return Response(
                    content=packed_response,
                    media_type="application/msgpack",
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                remote_traceback = traceback.format_exc()
                logger.error(
                    f"Unhandled {self.tool_name} worker request failure: {error}\n"
                    f"{remote_traceback}"
                )
                output = build_worker_error_response(
                    self.tool_name,
                    error,
                    remote_traceback=remote_traceback,
                )
                packed_response = msgpack.packb(output, use_bin_type=True)
                return Response(
                    content=packed_response,
                    media_type="application/msgpack",
                )
            finally:
                if acquired:
                    self.release_model_semaphore()


        @self.app.post("/worker_get_status")
        async def get_status(request: Request):
            return self.get_status()

        @self.app.post("/model_details")
        async def model_details(request: Request):
            pass

        @self.app.post("/tool_instruction")
        async def tool_instruction(request: Request):
            params = await request.json()
            try:
                tool_instruction = self.get_tool_instruction()
                return JSONResponse({
                    "tool_instruction": tool_instruction,
                    "error_code": 0
                })
            except Exception as e:
                logger.error(f"Error getting tool instruction: {e}")
                return JSONResponse({
                    "text": SERVER_ERROR_MSG,
                    "error_code": ErrorCode.INTERNAL_ERROR
                }, status_code=500)

    def generate_gate(self, params):
        try:
            ret = self.generate(params)
            # ret = asyncio.get_event_loop().run_until_complete(self.async_generate(params))
        except Exception as error:
            remote_traceback = traceback.format_exc()
            logger.error(
                f"{self.tool_name} generate failed: {error}\n{remote_traceback}"
            )
            ret = build_worker_error_response(
                self.tool_name,
                error,
                remote_traceback=remote_traceback,
            )
        return ret

    async def generate_gate_async(self, params):
        """Async-compatible generation entry point.

        Synchronous workers keep their existing behavior through
        ``generate_gate``. Workers backed by an asynchronous inference engine
        can override this method without changing the shared HTTP routes.
        """
        return self.generate_gate(params)


    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_name,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200, f"Failed to register to controller: {r.text}"

    def send_heart_beat(self):
        logger.info(f"Send heart beat. Worker: [{self.worker_name}]. "
                    f"Active requests: {self.active_requests}. "
                    f"global_counter: {self.global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_name,
                    "queue_length": self.get_queue_length()}, timeout=5)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        with self.queue_lock:
            return self.active_requests

    def get_status(self):
        return {
            "model_names": [self.tool_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
            "worker_addr": self.worker_addr,
        }

    # Launch method
    def run(self):
        self.release_port(self.port)
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info", log_config=None)

    # abstract methods
    def init_model(self):
        pass

    @torch.inference_mode()
    def generate(self, params):
        pass

    def get_tool_instruction(self):
        pass

    def release_port(self, port: int):
        """
        查找并强杀占用指定端口的进程
        """
        print(f"🔍 正在检查 {port} 端口是否被占用...")

        # 遍历当前机器上的所有网络连接
        for conn in psutil.net_connections(kind='inet'):
            # 筛选出本地占用该端口，且状态为 LISTEN（监听中）的连接
            if conn.laddr.port == port and conn.status == 'LISTEN':
                pid = conn.pid
                if pid:
                    try:
                        # 根据 PID 获取进程对象
                        process = psutil.Process(pid)
                        process_name = process.name()

                        print(f"⚠️ 发现进程 '{process_name}' (PID: {pid}) 正在占用 {port} 端口。")
                        print("🔪 正在尝试结束该进程...")

                        # 相当于执行 kill -9
                        process.kill()

                        # 等待进程完全退出，确保端口被彻底释放
                        process.wait(timeout=3)
                        print("✅ 端口清理完毕！\n")
                        return

                    except psutil.NoSuchProcess:
                        print("❌ 进程已经不存在了。")
                    except psutil.AccessDenied:
                        print("❌ 权限不足！请尝试用 sudo 或管理员权限运行此脚本。")
                    except Exception as e:
                        print(f"❌ 发生未知错误: {e}")

        print(f"✅ {port} 端口目前是空闲的，可以安全使用。\n")
