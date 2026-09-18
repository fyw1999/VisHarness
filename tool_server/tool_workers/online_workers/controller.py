"""
A controller manages distributed workers.
It sends worker addresses to clients.
"""
import sys
import uvicorn
import requests
import numpy as np
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, Request
import threading
from typing import List, Union
import time
import logging
import json
from enum import Enum, auto
import dataclasses
import asyncio
import argparse
import os

from tool_server.tool_workers.online_workers.utils import build_logger, SERVER_ERROR_MSG
from tool_server.tool_workers.online_workers.constants import CONTROLLER_HEART_BEAT_EXPIRATION

logger = build_logger("controller", "controller.log")


class DispatchMethod(Enum):
    LOTTERY = auto()
    SHORTEST_QUEUE = auto()

    @classmethod
    def from_str(cls, name):
        if name == "lottery":
            return cls.LOTTERY
        elif name == "shortest_queue":
            return cls.SHORTEST_QUEUE
        else:
            raise ValueError(f"Invalid dispatch method")


@dataclasses.dataclass
class WorkerInfo:
    model_names: List[str]
    speed: int
    queue_length: int
    worker_addr: str
    check_heart_beat: bool
    last_heart_beat: float  # 修复：time.time() 返回的是 float


class Controller:
    def __init__(self, dispatch_method: str):
        # Dict[str -> WorkerInfo]
        self.worker_info = {}
        self.lock = threading.Lock()  # 🌟 新增：保护字典的互斥锁
        self.dispatch_method = DispatchMethod.from_str(dispatch_method)

        self.heart_beat_thread = threading.Thread(target=self.heart_beat_controller)
        self.heart_beat_thread.daemon = True # 设置为守护线程，主程序退出时自动关闭
        self.heart_beat_thread.start()

        logger.info("Init controller")

    def heart_beat_controller(self):
        while True:
            time.sleep(CONTROLLER_HEART_BEAT_EXPIRATION)
            self.remove_stable_workers_by_expiration()

    def register_worker(self, worker_name: str, check_heart_beat: bool,
                        worker_status: dict):
        with self.lock:  # 🌟 加锁
            if worker_name not in self.worker_info:
                logger.info(f"Register a new worker: {worker_name}")
            else:
                logger.info(f"Register an existing worker: {worker_name}")

            if not worker_status:
                # 🌟 修复：防止新 Worker 注册时抛出 KeyError
                if worker_name in self.worker_info:
                    worker_status = self.get_worker_status(self.worker_info[worker_name].worker_addr)
                else:
                    logger.error(f"Cannot register new worker {worker_name} without status.")
                    return False

            if not worker_status:
                return False

            self.worker_info[worker_name] = WorkerInfo(
                worker_status["model_names"], worker_status["speed"], worker_status["queue_length"], worker_status["worker_addr"],
                check_heart_beat, time.time())

        logger.info(f"Register done: {worker_name}, {worker_status}")
        return True

    def get_worker_status(self, worker_addr: str):
        try:
            r = requests.post(worker_addr + "/worker_get_status", timeout=5)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {worker_addr}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {worker_addr}, {r}")
            return None

        return r.json()

    def remove_worker(self, worker_name: str):
        with self.lock:  # 🌟 加锁
            if worker_name in self.worker_info:
                del self.worker_info[worker_name]

    def refresh_all_workers(self):
        with self.lock:  # 🌟 加锁
            old_info = dict(self.worker_info)
            self.worker_info = {}

        for w_name, w_info in old_info.items():
            if not self.register_worker(w_name, w_info.check_heart_beat, None):
                logger.info(f"Remove stale worker: {w_name}")

    def list_models(self):
        model_names = set()
        with self.lock:  # 🌟 加锁
            for w_name, w_info in self.worker_info.items():
                model_names.update(w_info.model_names)
        return list(model_names)

    def list_workers(self):
        worker_names = []
        with self.lock:  # 🌟 加锁
            for w_name, w_info in self.worker_info.items():
                worker_names.append(w_name)
        return worker_names

    def get_worker_address(self, model_name: str):
        with self.lock:  # 🌟 加锁，防止遍历时字典被后台线程修改
            if self.dispatch_method == DispatchMethod.LOTTERY:
                # 注：由于你目前的客户端依赖返回 {"worker_name", "worker_addr"}
                # 此 LOTTERY 分支的返回逻辑与客户端不兼容（返回了字符串，且可能进入死循环测试状态）
                # 建议在未来彻底删除此分支，只保留 SHORTEST_QUEUE
                worker_names = []
                worker_speeds = []
                for w_name, w_info in self.worker_info.items():
                    if model_name in w_info.model_names:
                        worker_names.append(w_name)
                        worker_speeds.append(w_info.speed)
                worker_speeds = np.array(worker_speeds, dtype=np.float32)
                norm = np.sum(worker_speeds)
                if norm < 1e-4:
                    return {"worker_name": "", "worker_addr": ""}
                worker_speeds = worker_speeds / norm
                if True:  # Directly return address
                    pt = np.random.choice(np.arange(len(worker_names)),
                                          p=worker_speeds)
                    worker_name = worker_names[pt]
                    # 强行适配客户端，但如果 LOTTERY 不用建议删除整块
                    return {"worker_name": worker_name, "worker_addr": self.worker_info[worker_name].worker_addr}

            elif self.dispatch_method == DispatchMethod.SHORTEST_QUEUE:
                worker_addrs = []
                worker_qlen = []
                worker_names = []
                for w_name, w_info in self.worker_info.items():
                    if model_name in w_info.model_names:
                        worker_names.append(w_name)
                        worker_addrs.append(w_info.worker_addr)
                        worker_qlen.append(w_info.queue_length / w_info.speed)

                if len(worker_addrs) == 0:
                    return {"worker_name": "", "worker_addr": ""}

                min_index = np.argmin(worker_qlen)
                worker_addr = worker_addrs[min_index]
                worker_name = worker_names[min_index]
                self.worker_info[worker_name].queue_length += 1
                logger.info(
                    f"names: {worker_names}, queue_lens: {worker_qlen}, ret: {worker_name}")
                return {"worker_name": worker_name, "worker_addr": worker_addr}
            else:
                raise ValueError(
                    f"Invalid dispatch method: {self.dispatch_method}")

    def receive_heart_beat(self, worker_name: str, queue_length: int):
        with self.lock:
            if worker_name not in self.worker_info:
                logger.info(f"Receive unknown heart beat. {worker_name}")
                return False

            self.worker_info[worker_name].queue_length = queue_length
            self.worker_info[worker_name].last_heart_beat = time.time()
            logger.info(f"Receive heart beat. {worker_name}")
            return True

    def remove_stable_workers_by_expiration(self):
        expire = time.time() - CONTROLLER_HEART_BEAT_EXPIRATION
        to_delete = []
        with self.lock:
            for worker_name, w_info in self.worker_info.items():
                if w_info.check_heart_beat and w_info.last_heart_beat < expire:
                    to_delete.append(worker_name)

        # remove_worker 内部有锁，所以这里不要包在上面的 with self.lock 里，防止死锁
        for worker_name in to_delete:
            self.remove_worker(worker_name)


app = FastAPI()


@app.post("/register_worker")
async def register_worker(request: Request):
    data = await request.json()
    controller.register_worker(
        data["worker_name"], data["check_heart_beat"],
        data.get("worker_status", None))
    return {"status": "ok"} # 增加返回状态


@app.post("/refresh_all_workers")
async def refresh_all_workers():
    controller.refresh_all_workers()
    return {"status": "ok"} # 修复：原代码接收了并不存在的返回值


@app.post("/list_models")
async def list_models():
    models = controller.list_models()
    return {"models": models}


@app.post("/list_workers")
async def list_workers():
    workers = controller.list_workers()
    return {"workers": workers}


@app.post("/get_worker_address")
async def get_worker_address(request: Request):
    data = await request.json()
    worker_info = controller.get_worker_address(data["model"])
    return worker_info


@app.post("/receive_heart_beat")
async def receive_heart_beat(request: Request):
    data = await request.json()
    exist = controller.receive_heart_beat(
        data["worker_name"], data["queue_length"])
    return {"exist": exist}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=20001)
    parser.add_argument("--dispatch-method", type=str, choices=[
        "lottery", "shortest_queue"], default="shortest_queue")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    controller = Controller(args.dispatch_method)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")