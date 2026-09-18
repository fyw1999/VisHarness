import os
import requests
from tool_server.utils.utils import load_json_file
from tool_server.utils.server_utils import build_logger
from contextlib import contextmanager
import time
import msgpack
import aiohttp
import asyncio
from collections import Counter
from tool_server.tool_workers.oom_utils import response_indicates_cuda_oom
logger = build_logger("tool_manager")


class ToolTransportError(RuntimeError):
    """The controller or worker could not be reached."""


class ToolWorkerProtocolError(RuntimeError):
    """A worker returned an invalid HTTP/msgpack response."""


class ToolWorkerExecutionError(RuntimeError):
    """A worker reported a non-OOM execution failure."""


def _is_oom_response(ret_message):
    return response_indicates_cuda_oom(ret_message)


def _validate_worker_response(content, *, tool_name, worker_name, worker_addr):
    try:
        ret_message = msgpack.unpackb(content, raw=False)
    except Exception as error:
        preview = repr(content[:256])
        raise ToolWorkerProtocolError(
            f"Worker {worker_name} ({worker_addr}) returned invalid msgpack for "
            f"tool {tool_name}. body_prefix={preview}"
        ) from error

    if not isinstance(ret_message, dict):
        raise ToolWorkerProtocolError(
            f"Worker {worker_name} ({worker_addr}) returned {type(ret_message).__name__} "
            f"instead of a mapping for tool {tool_name}: {ret_message!r}"
        )
    if ret_message.get("status") not in {"success", "error"}:
        raise ToolWorkerProtocolError(
            f"Worker {worker_name} ({worker_addr}) returned an invalid or missing "
            f"status for tool {tool_name}: {ret_message!r}"
        )
    return ret_message


def _raise_worker_execution_error(tool_name, worker_name, worker_addr, response):
    message = response.get("message", "Worker returned an error without a message")
    exception_type = response.get("exception_type")
    remote_traceback = response.get("remote_traceback")
    details = [
        f"Tool {tool_name} failed on worker {worker_name} ({worker_addr})",
        f"{exception_type + ': ' if exception_type else ''}{message}",
    ]
    if remote_traceback:
        details.append(f"Remote worker traceback:\n{remote_traceback}")
    raise ToolWorkerExecutionError("\n".join(details))


def _oom_retry_exhausted_response(tool_name, attempts, worker_history, last_response):
    return {
        "status": "error",
        "error_type": "tool_oom_retry_exhausted",
        "error_code": 50002,
        "message": (
            f"Tool {tool_name} failed with CUDA OOM {attempts} consecutive times; "
            "the tool call has been stopped."
        ),
        "tool_name": tool_name,
        "oom_attempts": attempts,
        "oom_worker_history": worker_history,
        "last_oom_message": str(last_response.get("message", "")),
    }


class ToolManager(object):
    def __init__(self, controller_url_location=None, max_consecutive_oom=5):
        self.controller_url_location = controller_url_location
        self.max_consecutive_oom = int(max_consecutive_oom)
        if self.max_consecutive_oom <= 0:
            raise ValueError(
                f"max_consecutive_oom must be positive, got {self.max_consecutive_oom}"
            )
        self.init_online_tools(self.controller_url_location)
        logger.info(f"ToolManager is initialized.")
        self.headers = {"VisionAgent": "Client"}

    def init_online_tools(self, controller_url_location=None):
        self.available_tools = []
        if controller_url_location is None:
            current_file_path = os.path.dirname(os.path.abspath(__file__))
            self.controller_addr_location = f"{current_file_path}/../online_workers/controller_addr/controller_addr.json"
            logger.info("controller_addr is None, using default from controller_addr_location")
        else:
            self.controller_addr_location = controller_url_location
            logger.info(f"controller_addr exsits, controller_url_location is {controller_url_location}")

        if os.path.exists(self.controller_addr_location):
            self.controller_addr = load_json_file(self.controller_addr_location)["controller_addr"]
        else:
            self.controller_addr = self.controller_addr_location

        if self.controller_addr is not None and isinstance(self.controller_addr,str):
            session = requests.Session()
            session.trust_env = False
            ret = session.post(self.controller_addr + "/list_models")
            models = ret.json()["models"]
            if Counter(models) != Counter(["SuperResolution", "SplitImageIntoPatches", "PhraseToPoint", "PhraseToBoxMask", "PointToBoxMask", "MergeBoxMask"]):
                raise ValueError(f"ToolManager init_online_tools failed, tool list is not right: {models}")
            logger.info(f"Online Tools: {models}")
            self.available_tools = models
            ret = session.post(self.controller_addr + "/list_workers")
            workers = ret.json()["workers"]
            logger.info(f"Online Workers: {workers}")
        else:
            raise ValueError("Invalid controller address")

    def dynamic_call_tool(self, tool_name, params):
        consecutive_oom_count = 0
        oom_worker_history = []
        with requests.Session() as session:
            session.trust_env = False
            while True:
                try:
                    addr_response = session.post(
                        f"{self.controller_addr}/get_worker_address",
                        json={"model": tool_name},
                        timeout=None,
                    )
                    addr_response.raise_for_status()
                    tool_worker_info = addr_response.json()
                except requests.exceptions.RequestException as error:
                    raise ToolTransportError(
                        f"Cannot access controller {self.controller_addr} for tool "
                        f"{tool_name}: {error}"
                    ) from error
                except (TypeError, ValueError) as error:
                    raise ToolWorkerProtocolError(
                        f"Controller {self.controller_addr} returned invalid JSON for "
                        f"tool {tool_name}: {addr_response.text[:256]!r}"
                    ) from error

                tool_worker_name = tool_worker_info.get("worker_name")
                tool_worker_addr = tool_worker_info.get("worker_addr")

                if not tool_worker_name or not tool_worker_addr:
                    raise ToolWorkerProtocolError(
                        f"Controller can not find a valid worker for tool {tool_name}: "
                        f"{tool_worker_info!r}"
                    )

                logger.info(f"ready to call worker {tool_worker_name} at address {tool_worker_addr}")

                try:
                    ret = session.post(
                        f"{tool_worker_addr}/worker_generate",
                        headers=self.headers,
                        data=params,
                        timeout=None,
                    )
                except requests.exceptions.RequestException as error:
                    raise ToolTransportError(
                        f"Cannot access worker {tool_worker_name} ({tool_worker_addr}) "
                        f"for tool {tool_name}: {error}"
                    ) from error

                ret_message = _validate_worker_response(
                    ret.content,
                    tool_name=tool_name,
                    worker_name=tool_worker_name,
                    worker_addr=tool_worker_addr,
                )
                status = ret_message["status"]
                if ret.status_code >= 400 and status == "success":
                    raise ToolWorkerProtocolError(
                        f"Worker {tool_worker_name} ({tool_worker_addr}) returned HTTP "
                        f"{ret.status_code} with a success payload for tool {tool_name}"
                    )

                if status != "success":
                    if _is_oom_response(ret_message):
                        consecutive_oom_count += 1
                        oom_worker_history.append(
                            {
                                "worker_name": tool_worker_name,
                                "worker_addr": tool_worker_addr,
                            }
                        )
                        if consecutive_oom_count >= self.max_consecutive_oom:
                            logger.error(
                                f"Tool {tool_name} reached the consecutive OOM limit "
                                f"({self.max_consecutive_oom}); stop retrying."
                            )
                            return _oom_retry_exhausted_response(
                                tool_name,
                                consecutive_oom_count,
                                oom_worker_history,
                                ret_message,
                            )
                        logger.warning(f"Worker {tool_worker_name} OutOfMemoryError, request a new address from Controller again...")
                        time.sleep(5)
                        continue
                    _raise_worker_execution_error(
                        tool_name,
                        tool_worker_name,
                        tool_worker_addr,
                        ret_message,
                    )

                return ret_message

    async def async_dynamic_call_tool(self, tool_name, params):
        consecutive_oom_count = 0
        oom_worker_history = []
        # Disable every client-side timeout. A tool request may wait in a
        # worker queue or run inference for an unlimited amount of time.
        client_timeout = aiohttp.ClientTimeout(total=None)
        async with aiohttp.ClientSession(
            trust_env=False,
            timeout=client_timeout,
        ) as session:
            while True:
                try:
                    async with session.post(
                        f"{self.controller_addr}/get_worker_address",
                        json={"model": tool_name},
                    ) as addr_response:
                        if addr_response.status != 200:
                            body = await addr_response.read()
                            raise ToolTransportError(
                                f"Controller {self.controller_addr} returned HTTP "
                                f"{addr_response.status} for tool {tool_name}: "
                                f"{body[:256]!r}"
                            )
                        tool_worker_info = await addr_response.json()
                except aiohttp.ClientError as error:
                    raise ToolTransportError(
                        f"Cannot access controller {self.controller_addr} for tool "
                        f"{tool_name}: {error}"
                    ) from error
                except (TypeError, ValueError) as error:
                    raise ToolWorkerProtocolError(
                        f"Controller {self.controller_addr} returned invalid JSON for "
                        f"tool {tool_name}"
                    ) from error

                tool_worker_name = tool_worker_info.get("worker_name")
                tool_worker_addr = tool_worker_info.get("worker_addr")

                if not tool_worker_name or not tool_worker_addr:
                    raise ToolWorkerProtocolError(
                        f"Controller can not find a valid worker for tool {tool_name}: "
                        f"{tool_worker_info!r}"
                    )

                logger.info(f"ready to call worker {tool_worker_name} at address {tool_worker_addr}")

                try:
                    async with session.post(
                        f"{tool_worker_addr}/worker_generate",
                        headers=self.headers,
                        data=params,
                    ) as ret:
                        content = await ret.read()
                        http_status = ret.status
                except aiohttp.ClientError as error:
                    raise ToolTransportError(
                        f"Cannot access worker {tool_worker_name} ({tool_worker_addr}) "
                        f"for tool {tool_name}: {error}"
                    ) from error

                ret_message = _validate_worker_response(
                    content,
                    tool_name=tool_name,
                    worker_name=tool_worker_name,
                    worker_addr=tool_worker_addr,
                )
                status = ret_message["status"]
                if http_status >= 400 and status == "success":
                    raise ToolWorkerProtocolError(
                        f"Worker {tool_worker_name} ({tool_worker_addr}) returned HTTP "
                        f"{http_status} with a success payload for tool {tool_name}"
                    )

                if status != "success":
                    if _is_oom_response(ret_message):
                        consecutive_oom_count += 1
                        oom_worker_history.append(
                            {
                                "worker_name": tool_worker_name,
                                "worker_addr": tool_worker_addr,
                            }
                        )
                        if consecutive_oom_count >= self.max_consecutive_oom:
                            logger.error(
                                f"Tool {tool_name} reached the consecutive OOM limit "
                                f"({self.max_consecutive_oom}); stop retrying."
                            )
                            return _oom_retry_exhausted_response(
                                tool_name,
                                consecutive_oom_count,
                                oom_worker_history,
                                ret_message,
                            )
                        logger.warning(f"Worker {tool_worker_name} OutOfMemoryError, request a new address from Controller again...")
                        await asyncio.sleep(20)
                        continue
                    _raise_worker_execution_error(
                        tool_name,
                        tool_worker_name,
                        tool_worker_addr,
                        ret_message,
                    )

                return ret_message
