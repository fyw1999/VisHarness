import asyncio

import msgpack
import pytest

from visharness.tools.controller_client import VisionToolControllerClient
from visharness.tools.errors import ToolOOMRetriesExhaustedError
from visharness.tools.visual_tool import ControllerBackedVisualTool


class FakeManager:
    def __init__(self):
        self.calls = []

    async def async_dynamic_call_tool(self, tool_name, payload):
        self.calls.append((tool_name, msgpack.unpackb(payload, raw=False)))
        return {"status": "success", "results": {"ok": True}}


class FakeClient:
    def __init__(self):
        self.calls = []

    async def call(self, tool_name, tool_parameters):
        self.calls.append((tool_name, tool_parameters))
        return {"status": "success", "results": {"points": [1, 2]}}


class FakeOOMClient:
    async def call(self, tool_name, tool_parameters):
        return {
            "status": "error",
            "error_type": "tool_oom_retry_exhausted",
            "tool_name": tool_name,
            "oom_attempts": 5,
            "oom_worker_history": [
                {"worker_name": "PointToBoxMask_h20_2", "worker_addr": "http://localhost:8103"}
            ],
            "message": "five consecutive OOM responses",
        }


def test_controller_client_passes_name_and_msgpacked_tool_parameters():
    client = VisionToolControllerClient()
    manager = FakeManager()
    client._manager = manager

    result = asyncio.run(client.call("PhraseToPoint", {"phrase": "person"}))

    assert result == {"status": "success", "results": {"ok": True}}
    assert manager.calls == [("PhraseToPoint", {"phrase": "person"})]


def test_controller_backed_visual_tool_adapts_result_for_agent_loop():
    tool = ControllerBackedVisualTool(
        config={"type": "native", "name": "PhraseToPoint"},
        tool_schema=None,
    )
    client = FakeClient()
    tool.client = client

    response, reward, extra = asyncio.run(
        tool.execute(
            instance_id="trajectory-1",
            parameters={"phrase": "person", "image_dict": {"img_0": b"jpeg"}},
        )
    )

    assert response.is_empty()
    assert reward == 0.0
    assert client.calls == [
        ("PhraseToPoint", {"phrase": "person", "image_dict": {"img_0": b"jpeg"}}),
    ]
    assert extra == {
        "tool_name": "PhraseToPoint",
        "tool_result": {"points": [1, 2]},
    }


def test_controller_backed_visual_tool_raises_typed_error_after_oom_limit():
    tool = ControllerBackedVisualTool(
        config={"type": "native", "name": "PointToBoxMask", "max_consecutive_oom": 5},
        tool_schema=None,
    )
    tool.client = FakeOOMClient()

    with pytest.raises(ToolOOMRetriesExhaustedError) as exc_info:
        asyncio.run(
            tool.execute(
                instance_id="trajectory-oom",
                parameters={"mode": "area", "image_dict": {}},
            )
        )

    assert exc_info.value.tool_name == "PointToBoxMask"
    assert exc_info.value.oom_attempts == 5
    assert exc_info.value.worker_history[0]["worker_name"] == "PointToBoxMask_h20_2"
