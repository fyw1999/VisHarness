"""Thin verl adapter around the existing visual-tool manager."""

import copy
from typing import Any

import msgpack


class VisionToolControllerClient:
    """Pass verl tool calls to the original asynchronous ``ToolManager``."""

    _manager_cache: dict[tuple[str, int], Any] = {}

    def __init__(self, controller_url_location: str | None = None, max_consecutive_oom: int = 5) -> None:
        self.controller_url_location = controller_url_location
        self.max_consecutive_oom = int(max_consecutive_oom)
        if self.max_consecutive_oom <= 0:
            raise ValueError(
                f"max_consecutive_oom must be positive, got {self.max_consecutive_oom}"
            )
        self._manager = None

    @property
    def manager(self):
        if self._manager is not None:
            return self._manager
        cache_key = (self.controller_url_location or "", self.max_consecutive_oom)
        if cache_key not in self._manager_cache:
            from tool_server.tool_workers.tool_manager.base_manager import ToolManager

            self._manager_cache[cache_key] = ToolManager(
                controller_url_location=self.controller_url_location,
                max_consecutive_oom=self.max_consecutive_oom,
            )
        return self._manager_cache[cache_key]

    async def call(self, tool_name: str, tool_parameters: dict[str, Any]) -> Any:
        """Encode parameters and delegate routing and retries to ``ToolManager``."""
        payload = msgpack.packb(tool_parameters, use_bin_type=True)
        response = await self.manager.async_dynamic_call_tool(tool_name, payload)
        return copy.deepcopy(response)
