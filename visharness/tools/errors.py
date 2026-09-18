"""Typed failures raised by VisHarness tool adapters."""

from typing import Any


class ToolOOMRetriesExhaustedError(RuntimeError):
    """Raised when one tool call reaches its consecutive CUDA OOM limit."""

    def __init__(self, response: dict[str, Any]):
        self.response = response
        self.tool_name = str(response.get("tool_name", "unknown"))
        self.oom_attempts = int(response.get("oom_attempts", 0))
        self.worker_history = list(response.get("oom_worker_history") or [])
        super().__init__(
            str(response.get("message", f"Tool {self.tool_name} exhausted its CUDA OOM retries"))
        )
