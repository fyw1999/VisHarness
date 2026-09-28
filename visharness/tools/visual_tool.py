"""verl BaseTool adapter for controller-backed visual tools."""

from typing import Any

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from .controller_client import VisionToolControllerClient
from .errors import ToolOOMRetriesExhaustedError
from .tool_schemas import TOOL_SCHEMAS


class ControllerBackedVisualTool(BaseTool):
    """Expose a controller-routed visual service through verl's ``BaseTool`` API."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema | None):
        super().__init__(config=config, tool_schema=tool_schema)
        self.client = VisionToolControllerClient(
            controller_url_location=config.get("controller_url_location"),
            max_consecutive_oom=int(config.get("max_consecutive_oom", 5)),
        )

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        tool_name = self.config.get("name")
        if tool_name not in TOOL_SCHEMAS:
            raise ValueError(f"Unknown VisHarness visual tool: {tool_name!r}")
        return OpenAIFunctionToolSchema.model_validate(TOOL_SCHEMAS[tool_name])

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        response = await self.client.call(
            tool_name=self.name,
            tool_parameters=parameters,
        )
        if response.get("status") != "success":
            if response.get("error_type") == "tool_oom_retry_exhausted":
                raise ToolOOMRetriesExhaustedError(response)
            raise RuntimeError(f"Visual tool {self.name} failed: {response.get('message', response)!r}")

        return ToolResponse(), 0.0, {
            "tool_name": self.name,
            "tool_result": response["results"],
        }
