"""Expose the canonical VisHarness tool schemas to verl."""

from visharness.prompts import TOOLS_LIST


TOOL_SCHEMAS = {tool["function"]["name"]: tool for tool in TOOLS_LIST}
