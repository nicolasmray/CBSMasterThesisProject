"""Shared MCP client helper.

Provides helpers for agents to connect to the MCP server subprocess and call tools
via the MCP protocol. Agents must use this module instead of importing tool functions.
"""

import json
import sys
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _get_server_script_path() -> str:
    """Return the absolute path to the MCP server entry-point."""
    return os.path.join(os.path.dirname(__file__), "mcp_server", "server.py")


def get_server_params() -> StdioServerParameters:
    """Build StdioServerParameters pointing at the MCP server script."""
    return StdioServerParameters(
        command=sys.executable,
        args=[_get_server_script_path()],
        env={**os.environ},
    )


async def call_tool(session: ClientSession, tool_name: str, args: dict) -> any:
    """Invoke a tool on the MCP server and return the parsed result.

    Args:
        session: An active MCP ClientSession.
        tool_name: Name of the registered MCP tool.
        args: Dict of keyword arguments to pass to the tool.

    Returns:
        The parsed tool result (usually a Python object deserialized from JSON).
    """
    result = await session.call_tool(tool_name, arguments=args)

    # The MCP SDK returns content blocks. When the tool returns a list,
    # each element is serialized as a separate content block.
    if not result.content or len(result.content) == 0:
        return None

    if len(result.content) == 1:
        # Single content block: return the parsed value as-is (dict, str, bool, etc.)
        text = result.content[0].text
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text

    # Multiple content blocks: the tool returned a list and MCP split each
    # element into its own block. Reassemble as a list, preserving order.
    items = []
    for block in result.content:
        try:
            items.append(json.loads(block.text))
        except (json.JSONDecodeError, TypeError):
            items.append(block.text)
    return items


async def list_tools(session: ClientSession) -> list[str]:
    """Return the names of all tools registered on the MCP server.

    Args:
        session: An active MCP ClientSession.

    Returns:
        List of tool name strings.
    """
    result = await session.list_tools()
    return [tool.name for tool in result.tools]
