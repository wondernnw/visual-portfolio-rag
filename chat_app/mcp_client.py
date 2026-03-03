"""Sync bridge for Streamlit to call MCP tools.

Wraps the in-process FastMCP server with a synchronous interface
since Streamlit runs in a sync context.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional

from fastmcp import Client


class MCPBridge:
    """Synchronous wrapper around fastmcp.Client for Streamlit.

    Uses an in-process connection to the FastMCP server,
    sharing GPU state without needing a separate process.
    """

    def __init__(self, mcp_server):
        """Initialize with a FastMCP server instance.

        Args:
            mcp_server: The FastMCP server (from mcp_server.create_server()).
        """
        self._server = mcp_server
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create an event loop for sync execution."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def _run(self, coro):
        """Run an async coroutine synchronously."""
        loop = self._get_loop()
        return loop.run_until_complete(coro)

    async def _call_tool_async(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server asynchronously."""
        async with Client(self._server) as client:
            result = await client.call_tool(name, arguments)
            # fastmcp returns a list of content blocks; extract text
            if result and hasattr(result[0], "text"):
                return json.loads(result[0].text)
            return result

    async def _list_tools_async(self) -> List[Dict]:
        """List available tools asynchronously."""
        async with Client(self._server) as client:
            tools = await client.list_tools()
            return [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema,
                }
                for t in tools
            ]

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """Call an MCP tool synchronously.

        Args:
            name: Tool name (e.g., "index_portfolio").
            arguments: Tool arguments as a dict.

        Returns:
            Tool result (parsed from JSON).
        """
        return self._run(self._call_tool_async(name, arguments or {}))

    def list_tools(self) -> List[Dict]:
        """List all available MCP tools.

        Returns:
            List of tool definitions with name, description, and parameters.
        """
        return self._run(self._list_tools_async())

    def get_tool_schemas(self) -> List[Dict]:
        """Get tool schemas in OpenAI-compatible function format for Groq.

        Returns:
            List of tool definitions in OpenAI function calling format.
        """
        tools = self.list_tools()
        schemas = []
        for t in tools:
            params = t.get("parameters", {})
            # Remove 'title' fields that Groq doesn't expect
            cleaned = _clean_schema(params)
            schemas.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": cleaned,
                },
            })
        return schemas


def _clean_schema(schema: dict) -> dict:
    """Remove extra fields from JSON schema for Groq compatibility."""
    cleaned = {}
    for k, v in schema.items():
        if k == "title":
            continue
        if isinstance(v, dict):
            cleaned[k] = _clean_schema(v)
        else:
            cleaned[k] = v
    return cleaned
