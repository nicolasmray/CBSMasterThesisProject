"""Real MCP server using the `mcp` package.

Runs as a standalone process (stdio transport) and exposes tools via the MCP protocol.
Agents connect to this server through an MCP client — they never import tool functions directly.

Launch:  python -m mcp_server.server
"""

import sys
import os

# Ensure the project root is on sys.path so that `db.*` imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp.server.fastmcp import FastMCP

from mcp_server.tools.sql_tools import run_sql_query as _run_sql_query

mcp = FastMCP("prototype2-bi-assistant")


@mcp.tool()
def run_sql_query(sql: str) -> list[dict]:
    """Execute a SQL query against PostgreSQL and return rows as list of dicts.

    Args:
        sql: The SQL query string to execute.
    """
    return _run_sql_query(sql)


if __name__ == "__main__":
    mcp.run(transport="stdio")
