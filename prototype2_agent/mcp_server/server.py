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
from mcp_server.tools.sql_tools import get_schema_snapshot as _get_schema_snapshot
from mcp_server.tools.rag_tools import semantic_search as _semantic_search
from mcp_server.tools.rag_tools import embed_and_store as _embed_and_store

mcp = FastMCP("prototype2-bi-assistant")


@mcp.tool()
def run_sql_query(sql: str) -> list[dict]:
    """Execute a SQL query against PostgreSQL and return rows as list of dicts.

    Args:
        sql: The SQL query string to execute.
    """
    return _run_sql_query(sql)


@mcp.tool()
def get_schema_snapshot() -> dict:
    """Return the database schema snapshot (table names, columns, data types)."""
    return _get_schema_snapshot()


@mcp.tool()
def semantic_search(query: str, top_k: int = 5) -> list[str]:
    """Search the document store for chunks most relevant to the query.

    Args:
        query: Natural-language search query.
        top_k: Number of results to return (default 5).
    """
    return _semantic_search(query, top_k)


@mcp.tool()
def embed_and_store(text: str, meta: dict) -> bool:
    """Embed a text chunk and store it in the pgvector document table.

    Args:
        text: The text content to embed.
        meta: JSON-serializable metadata dict.
    """
    return _embed_and_store(text, meta)


if __name__ == "__main__":
    mcp.run(transport="stdio")
