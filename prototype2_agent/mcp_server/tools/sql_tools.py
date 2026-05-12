"""SQL tool implementations — registered as MCP tools in mcp_server/server.py."""

from sqlalchemy import text

from db.connection import get_engine


def run_sql_query(sql: str) -> list[dict]:
    """Execute a SQL query against PostgreSQL and return rows as list of dicts.

    Args:
        sql: The SQL query string to execute.

    Returns:
        List of dictionaries, one per row.
    """
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        columns = list(result.keys())
        rows = [dict(zip(columns, row)) for row in result.fetchall()]
    return rows
