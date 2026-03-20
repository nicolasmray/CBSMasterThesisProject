"""SQL tool implementations — imported by the MCP server, never by agents directly."""

import json
import os

from sqlalchemy import text

from db.connection import get_engine

SNAPSHOT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "schema_snapshot.json"
)


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


def get_schema_snapshot() -> dict:
    """Read the schema snapshot JSON file and return it.

    Returns:
        dict mapping table_name -> list of {column_name, data_type}.
    """
    with open(SNAPSHOT_PATH, "r") as f:
        return json.load(f)
