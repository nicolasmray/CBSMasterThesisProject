"""Queries the database on startup and saves table/column metadata to schema_snapshot.json."""

import json
import os

from sqlalchemy import text

from db.connection import get_engine

SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "..", "schema_snapshot.json")


def capture_schema_snapshot() -> dict:
    """Query information_schema.columns for all non-system tables and save to JSON.

    Returns:
        Flat dict mapping "schema.table" -> list of {column_name, data_type}.
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT table_schema, table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY table_schema, table_name, ordinal_position
        """)).fetchall()

    tables: dict[str, list[dict]] = {}
    for table_schema, table_name, column_name, data_type in rows:
        key = f"{table_schema}.{table_name}"
        tables.setdefault(key, []).append({
            "column_name": column_name,
            "data_type": data_type,
        })

    with open(SNAPSHOT_PATH, "w") as f:
        json.dump(tables, f, indent=2)

    return tables


def load_schema_snapshot() -> dict:
    """Load the schema snapshot from disk.

    Returns:
        dict mapping table_name -> list of {column_name, data_type}.
    """
    with open(SNAPSHOT_PATH, "r") as f:
        return json.load(f)


# ── Cached compact schema string ─────────────────────────────────────────────
_compact_schema_cache: str | None = None


def get_compact_schema() -> str:
    """Return the compact schema string, building and caching it on first call.

    Format: one line per table — "schema.table: col(type), col(type), ..."
    """
    global _compact_schema_cache
    if _compact_schema_cache is not None:
        return _compact_schema_cache

    schema = load_schema_snapshot()
    lines = []
    for table, cols in schema.items():
        col_defs = ", ".join(
            f"{c['column_name']}({c['data_type']})" for c in cols
        )
        lines.append(f"{table}: {col_defs}")
    _compact_schema_cache = "\n".join(lines)
    return _compact_schema_cache
