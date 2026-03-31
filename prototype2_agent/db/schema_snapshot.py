"""Queries the database on startup and saves table/column metadata to schema_snapshot.json."""

import json
import os

from sqlalchemy import text

from db.connection import get_engine
from db.banned_columns import get_table_annotations
from db.fk_snapshot import load_fk_snapshot

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


def column_exists_anywhere(col_name: str) -> list[str]:
    """Return a list of 'schema.table' names that have a column called col_name.

    Case-insensitive. Returns an empty list if the column exists nowhere.
    Used to distinguish "data not available" (empty) from "wrong table" (non-empty).
    """
    schema = load_schema_snapshot()
    col_lower = col_name.lower()
    return [
        table
        for table, cols in schema.items()
        if any(c["column_name"].lower() == col_lower for c in cols)
    ]


def invalidate_compact_schema_cache() -> None:
    """Force get_compact_schema() to rebuild on its next call.

    Call this after new entries are added to banned_columns.json so the
    updated table annotations appear in the next SQL generation prompt.
    """
    global _compact_schema_cache
    _compact_schema_cache = None


def get_compact_schema() -> str:
    """Return the compact schema string, building and caching it on first call.

    Format: one line per table — "schema.table: col(type), col(type), ... [NOTE: ...]"
    Table-level notes from banned_columns.json are appended inline so the LLM
    sees the prohibition at the exact point it is considering that table.
    """
    global _compact_schema_cache
    if _compact_schema_cache is not None:
        return _compact_schema_cache

    schema = load_schema_snapshot()
    annotations = get_table_annotations()
    fks = load_fk_snapshot()  # {table: {col: target}}
    lines = []
    for table, cols in schema.items():
        table_fks = fks.get(table, {})
        col_defs = ", ".join(
            f"{c['column_name']}({c['data_type']})"
            + (f"[→{table_fks[c['column_name']]}]" if c['column_name'] in table_fks else "")
            for c in cols
        )
        note = f"  {annotations[table]}" if table in annotations else ""
        lines.append(f"{table}: {col_defs}{note}")
    _compact_schema_cache = "\n".join(lines)
    return _compact_schema_cache


# ── Database date range (cached) ──────────────────────────────────────────────
_date_range_cache: dict | None = None


def get_db_date_range() -> dict:
    """Query and cache the min/max dates from key tables in the database.

    Returns:
        Dict with 'min_year', 'max_year', 'min_date', 'max_date' and
        per-table ranges in 'tables'.
    """
    global _date_range_cache
    if _date_range_cache is not None:
        return _date_range_cache

    engine = get_engine()

    # Key date columns across the main business tables
    date_queries = [
        ("sales.salesorderheader", "orderdate"),
        ("purchasing.purchaseorderheader", "orderdate"),
        ("production.workorder", "startdate"),
        ("production.transactionhistory", "transactiondate"),
    ]

    tables = {}
    global_min = None
    global_max = None

    with engine.connect() as conn:
        for table, col in date_queries:
            try:
                row = conn.execute(text(
                    f"SELECT MIN({col}), MAX({col}) FROM {table}"
                )).fetchone()
                if row and row[0] and row[1]:
                    tables[table] = {
                        "column": col,
                        "min_date": str(row[0]),
                        "max_date": str(row[1]),
                    }
                    if global_min is None or row[0] < global_min:
                        global_min = row[0]
                    if global_max is None or row[1] > global_max:
                        global_max = row[1]
            except Exception:
                continue

    _date_range_cache = {
        "min_year": global_min.year if global_min else None,
        "max_year": global_max.year if global_max else None,
        "min_date": str(global_min) if global_min else None,
        "max_date": str(global_max) if global_max else None,
        "tables": tables,
    }
    return _date_range_cache


def check_date_in_range(user_query: str) -> str | None:
    """Check if the user query mentions a year outside the database range.

    Returns:
        A warning message string if the year is out of range, or None if OK.
    """
    import re

    date_range = get_db_date_range()
    min_year = date_range.get("min_year")
    max_year = date_range.get("max_year")

    if not min_year or not max_year:
        return None

    # Extract 4-digit years from the query
    years_mentioned = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", user_query)]

    if not years_mentioned:
        return None

    out_of_range = [y for y in years_mentioned if y < min_year or y > max_year]

    if out_of_range:
        return (
            f"The year(s) {', '.join(str(y) for y in out_of_range)} appear to be outside "
            f"the database date range ({min_year}–{max_year}). "
            f"The database contains data from {date_range['min_date'][:10]} "
            f"to {date_range['max_date'][:10]}. "
            f"Please adjust your query to use a year within this range."
        )

    return None
