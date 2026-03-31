"""Captures and persists foreign-key relationships for the connected database.

Declared FKs are read from information_schema at startup.  Manual entries
already present in fk_snapshot.json (e.g. logical FKs not enforced by
constraints) are preserved so they survive a re-capture.

JSON format:
    {
        "schema.table": {
            "fk_column": "target_schema.target_table.target_column",
            ...
        },
        ...
    }
"""

import json
import os

from sqlalchemy import text

from db.connection import get_engine

FK_PATH = os.path.join(os.path.dirname(__file__), "..", "fk_snapshot.json")

_FK_QUERY = """
    SELECT
        kcu.table_schema || '.' || kcu.table_name   AS fk_table,
        kcu.column_name                              AS fk_col,
        ccu.table_schema || '.' || ccu.table_name   AS pk_table,
        ccu.column_name                              AS pk_col
    FROM information_schema.key_column_usage AS kcu
    JOIN information_schema.referential_constraints AS rc
        ON kcu.constraint_name   = rc.constraint_name
       AND kcu.constraint_schema = rc.constraint_schema
    JOIN information_schema.constraint_column_usage AS ccu
        ON rc.unique_constraint_name   = ccu.constraint_name
       AND rc.unique_constraint_schema = ccu.constraint_schema
    WHERE kcu.table_schema NOT IN ('pg_catalog', 'information_schema')
    ORDER BY fk_table, fk_col
"""


def _load() -> dict:
    """Return the current FK dict from disk, or {} on any error."""
    if os.path.exists(FK_PATH):
        try:
            with open(FK_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def capture_fk_snapshot() -> dict:
    """Query declared FK constraints from information_schema and save to fk_snapshot.json.

    Any existing manual entries (logical FKs not enforced by DB constraints)
    are preserved — they are only overwritten if the DB now declares the same
    column as a real constraint.

    Returns the merged FK dict.
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(_FK_QUERY)).fetchall()

    # Build from declared constraints
    declared: dict[str, dict[str, str]] = {}
    for fk_table, fk_col, pk_table, pk_col in rows:
        declared.setdefault(fk_table, {})[fk_col] = f"{pk_table}.{pk_col}"

    # Merge: manual entries win for tables/columns NOT covered by declared FKs
    existing = _load()
    for table, cols in existing.items():
        for col, target in cols.items():
            if col not in declared.get(table, {}):
                declared.setdefault(table, {})[col] = target

    with open(FK_PATH, "w", encoding="utf-8") as f:
        json.dump(declared, f, indent=2, sort_keys=True)

    return declared


def load_fk_snapshot() -> dict:
    """Load FK snapshot from disk, returning {} if missing or unreadable."""
    return _load()
