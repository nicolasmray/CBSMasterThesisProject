"""Queries the database on startup and saves table/column metadata to schema_snapshot.json."""

import json
import os

from sqlalchemy import text

from db.connection import get_engine

SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "..", "schema_snapshot.json")


def capture_schema_snapshot() -> dict:
    """Query information_schema.columns for all non-system tables and save to JSON.

    Returns:
        dict mapping "schema.table" -> list of {column_name, data_type}.
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
<<<<<<< HEAD
            SELECT table_schema || '.' || table_name AS table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
=======
            SELECT table_schema, table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema', 'public', 'hr', 'pe', 'pr', 'pu', 'sa')
>>>>>>> origin/main
            ORDER BY table_schema, table_name, ordinal_position
        """)).fetchall()

    tables: dict[str, list[dict]] = {}
    for table_schema, table_name, column_name, data_type in rows:
        key = f"{table_schema}.{table_name}"
        tables.setdefault(key, []).append({
            "column_name": column_name,
            "data_type": data_type,
        })

    # Capture foreign key relationships using pg_constraint (handles cross-schema FKs)
    with engine.connect() as conn:
        fk_rows = conn.execute(text("""
            SELECT
                n.nspname || '.' || c2.relname AS source_table,
                a1.attname AS source_column,
                n2.nspname || '.' || c3.relname AS target_table,
                a2.attname AS target_column
            FROM pg_constraint c
            JOIN pg_class c2 ON c.conrelid = c2.oid
            JOIN pg_namespace n ON c2.relnamespace = n.oid
            JOIN pg_class c3 ON c.confrelid = c3.oid
            JOIN pg_namespace n2 ON c3.relnamespace = n2.oid
            JOIN pg_attribute a1 ON a1.attrelid = c.conrelid AND a1.attnum = ANY(c.conkey)
            JOIN pg_attribute a2 ON a2.attrelid = c.confrelid AND a2.attnum = ANY(c.confkey)
            WHERE c.contype = 'f'
              AND n.nspname NOT IN ('pg_catalog', 'information_schema', 'public', 'hr', 'pe', 'pr', 'pu', 'sa')
            ORDER BY source_table, source_column
        """)).fetchall()

    foreign_keys = []
    for source_table, source_column, target_table, target_column in fk_rows:
        foreign_keys.append({
            "source": f"{source_table}.{source_column}",
            "target": f"{target_table}.{target_column}",
        })

    schema = {"tables": tables, "foreign_keys": foreign_keys}

    with open(SNAPSHOT_PATH, "w") as f:
        json.dump(schema, f, indent=2)

    return schema


def load_schema_snapshot() -> dict:
    """Load the schema snapshot from disk.

    Returns:
        dict mapping table_name -> list of {column_name, data_type}.
    """
    with open(SNAPSHOT_PATH, "r") as f:
        return json.load(f)
