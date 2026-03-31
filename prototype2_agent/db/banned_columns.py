"""Persistent registry of confirmed-nonexistent columns.

When PostgreSQL returns an UndefinedColumn error the offending alias.column is
resolved to a fully-qualified schema.table.column using the failed SQL, then
persisted to banned_columns.json.

On every SQL generation:
  - get_banned_columns_prompt()  injects the list into the system prompt header
  - get_table_annotations()      appends inline warnings to affected schema lines
"""

import json
import os
import re

import sqlglot
import sqlglot.expressions as exp

BANNED_PATH = os.path.join(os.path.dirname(__file__), "..", "banned_columns.json")

_UNDEF_COL_RE = re.compile(
    r'column\s+"?([a-zA-Z_][a-zA-Z0-9_.]*)"?\s+does not exist',
    re.IGNORECASE,
)


# ── persistence ───────────────────────────────────────────────────────────────

def _load() -> dict:
    if os.path.exists(BANNED_PATH):
        try:
            with open(BANNED_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save(data: dict) -> None:
    with open(BANNED_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


# ── alias resolution ──────────────────────────────────────────────────────────

def _build_alias_map(sql: str) -> dict[str, tuple[str, str]]:
    """Return {alias: (schema, table)} for every FROM/JOIN table in sql."""
    try:
        tree = sqlglot.parse_one(sql)
    except Exception:
        return {}
    result = {}
    for t in tree.find_all(exp.Table):
        key = t.alias if t.alias else t.name
        result[key] = (t.db or "", t.name or "")
    return result


def _resolve_column(col_ref: str, alias_map: dict[str, tuple[str, str]]) -> str:
    """Resolve 'alias.col' to 'schema.table.col' using alias_map."""
    parts = col_ref.split(".")
    if len(parts) == 2:
        alias, col = parts
        if alias in alias_map:
            schema, table = alias_map[alias]
            if schema and table:
                return f"{schema}.{table}.{col}"
            if table:
                return f"{table}.{col}"
    return col_ref


# ── public API ────────────────────────────────────────────────────────────────

def record_banned_column(error_str: str, failed_sql: str = "") -> str | None:
    """Parse error_str for an UndefinedColumn, resolve it, and persist.

    Returns the fully-qualified key recorded, or None if no match.
    """
    m = _UNDEF_COL_RE.search(error_str)
    if not m:
        return None
    raw_ref = m.group(1)
    alias_map = _build_alias_map(failed_sql) if failed_sql else {}
    fq_ref = _resolve_column(raw_ref, alias_map)
    data = _load()
    if fq_ref not in data:
        data[fq_ref] = "UndefinedColumn confirmed by PostgreSQL"
        _save(data)
    return fq_ref


def get_table_annotations() -> dict[str, str]:
    """Return {schema.table: inline warning} for tables with banned columns."""
    data = _load()
    table_entries: dict[str, dict] = {}
    for fq_col, reason in data.items():
        parts = fq_col.rsplit(".", 1)
        if len(parts) == 2:
            table, col = parts
            entry = table_entries.setdefault(table, {"cols": [], "reason": reason})
            entry["cols"].append(col)
    return {
        table: f"[WARNING column(s) {', '.join(sorted(e['cols']))}: {e['reason']}]"
        for table, e in table_entries.items()
    }


def get_banned_columns_prompt() -> str:
    """Return a prompt block listing all banned columns, or '' if empty."""
    data = _load()
    if not data:
        return ""
    lines = [
        f"  - column '{fq.rsplit('.', 1)[1]}' on table '{fq.rsplit('.', 1)[0]}' does not exist"
        for fq in sorted(data)
        if "." in fq
    ]
    return (
        "BANNED COLUMNS — confirmed non-existent by the database.\n"
        "Never reference these columns regardless of alias used.\n"
        "If a metric cannot be found as a direct column, compute it from the underlying "
        "transaction tables (e.g. sales.salesorderheader, sales.salesorderdetail).\n"
        + "\n".join(lines) + "\n"
    )
