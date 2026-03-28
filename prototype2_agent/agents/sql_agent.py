"""SQL specialist agent.

Generates and executes SQL queries against PostgreSQL via MCP tools.
Uses sqlglot for AST-based SQL validation and has a retry loop that falls
back to semantic_search for extra context on failure.
"""

import asyncio
import concurrent.futures
import json

import sqlglot
from langchain_core.messages import SystemMessage, HumanMessage
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from state import AgentState
from mcp_client import get_server_params, call_tool
from llm_config import invoke_with_retry
from db.schema_snapshot import get_compact_schema

# ── System prompt ─────────────────────────────────────────────────────────────
SQL_SYSTEM_PROMPT = """\
You are a SQL specialist agent for a PostgreSQL database.

You will be given:
- A database schema snapshot listing every table and its columns in the format:
    schema.table: col1(type1), col2(type2), ...
- The user's question.
- Optionally, extra documentation context from a knowledge base.

Your job:
1. Write a single, correct PostgreSQL SQL query that answers the user's question.
2. Use ONLY tables and columns that appear in the schema snapshot. NEVER invent or guess
   column names — if a column is not listed in the schema, do not use it.
   - If no pre-calculated revenue/amount column exists, derive it:
     line total = unitprice * orderqty * (1 - unitpricediscount)
3. Tables are listed as "schema.table" (e.g. "production.product", "sales.salesorderdetail").
   You MUST use fully-qualified names (schema.table) in your SQL.
4. Infer JOIN conditions by matching ID columns across tables:
   - Columns like "personid", "storeid", "employeeid" often reference the primary key
     ("businessentityid") of their corresponding table (person.person, sales.store,
     humanresources.employee). In this database, "businessentityid" is the universal
     primary key for people, stores, vendors, and employees.
   - Example: sales.customer.personid → person.person.businessentityid
   - Example: purchasing.purchaseorderheader.employeeid → humanresources.employee.businessentityid
6. When the user asks for names, labels, or descriptions, always JOIN to the table that
   contains that human-readable text — do not return just IDs. For customer names,
   join through person.person to get firstname and lastname.
7. For time-series queries, always use EXTRACT to return date parts as separate integer columns
   (e.g. year, month) and ORDER BY them ASC. Never use TO_CHAR or combined date strings.
   Example: EXTRACT(YEAR FROM orderdate)::int AS year, EXTRACT(MONTH FROM orderdate)::int AS month
8. For relative time ranges ("last N months", "past N days", "previous N years"):
   - NEVER use CURRENT_DATE or NOW() — the database may contain historical data and
     anchoring to today will return zero rows.
   - Anchor to the latest date in the relevant table using a subquery:
       (SELECT MAX(date_col) FROM schema.table)
   - Example — "last 12 months" on sales.salesorderheader.orderdate:
       WHERE soh.orderdate >= DATE_TRUNC('month', (SELECT MAX(orderdate) FROM sales.salesorderheader)) - INTERVAL '12 months'
         AND soh.orderdate <  DATE_TRUNC('month', (SELECT MAX(orderdate) FROM sales.salesorderheader))
   - INTERVAL syntax: always write the value and unit as ONE quoted string — INTERVAL '12 months',
     INTERVAL '1 year', INTERVAL '30 days'. Never write INTERVAL '12' MONTHS (invalid in PostgreSQL).
     The ONLY valid form is: INTERVAL '<number> <unit>' where both number and unit are inside the quotes.
   - Always include BOTH a lower bound AND an upper bound in the WHERE clause.
9. When the user asks to group or break down by a dimension (e.g. "by category", "by region",
   "by product"), always include that column in both SELECT and GROUP BY — never drop it.
   When the user provides a list of specific IDs or named entities and asks for analysis,
   comparison, or distribution across them, always include that entity's identifier (or name)
   in both SELECT and GROUP BY so results are broken down per entity — never collapse them
   into a single aggregate. Example: "for these customers [IDs] analyse revenue" →
   include customerid in SELECT and GROUP BY.
10. When the user says "by category" without further specification, prefer the product
    classification hierarchy (e.g. productcategory.name, productsubcategory.name) over
    promotional or offer descriptions (e.g. specialoffer.description). Only use offer/promotion
    tables if the user explicitly asks for offers, discounts, or promotions.
11. If the question contains a time filter ("last N months", "this year", etc.), you MUST include
    a WHERE clause that implements it. Never omit a time filter mentioned in the question.
12. Return ONLY the raw SQL query — no markdown fences, no explanation.
"""

SQL_RETRY_PROMPT = """\
The previous SQL query failed or returned no results.

Error/result: {error_info}

IMPORTANT: If the error mentions a column that does not exist, look up that column name in
the schema below and find which table actually contains it, or derive the value from columns
that do exist. Do NOT reuse the same column reference that caused the error.

Here is additional context from the knowledge base that may help:
{extra_context}

Schema:
{schema}

Original question: {question}

Write a corrected PostgreSQL SQL query. Return ONLY the raw SQL.
"""


def _validate_sql(sql: str) -> str:
    """Parse and transpile SQL to PostgreSQL dialect using sqlglot.

    Tries parsing as generic SQL first, then falls back to TSQL dialect
    (handles cases where the LLM generates SQL Server syntax like SELECT TOP N).

    Args:
        sql: Raw SQL string.

    Returns:
        Transpiled SQL string in Postgres dialect.

    Raises:
        sqlglot.errors.ParseError: If the SQL is syntactically invalid in all dialects.
    """
    for source_dialect in (None, "tsql"):
        try:
            parsed = sqlglot.parse_one(sql, dialect=source_dialect)
            return parsed.sql(
                dialect="postgres",
                unsupported_level=sqlglot.ErrorLevel.IGNORE,
            )
        except sqlglot.errors.ParseError:
            continue
        except Exception:
            # Transpilation failed (e.g. TO_CHAR format unsupported) — original SQL is fine
            return sql
    # If all dialects fail, raise the error from the default parser
    parsed = sqlglot.parse_one(sql)
    return parsed.sql(dialect="postgres", unsupported_level=sqlglot.ErrorLevel.IGNORE)


async def _run_sql(state: AgentState) -> AgentState:
    """Async implementation: generate, validate, and execute SQL via MCP."""
    user_query = state["user_query"]
    plan = state.get("plan", "")
    retry_count = state.get("retry_count", 0)

    server_params = get_server_params()
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with _mcp_session(read_stream, write_stream) as session:
            await session.initialize()

            # Schema is read directly from disk and cached in-process
            # (avoids spawning an MCP subprocess on every query).
            schema_str = get_compact_schema()

            # Initial SQL generation
            messages = [
                SystemMessage(content=SQL_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Schema:\n{schema_str}\n\n"
                        f"Plan: {plan}\n\n"
                        f"Question: {user_query}"
                    )
                ),
            ]

            response = invoke_with_retry("sql", messages)
            raw_sql = response.content.strip().strip("`").strip()
            # Remove markdown SQL fences if present
            if raw_sql.lower().startswith("sql"):
                raw_sql = raw_sql[3:].strip()

            # Retry loop
            while retry_count < 3:
                try:
                    validated_sql = _validate_sql(raw_sql)
                except Exception as e:
                    error_info = f"SQL parse error: {e}"
                    retry_count += 1
                    if retry_count >= 3:
                        return {
                            "sql_query": raw_sql,
                            "sql_result": [],
                            "error": error_info,
                            "retry_count": retry_count,
                            "schema_context": schema_str,
                        }
                    # Fetch extra context via semantic search
                    extra = await call_tool(
                        session,
                        "semantic_search",
                        {"query": raw_sql, "top_k": 3},
                    )
                    extra_ctx = "\n".join(str(c) for c in extra) if extra else "None."
                    retry_msg = SQL_RETRY_PROMPT.format(
                        error_info=error_info,
                        extra_context=extra_ctx,
                        schema=schema_str,
                        question=user_query,
                    )
                    messages = [
                        SystemMessage(content=SQL_SYSTEM_PROMPT),
                        HumanMessage(content=retry_msg),
                    ]
                    response = invoke_with_retry("sql", messages)
                    raw_sql = response.content.strip().strip("`").strip()
                    if raw_sql.lower().startswith("sql"):
                        raw_sql = raw_sql[3:].strip()
                    continue

                # Execute the validated SQL
                try:
                    result = await call_tool(
                        session, "run_sql_query", {"sql": validated_sql}
                    )
                except Exception as e:
                    error_info = f"SQL execution error: {e}"
                    retry_count += 1
                    if retry_count >= 3:
                        return {
                            "sql_query": validated_sql,
                            "sql_result": [],
                            "error": error_info,
                            "retry_count": retry_count,
                            "schema_context": schema_str,
                        }
                    extra = await call_tool(
                        session,
                        "semantic_search",
                        {"query": user_query, "top_k": 3},
                    )
                    extra_ctx = "\n".join(str(c) for c in extra) if extra else "None."
                    retry_msg = SQL_RETRY_PROMPT.format(
                        error_info=error_info,
                        extra_context=extra_ctx,
                        schema=schema_str,
                        question=user_query,
                    )
                    messages = [
                        SystemMessage(content=SQL_SYSTEM_PROMPT),
                        HumanMessage(content=retry_msg),
                    ]
                    response = invoke_with_retry("sql", messages)
                    raw_sql = response.content.strip().strip("`").strip()
                    if raw_sql.lower().startswith("sql"):
                        raw_sql = raw_sql[3:].strip()
                    continue

                # Normalise result to list[dict].
                # FastMCP returns a bare dict for single-row results instead of
                # a one-element list.  Wrap valid dicts; treat dicts with an
                # "error" key as DB-level errors and retry.
                if isinstance(result, dict):
                    if "error" not in result:
                        result = [result]  # single-row result — wrap and continue
                if result is not None and not isinstance(result, list):
                    if isinstance(result, dict) and "error" in result:
                        error_info = f"Database error: {result['error']}"
                    else:
                        error_info = f"Unexpected result from database: {result!r:.200}"
                    retry_count += 1
                    if retry_count >= 3:
                        return {
                            "sql_query": validated_sql,
                            "sql_result": [],
                            "error": error_info,
                            "retry_count": retry_count,
                            "schema_context": schema_str,
                        }
                    extra = await call_tool(
                        session,
                        "semantic_search",
                        {"query": user_query, "top_k": 3},
                    )
                    extra_ctx = "\n".join(str(c) for c in extra) if extra else "None."
                    retry_msg = SQL_RETRY_PROMPT.format(
                        error_info=error_info,
                        extra_context=extra_ctx,
                        schema=schema_str,
                        question=user_query,
                    )
                    messages = [
                        SystemMessage(content=SQL_SYSTEM_PROMPT),
                        HumanMessage(content=retry_msg),
                    ]
                    response = invoke_with_retry("sql", messages)
                    raw_sql = response.content.strip().strip("`").strip()
                    if raw_sql.lower().startswith("sql"):
                        raw_sql = raw_sql[3:].strip()
                    continue

                # Check for genuinely empty result (valid query, zero rows)
                if not result:
                    retry_count += 1
                    if retry_count >= 3:
                        return {
                            "sql_query": validated_sql,
                            "sql_result": [],
                            "error": "Query returned no results after 3 attempts.",
                            "retry_count": retry_count,
                            "schema_context": schema_str,
                        }
                    extra = await call_tool(
                        session,
                        "semantic_search",
                        {"query": user_query, "top_k": 3},
                    )
                    extra_ctx = "\n".join(str(c) for c in extra) if extra else "None."
                    retry_msg = SQL_RETRY_PROMPT.format(
                        error_info="Query returned empty results.",
                        extra_context=extra_ctx,
                        schema=schema_str,
                        question=user_query,
                    )
                    messages = [
                        SystemMessage(content=SQL_SYSTEM_PROMPT),
                        HumanMessage(content=retry_msg),
                    ]
                    response = invoke_with_retry("sql", messages)
                    raw_sql = response.content.strip().strip("`").strip()
                    if raw_sql.lower().startswith("sql"):
                        raw_sql = raw_sql[3:].strip()
                    continue

                # Success — result is a non-empty list of row dicts
                return {
                    "sql_query": validated_sql,
                    "sql_result": result,
                    "error": "",
                    "retry_count": retry_count,
                    "schema_context": schema_str,
                }

    # Should not reach here, but safety fallback
    return {
        "sql_query": "",
        "sql_result": [],
        "error": "SQL agent exhausted retries.",
        "retry_count": retry_count,
        "schema_context": "",
    }


class _mcp_session:
    """Async context manager wrapping ClientSession."""

    def __init__(self, read_stream, write_stream):
        self._session = ClientSession(read_stream, write_stream)

    async def __aenter__(self):
        await self._session.__aenter__()
        return self._session

    async def __aexit__(self, *args):
        await self._session.__aexit__(*args)


def sql_agent(state: AgentState) -> AgentState:
    """Generate, validate, and execute a SQL query via MCP tools.

    Uses sqlglot for AST-based validation and retries up to 3 times with
    semantic_search fallback for extra context on failures.

    Args:
        state: Current pipeline state with user_query, plan.

    Returns:
        Partial AgentState update with sql_query, sql_result, error, retry_count, schema_context.
    """
    # Run in a dedicated thread so asyncio.run() creates a completely fresh
    # event loop, isolated from Streamlit's own loop.  This avoids the
    # "unhandled errors in a TaskGroup" error on Windows without patching
    # the global event loop (which causes shutdown hangs with nest_asyncio).
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _run_sql(state)).result()
