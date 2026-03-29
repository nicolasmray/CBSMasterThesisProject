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
from db.schema_snapshot import get_compact_schema, check_date_in_range, get_db_date_range

# ── System prompt ─────────────────────────────────────────────────────────────
SQL_SYSTEM_PROMPT = """\
You are a SQL specialist agent for a PostgreSQL database.

You will be given:
- A database schema snapshot listing every table and its columns in the format:
    schema.table: col1(type1), col2(type2), ...
- The user's question.
- Optionally, extra documentation context from a knowledge base.

⚠️  CRITICAL — READ BEFORE WRITING ANY SQL:
Never reference a column that is not explicitly listed in the schema snapshot.
Do NOT assume convenience columns like "total", "revenue", "amount", "linetotal",
"lineamount", "line_revenue", "subtotal", or "extended_price" exist unless they appear in the schema.
If a value must be computed, derive it from columns that do exist.

sales.salesorderdetail contains exactly these revenue-related columns:
    unitprice, unitpricediscount, orderqty
Line revenue MUST always be written as:
    unitprice * orderqty * (1 - unitpricediscount)

Your job:
1. Write a single, correct PostgreSQL SQL query that answers the user's question.
2. Use ONLY tables and columns that appear in the schema snapshot. NEVER invent or guess
   column names — if a column is not listed in the schema, do not use it.
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
13. LIMIT rule — apply exactly one of these cases:
    a) Ranking intent — user uses "most", "top", "highest", "largest", "best",
       "worst", "lowest", "least", "biggest", "fewest":
       → add ORDER BY <metric> DESC/ASC and LIMIT 10 (or the number the user states).
    b) Distribution/breakdown intent — user uses "breakdown", "split", "distribution",
       "by region", "by category", "by product", "all", "every", "each", "show me":
       → return ALL rows, no LIMIT. The user wants the complete picture.
    c) All other queries with no explicit ranking or count signal:
       → no LIMIT unless the result set would obviously be unbounded (e.g. raw fact
         tables with millions of rows). Aggregated GROUP BY queries are fine without LIMIT.
    Never add LIMIT to a query just because it has an ORDER BY clause.
14. When the user asks for period-over-period change, growth, decline, or comparison
    to a previous period — phrased with "growth", "change", "increase", "decrease",
    "compared to previous", "month over month", "year over year", "MoM", "YoY",
    "trend", "evolution" — you MUST compute the delta in the SQL itself using window
    functions. Do NOT return only the raw metric and leave the comparison to the caller.
    Use a CTE to first aggregate by period, then apply LAG/LEAD in a second step:

    WITH period_data AS (
        SELECT <period_cols>, <metric> AS value
        FROM ...
        GROUP BY <period_cols>
    )
    SELECT
        <period_cols>,
        value,
        LAG(value) OVER (ORDER BY <period_cols>) AS prev_value,
        ROUND((
            (value - LAG(value) OVER (ORDER BY <period_cols>))
            / NULLIF(LAG(value) OVER (ORDER BY <period_cols>), 0) * 100
        )::numeric, 2) AS pct_change
    FROM period_data
    ORDER BY <period_cols>;

    Scoping — how many rows to return:
    - "compared to THE previous month/quarter/year" (singular, definite article "the"):
      the user wants ONLY the most recent period vs the one before it.
      Add ORDER BY <period_cols> DESC LIMIT 2 so only those two rows are returned.
    - "month over month", "how has it changed over time", "trend", "evolution",
      "each month", "every month", "previous months" (plural):
      the user wants the full historical series — return all rows without a LIMIT.

    - For month-over-month: partition by nothing, order by year, month.
    - For year-over-year by category: PARTITION BY category ORDER BY year.
    - Always include both the absolute change and the percentage change.
    - ROUND requires numeric type: always cast the expression with ::numeric before rounding,
      e.g. ROUND((expr)::numeric, 2). Never pass double precision directly to ROUND.
    - The first period row will have NULL for prev_value and pct_change — that is correct.
15. Geographic joins — when the user asks to break down by province, state, region, or address:
    Always prefer the shortest join path. If the order/transaction table already contains a
    direct foreign key to an address or location table, join through that key directly.
    Do NOT route through intermediate person or customer entity tables to reach an address —
    this creates unnecessary joins and alias conflicts that break the query.
"""

SQL_RETRY_PROMPT = """\
The previous SQL query failed or returned no results.

Error/result: {error_info}

IMPORTANT: If the error mentions a column that does not exist, look up that column name in
the schema below and find which table actually contains it, or derive the value from columns
that do exist. Do NOT reuse the same column reference that caused the error.

Reminder: never use a column that is not in the schema. If a value must be computed,
derive it from columns that do exist (e.g. line revenue = unitprice * orderqty * (1 - unitpricediscount)).

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

            # Check if query references dates outside the DB range
            date_warning = check_date_in_range(user_query)
            if date_warning:
                return {
                    "sql_query": "",
                    "sql_result": [],
                    "error": date_warning,
                    "retry_count": 0,
                    "schema_context": schema_str,
                }

            # Include DB date range in the prompt so the LLM knows valid years
            db_dates = get_db_date_range()
            date_hint = ""
            if db_dates.get("min_year") and db_dates.get("max_year"):
                date_hint = (
                    f"\n\nDatabase date range: {db_dates['min_date'][:10]} to "
                    f"{db_dates['max_date'][:10]} (years {db_dates['min_year']}–{db_dates['max_year']}). "
                    f"Only use dates within this range."
                )

            # Initial SQL generation
            messages = [
                SystemMessage(content=SQL_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Schema:\n{schema_str}\n\n"
                        f"Plan: {plan}{date_hint}\n\n"
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

                # Success — result is a non-empty list of row dicts.
                # Point-comparison queries ("compared to the previous month/quarter/year")
                # should return only the most recent row which already includes previous row.  The SQL returns rows in
                # ascending order, so [-1:] gives the current and previous period.
                if "compared to the previous" in user_query.lower():
                    result = result[-1:]

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
