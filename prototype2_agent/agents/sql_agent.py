"""SQL specialist agent.

Generates and executes SQL queries against PostgreSQL via MCP tools.
Uses sqlglot for AST-based SQL validation and has a retry loop that falls
back to semantic_search for extra context on failure.
"""

import asyncio
import concurrent.futures
import re

import sqlglot
from langchain_core.messages import SystemMessage, HumanMessage
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from sqlalchemy import text

from state import AgentState
from mcp_client import get_server_params, call_tool
from llm_config import invoke_with_retry
from db.connection import get_engine
from db.schema_snapshot import get_compact_schema, invalidate_compact_schema_cache
from db.banned_columns import get_banned_columns_prompt, record_banned_column
from db.vector_store import semantic_search

# ── System prompt ─────────────────────────────────────────────────────────────
SQL_SYSTEM_PROMPT = """\
You are a SQL specialist agent for a PostgreSQL database.

You will be given:
- A database schema snapshot listing every table and its columns in the format:
    schema.table: col1(type1), col2(type2), ...
- The user's question.
- Optionally, extra documentation context from a knowledge base.

Output: Return ONLY the raw SQL query — no markdown fences, no explanation.

⚠️  CRITICAL — READ BEFORE WRITING ANY SQL:
Never reference a column that is not explicitly listed in the schema snapshot.
Do NOT assume convenience columns like "total", "revenue", "amount", "linetotal",
"lineamount", "line_revenue", "subtotal", or "extended_price" exist unless they appear in the schema.
If a value must be computed, derive it from columns that do exist.

sales.salesorderdetail contains exactly these revenue-related columns:
    unitprice, unitpricediscount, orderqty
Line revenue MUST always be written as:
    unitprice * orderqty * (1 - unitpricediscount)
unitpricediscount is already a fraction (0.00–1.00). To get average discount as a percentage:
    ROUND(CAST(AVG(sod.unitpricediscount) * 100 AS numeric), 2)
Never derive the discount by dividing revenue — use the column directly.

Your job:

── Schema & table basics ────────────────────────────────────────────────────
1. Write a single, correct PostgreSQL SQL query that answers the user's question.
2. Use ONLY tables and columns that appear in the schema snapshot. NEVER invent or guess
   column names — if a column is not listed in the schema, do not use it.
3. Tables are listed as "schema.table" (e.g. "production.product", "sales.salesorderdetail").
   You MUST use fully-qualified names (schema.table) in your SQL.

── JOINs ────────────────────────────────────────────────────────────────────
4. Infer JOIN conditions by matching ID columns across tables:
   - Columns like "personid", "storeid", "employeeid" often reference the primary key
     ("businessentityid") of their corresponding table (person.person, sales.store,
     humanresources.employee). In this database, "businessentityid" is the universal
     primary key for people, stores, vendors, and employees.
   - Example: sales.customer.personid → person.person.businessentityid
   - Example: purchasing.purchaseorderheader.employeeid → humanresources.employee.businessentityid
5. Whenever a query groups or breaks down by any entity, always JOIN to get the
   human-readable name and use it instead of the raw ID — even if the user did not
   explicitly ask for "names". Never return bare numeric IDs as dimension labels.
   For customer names, join through person.person to get firstname and lastname.
14. JOIN column selection — always use the most specific foreign key available, not the
    most common one. When a table has a named FK column (e.g. salespersonid, storeid,
    customerid, vendorid) pointing to another table, use that named column for the JOIN.
    Only fall back to businessentityid when no named FK exists for the relationship.
    CORRECT:   JOIN sales.salesperson sp ON s.salespersonid = sp.businessentityid
    INCORRECT: JOIN sales.salesperson sp ON s.businessentityid = sp.businessentityid
    To find the right FK: look at the column names on the source table in the schema.
    A column like "salespersonid" on sales.store is the FK to sales.salesperson.
    A column like "customerid" on sales.salesorderheader is the FK to sales.customer.
    Never assume two tables join on businessentityid unless neither table has a more
    specific named FK for that relationship.
15. Geographic joins — when the user asks to break down by province, state, region, or address:
    Always prefer the shortest join path. If the order/transaction table already contains a
    direct foreign key to an address or location table, join through that key directly.
    Do NOT route through intermediate person or customer entity tables to reach an address —
    this creates unnecessary joins and alias conflicts that break the query.
16. Avoid implicit row filtering through joins — every JOIN you add that is not strictly
    needed for a selected column silently excludes rows where that FK is NULL.
    In particular:
    - Never join sales.salesperson just to reach sales.salesterritory.
      sales.salesorderheader already has a direct territoryid column — use that.
    - Never join sales.customer or person.person just to reach a territory or address
      when the order/transaction table has a direct FK.
    - If a dimension column (e.g. territoryid, storeid) exists directly on the fact table,
      join from there — do not route through intermediate entity tables.
    Wrong:  soh → salesperson → salesterritory   (excludes online orders with NULL salespersonid)
    Right:  soh → salesterritory via soh.territoryid

── Time handling ────────────────────────────────────────────────────────────
6. For time-series queries, always use EXTRACT to return date parts as separate integer columns
   (e.g. year, month) and ORDER BY them ASC. Never use TO_CHAR or combined date strings.
   Example: EXTRACT(YEAR FROM orderdate)::int AS year, EXTRACT(MONTH FROM orderdate)::int AS month
7. For relative time ranges ("last N months", "past N days", "previous N years"):
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
10. If the question contains a time filter ("last N months", "this year", etc.), you MUST include
    a WHERE clause that implements it. Never omit a time filter mentioned in the question.
12. When the user asks for period-over-period change, growth, decline, or comparison
    to a previous period — phrased with "growth", "change", "increase", "decrease",
    "compared to previous", "month over month", "year over year", "MoM", "YoY",
    "trend", "evolution" — you MUST compute the delta in the SQL itself using window
    functions. Do NOT return only the raw metric and leave the comparison to the caller.
    IMPORTANT: ONLY add window functions and period breakdowns when the user explicitly
    uses one of the keywords above. If the user asks for a simple breakdown by category
    (e.g. "by product category", "per region") with NO time or change keywords, return
    a flat aggregation — one row per category, no LAG, no year column, no date filter.
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
        ROUND(CAST(
            (value - LAG(value) OVER (ORDER BY <period_cols>))
            / NULLIF(LAG(value) OVER (ORDER BY <period_cols>), 0) * 100
        AS numeric), 2) AS pct_change
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
    - ROUND requires numeric type: always use CAST inside ROUND,
      e.g. ROUND(CAST(expr AS numeric), 2). Never pass double precision directly to ROUND,
      and never use ROUND((expr)::numeric, 2) — the ::numeric form is error-prone.
    - The first period row will have NULL for prev_value and pct_change — that is correct.

── Grouping & aggregation ───────────────────────────────────────────────────
8. When the user asks to group or break down by a dimension (e.g. "by category", "by region",
   "by product"), always include that column in both SELECT and GROUP BY — never drop it.
   When the user provides a list of specific IDs or named entities and asks for analysis,
   comparison, or distribution across them, always include that entity's identifier (or name)
   in both SELECT and GROUP BY so results are broken down per entity — never collapse them
   into a single aggregate. Example: "for these customers [IDs] analyse revenue" →
   include customerid in SELECT and GROUP BY.
9. When the user says "by category" without further specification, prefer the product
   classification hierarchy (e.g. productcategory.name, productsubcategory.name) over
   promotional or offer descriptions (e.g. specialoffer.description). Only use offer/promotion
   tables if the user explicitly asks for offers, discounts, or promotions.
11. LIMIT rule — apply exactly one of these cases:
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

── SQL syntax ───────────────────────────────────────────────────────────────
13. Column references in SELECT/WHERE/GROUP BY/ORDER BY must ALWAYS use the form
    alias.column — never schema.table.column. Only FROM and JOIN clauses use the
    fully-qualified schema.table form.
    CORRECT:   SELECT s.name, COUNT(e.businessentityid) FROM sales.store s JOIN humanresources.employee e ...
    INCORRECT: SELECT s.name, COUNT(hr.e.businessentityid) — this is invalid SQL.
    Also: NEVER use abbreviated schema aliases (hr.*, pe.*, sa.*, pr.*, pu.*) as table
    references in queries. Always use the full schema name (humanresources, person,
    sales, production, purchasing).
    Table aliases MUST be single words with no dots — "pc", "psc", "cat" are valid;
    "pr.pc", "prod.cat" are INVALID and will cause a parse error.
    Column references outside FROM/JOIN must have EXACTLY ONE dot: alias.column.
    Two-dot forms like alias.x.column or schema.table.column are NEVER valid in
    SELECT, WHERE, GROUP BY, or ON clauses — use the alias alone.

── Domain formulas ──────────────────────────────────────────────────────────
17. Date arithmetic — when computing a duration between two date/timestamp columns,
    ALWAYS emit BOTH of these columns (never just one):
        ROUND(CAST(AVG(EXTRACT(EPOCH FROM (col_end - col_start)) / 86400.0) AS numeric), 2) AS average_days,
        ROUND(CAST(AVG(EXTRACT(EPOCH FROM (col_end - col_start)) / 3600.0)  AS numeric), 1) AS average_hours
    average_days is the primary human-readable metric; average_hours gives sub-day precision
    so the reader can see e.g. "7.34 days (176.2 hours)".
    Do NOT use EXTRACT(EPOCH FROM ...) without dividing — epoch is seconds, not days or hours.
    Do NOT cast to ::date before subtracting — that discards time-of-day and forces integer days.
    Apply a validity filter where appropriate (e.g. shipdate IS NOT NULL, shipdate >= orderdate).
18. Cost and standard cost — always use p.standardcost directly from production.product.
    Do NOT join history or audit tables (e.g. productcosthistory, pricehistory) to retrieve
    the "current" cost — these tables have one row per change event and will cause silent
    row duplication or row loss when inner-joined to a fact table.
    Do NOT join the same dimension table twice under different aliases when one join suffices.
    Use the column from the single join you already have (e.g. p.standardcost, not pr.standardcost
    from a redundant second join to the same table).
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


# ── Database date range (cached) ─────────────────────────────────────────────
_date_range_cache: dict | None = None


def get_db_date_range() -> dict:
    """Query and cache the min/max dates from key business tables."""
    global _date_range_cache
    if _date_range_cache is not None:
        return _date_range_cache

    date_queries = [
        ("sales.salesorderheader", "orderdate"),
        ("purchasing.purchaseorderheader", "orderdate"),
        ("production.workorder", "startdate"),
        ("production.transactionhistory", "transactiondate"),
    ]

    tables = {}
    global_min = None
    global_max = None

    engine = get_engine()
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
    """Return a warning string if the query mentions a year outside the DB range, else None."""
    date_range = get_db_date_range()
    min_year = date_range.get("min_year")
    max_year = date_range.get("max_year")

    if not min_year or not max_year:
        return None

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


def _clean_sql(raw: str) -> str:
    """Strip markdown fences and leading 'sql' prefix from an LLM SQL response."""
    sql = raw.strip().strip("`").strip()
    if sql.lower().startswith("sql"):
        sql = sql[3:].strip()
    return sql


def _fix_round_casts(sql: str) -> str:
    """Ensure every ROUND(expr, n) call uses a ::numeric first argument.

    PostgreSQL has no ROUND(double precision, integer) overload. The LLM frequently
    generates expressions like ROUND(CAST(x AS DOUBLE PRECISION) / y * 100, 2) or
    ROUND(x / y, 2) where x/y produces double precision. Fix by replacing any
    CAST(... AS <float-type>) anywhere in the SQL with (...)::numeric, which coerces
    the whole expression to numeric before ROUND sees it.
    """
    # Replace CAST(expr AS DOUBLE PRECISION/FLOAT/REAL) → (expr)::numeric everywhere
    pattern = re.compile(
        r'CAST\s*\(\s*(.*?)\s*AS\s*(?:DOUBLE\s+PRECISION|FLOAT(?:4|8)?|REAL)\s*\)',
        re.IGNORECASE | re.DOTALL,
    )
    return pattern.sub(lambda m: f'({m.group(1)})::numeric', sql)


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
            if source_dialect is None:
                # SQL is already valid PostgreSQL — return the original with only
                # the ROUND/cast safety fix applied.  Transpiling PostgreSQL back
                # through sqlglot rewrites valid ::int / ::numeric casts into broken
                # "(expr AS INT)" forms that PostgreSQL rejects.
                return _fix_round_casts(sql)
            # For TSQL (SELECT TOP N, etc.) we do need sqlglot to transpile.
            transpiled = parsed.sql(
                dialect="postgres",
                unsupported_level=sqlglot.ErrorLevel.IGNORE,
            )
            return _fix_round_casts(transpiled)
        except sqlglot.errors.ParseError:
            continue
        except Exception:
            # Transpilation failed (e.g. TO_CHAR format unsupported) — original SQL is fine
            return _fix_round_casts(sql)
    # If all dialects fail, raise the error from the default parser
    parsed = sqlglot.parse_one(sql)
    transpiled = parsed.sql(dialect="postgres", unsupported_level=sqlglot.ErrorLevel.IGNORE)
    return _fix_round_casts(transpiled)


# ── Security blocklist (code-enforced, not LLM-dependent) ────────────────────
# These tables and columns must NEVER appear in generated SQL, regardless of
# what the user asks or how they phrase their prompt injection.

BLOCKED_TABLES = {
    "person.password",
}

BLOCKED_COLUMNS = {
    "passwordhash",
    "passwordsalt",
    "nationalidnumber",
}

# Destructive SQL keywords — the agent should only generate SELECT queries.
BLOCKED_KEYWORDS = {
    "DROP ", "DELETE ", "TRUNCATE ", "ALTER ", "INSERT ", "UPDATE ",
    "CREATE ", "GRANT ", "REVOKE ",
}


def _check_sql_security(sql: str) -> str | None:
    """Check SQL for blocked tables, columns, and destructive keywords.

    Returns an error message if blocked, or None if the SQL is safe.
    This is a hard code gate — the LLM cannot bypass it.
    """
    sql_upper = sql.upper()
    sql_lower = sql.lower()

    # Check destructive keywords
    for kw in BLOCKED_KEYWORDS:
        if kw in sql_upper:
            return (
                f"Query blocked: {kw.strip()} statements are not allowed. "
                f"This system only supports SELECT queries for data retrieval."
            )

    # Check blocked tables
    for table in BLOCKED_TABLES:
        if table.lower() in sql_lower:
            return (
                f"Query blocked: access to '{table}' is restricted. "
                f"This table contains sensitive data that cannot be queried."
            )

    # Check blocked columns
    for col in BLOCKED_COLUMNS:
        if col.lower() in sql_lower:
            return (
                f"Query blocked: the column '{col}' contains sensitive data "
                f"and cannot be included in queries. This is a security restriction."
            )

    return None


async def _retry_sql(
    session,
    system_prompt: str,
    error_info: str,
    search_query: str,
    schema_str: str,
    user_query: str,
) -> str:
    """Fetch semantic context and ask the LLM for a corrected SQL query.

    Returns the cleaned SQL string from the LLM response.
    """
    extra = semantic_search(search_query, top_k=3)
    extra_ctx = "\n".join(str(c) for c in extra) if extra else "None."
    retry_msg = SQL_RETRY_PROMPT.format(
        error_info=error_info,
        extra_context=extra_ctx,
        schema=schema_str,
        question=user_query,
    )
    response = invoke_with_retry("sql", [
        SystemMessage(content=system_prompt),
        HumanMessage(content=retry_msg),
    ])
    return _clean_sql(response.content)


async def _run_sql(state: AgentState) -> AgentState:
    """Async implementation: generate, validate, and execute SQL via MCP."""
    user_query = state["user_query"]
    plan = state.get("plan", "")
    retry_count = state.get("retry_count", 0)
    rag_context = state.get("rag_context", "")

    server_params = get_server_params()
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with _mcp_session(read_stream, write_stream) as session:
            await session.initialize()

            # Schema is read directly from disk and cached in-process
            # (avoids spawning an MCP subprocess on every query).
            schema_str = get_compact_schema()

            # Augment system prompt with any columns confirmed-nonexistent by the DB.
            banned_note = get_banned_columns_prompt()
            effective_system_prompt = (
                SQL_SYSTEM_PROMPT + "\n" + banned_note if banned_note else SQL_SYSTEM_PROMPT
            )

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
            rag_hint = f"\n\nKnowledge base context:\n{rag_context}" if rag_context else ""
            messages = [
                SystemMessage(content=effective_system_prompt),
                HumanMessage(
                    content=(
                        f"Schema:\n{schema_str}\n\n"
                        f"Plan: {plan}{date_hint}{rag_hint}\n\n"
                        f"Question: {user_query}"
                    )
                ),
            ]

            response = invoke_with_retry("sql", messages)
            raw_sql = _clean_sql(response.content)

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
                    raw_sql = await _retry_sql(
                        session, effective_system_prompt, error_info, raw_sql, schema_str, user_query
                    )
                    continue

                # Security check — block sensitive tables/columns/destructive SQL
                security_error = _check_sql_security(validated_sql)
                if security_error:
                    return {
                        "sql_query": validated_sql,
                        "sql_result": [],
                        "error": security_error,
                        "retry_count": retry_count,
                        "schema_context": schema_str,
                    }

                # Execute the validated SQL
                try:
                    result = await call_tool(
                        session, "run_sql_query", {"sql": validated_sql}
                    )
                except Exception as e:
                    error_info = f"SQL execution error: {e}"
                    # Record banned column from exception text too
                    if "does not exist" in str(e).lower():
                        banned_key = record_banned_column(str(e), validated_sql)
                        if banned_key:
                            invalidate_compact_schema_cache()
                            banned_note = get_banned_columns_prompt()
                            effective_system_prompt = (
                                SQL_SYSTEM_PROMPT + "\n" + banned_note
                                if banned_note else SQL_SYSTEM_PROMPT
                            )
                    retry_count += 1
                    if retry_count >= 3:
                        return {
                            "sql_query": validated_sql,
                            "sql_result": [],
                            "error": error_info,
                            "retry_count": retry_count,
                            "schema_context": schema_str,
                        }
                    raw_sql = await _retry_sql(
                        session, effective_system_prompt, error_info, validated_sql, schema_str, user_query
                    )
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
                        raw_error = result['error']
                        error_info = f"Database error: {raw_error}"
                        # Persist confirmed-nonexistent columns so future sessions
                        # don't repeat the same hallucination
                        if "does not exist" in raw_error.lower():
                            banned_key = record_banned_column(raw_error, validated_sql)
                            if banned_key:
                                invalidate_compact_schema_cache()
                                # Rebuild effective prompt with new banned entry
                                banned_note = get_banned_columns_prompt()
                                effective_system_prompt = (
                                    SQL_SYSTEM_PROMPT + "\n" + banned_note
                                    if banned_note else SQL_SYSTEM_PROMPT
                                )
                    else:
                        raw_str = repr(result)
                        error_info = f"Unexpected result from database: {raw_str[:200]}"
                        if "does not exist" in raw_str.lower():
                            banned_key = record_banned_column(raw_str, validated_sql)
                            if banned_key:
                                invalidate_compact_schema_cache()
                                banned_note = get_banned_columns_prompt()
                                effective_system_prompt = (
                                    SQL_SYSTEM_PROMPT + "\n" + banned_note
                                    if banned_note else SQL_SYSTEM_PROMPT
                                )
                    retry_count += 1
                    if retry_count >= 3:
                        return {
                            "sql_query": validated_sql,
                            "sql_result": [],
                            "error": error_info,
                            "retry_count": retry_count,
                            "schema_context": schema_str,
                        }
                    raw_sql = await _retry_sql(
                        session, effective_system_prompt, error_info, validated_sql, schema_str, user_query
                    )
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
                    raw_sql = await _retry_sql(
                        session, effective_system_prompt, "Query returned empty results.",
                        user_query, schema_str, user_query,
                    )
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
