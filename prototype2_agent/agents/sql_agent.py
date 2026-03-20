"""SQL specialist agent.

Generates and executes SQL queries against PostgreSQL via MCP tools.
Uses sqlglot for AST-based SQL validation and has a retry loop that falls
back to semantic_search for extra context on failure.
"""

import asyncio
import json
import os

import sqlglot
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from state import AgentState
from mcp_client import get_server_params, call_tool

load_dotenv()

# ── System prompt ─────────────────────────────────────────────────────────────
SQL_SYSTEM_PROMPT = """\
You are a SQL specialist agent for a PostgreSQL database.

You will be given:
- A database schema snapshot (table names, column names, data types).
- The user's question.
- Optionally, extra documentation context from a knowledge base.

Your job:
1. Write a single, correct PostgreSQL SQL query that answers the user's question.
2. Use ONLY tables and columns that appear in the schema snapshot.
3. Return ONLY the raw SQL query — no markdown fences, no explanation.
"""

SQL_RETRY_PROMPT = """\
The previous SQL query failed or returned no results.

Error/result: {error_info}

Here is additional context from the knowledge base that may help:
{extra_context}

Schema:
{schema}

Original question: {question}

Write a corrected PostgreSQL SQL query. Return ONLY the raw SQL.
"""


def _validate_sql(sql: str) -> str:
    """Parse and transpile SQL to PostgreSQL dialect using sqlglot.

    Args:
        sql: Raw SQL string.

    Returns:
        Transpiled SQL string in Postgres dialect.

    Raises:
        sqlglot.errors.ParseError: If the SQL is syntactically invalid.
    """
    parsed = sqlglot.parse_one(sql)
    return parsed.sql(dialect="postgres")


async def _run_sql(state: AgentState) -> AgentState:
    """Async implementation: generate, validate, and execute SQL via MCP."""
    user_query = state["user_query"]
    plan = state.get("plan", "")
    retry_count = state.get("retry_count", 0)

    server_params = get_server_params()
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with _mcp_session(read_stream, write_stream) as session:
            await session.initialize()

            # Fetch schema snapshot via MCP
            schema = await call_tool(session, "get_schema_snapshot", {})
            schema_str = json.dumps(schema, indent=2) if schema else "No schema available."

            llm = ChatGroq(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0,
            )

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

            response = llm.invoke(messages)
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
                    response = llm.invoke(messages)
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
                    response = llm.invoke(messages)
                    raw_sql = response.content.strip().strip("`").strip()
                    if raw_sql.lower().startswith("sql"):
                        raw_sql = raw_sql[3:].strip()
                    continue

                # Check for empty result
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
                    response = llm.invoke(messages)
                    raw_sql = response.content.strip().strip("`").strip()
                    if raw_sql.lower().startswith("sql"):
                        raw_sql = raw_sql[3:].strip()
                    continue

                # Success
                return {
                    "sql_query": validated_sql,
                    "sql_result": result if isinstance(result, list) else [],
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
    return asyncio.run(_run_sql(state))
