"""Streamlit chat UI for the multi-agent BI assistant.

Provides a conversational interface that sends user queries through the
LangGraph pipeline and displays results including:

- Natural-language answer from the Response Agent.
- Raw query results table (direct database output — no LLM involved).
- The executed SQL query (collapsible expander).
- A generated chart image when the chart agent ran.

Showing the raw data table separately from the LLM answer lets users
verify every number the assistant states against the actual database output.
"""

from __future__ import annotations

import base64
import os
import sys

import sqlglot
import streamlit as st

# Ensure the project root is on the Python path so the graph and agents can
# be imported regardless of where Streamlit is launched from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph import compiled_graph  # noqa: E402  (must follow sys.path insert)


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BI Assistant",
    page_icon="📊",
    layout="wide",
)

st.title("Multi-Agent BI Assistant")
st.caption("Ask questions about your data or documents — powered by LLaMA 4 Scout")


# ── UI helpers ─────────────────────────────────────────────────────────────────

def _render_assistant_turn(
    final_answer: str,
    sql_query: str,
    sql_result: list[dict],
    chart_b64: str | None,
    chart_title: str | None,
) -> None:
    """Render all components of one assistant response inside the current chat bubble.

    Render order (top → bottom):
    1. Natural-language answer from the LLM.
    2. Raw query results table (direct DB output — expandable, interactive).
    3. SQL query (collapsed expander).
    4. Chart image (when a chart was generated).

    The raw data table is shown separately from the LLM answer so users can
    instantly verify every number the assistant states against real DB values.

    Args:
        final_answer: Markdown-formatted LLM answer.
        sql_query:    Raw SQL string; empty string when no query was run.
        sql_result:   List of row-dicts from SQL execution; empty list when none.
        chart_b64:    Base64-encoded PNG, or ``None`` when no chart was generated.
        chart_title:  Caption string for the chart, or ``None``.
    """
    # 1. LLM answer
    st.markdown(final_answer)

    # 2. Raw data table — shown only when the DB returned rows
    if sql_result:
        with st.expander(
            f"Raw Query Results — {len(sql_result)} rows:",
            expanded=False,
        ):
            st.dataframe(sql_result, use_container_width=True)

    # 3. SQL query — collapsed by default to keep the chat clean
    if sql_query:
        try:
            formatted_sql = sqlglot.transpile(sql_query, pretty=True)[0]
        except Exception:
            formatted_sql = sql_query
        with st.expander("SQL Query"):
            st.code(formatted_sql, language="sql")

    # 4. Chart image
    if chart_b64:
        chart_bytes = base64.b64decode(chart_b64)
        st.image(chart_bytes, caption=chart_title or "Chart")


# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay previous turns so the full conversation is visible after page refreshes.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            _render_assistant_turn(
                final_answer=msg["content"],
                sql_query=msg.get("sql_query", ""),
                sql_result=msg.get("sql_result", []),
                chart_b64=msg.get("chart_b64"),
                chart_title=msg.get("chart_title"),
            )
        else:
            st.markdown(msg["content"])


# ── Chat input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a question about your business data...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = compiled_graph.invoke({"user_query": user_input})

                final_answer: str = result.get(
                    "final_answer", "I wasn't able to generate an answer."
                )
                sql_query: str = result.get("sql_query", "")
                sql_result: list[dict] = result.get("sql_result", [])
                chart_spec: dict = result.get("chart_spec", {})

                chart_b64: str | None = None
                chart_title: str | None = None
                if chart_spec and chart_spec.get("image_base64"):
                    chart_b64 = chart_spec["image_base64"]
                    chart_title = chart_spec.get("title", "Chart")

                _render_assistant_turn(
                    final_answer=final_answer,
                    sql_query=sql_query,
                    sql_result=sql_result,
                    chart_b64=chart_b64,
                    chart_title=chart_title,
                )

                # Store raw data so the replay loop re-renders the table on reload
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_answer,
                    "sql_query": sql_query,
                    "sql_result": sql_result,
                    "chart_b64": chart_b64,
                    "chart_title": chart_title,
                })

            except Exception as e:
                # Unwrap nested exceptions (TaskGroup, ExceptionGroup) to surface the real error
                root_cause = e
                while hasattr(root_cause, 'exceptions') and root_cause.exceptions:
                    root_cause = root_cause.exceptions[0]
                while root_cause.__cause__:
                    root_cause = root_cause.__cause__

                error_type = type(root_cause).__name__
                error_detail = str(root_cause)

                # Provide user-friendly messages for common errors
                if "rate_limit" in error_detail.lower() or "429" in error_detail:
                    error_msg = (
                        "**Rate limit reached on the Groq API.** "
                        "You've exceeded the free-tier request limit. "
                        "Wait a minute and try again, or upgrade your Groq plan."
                    )
                elif "authentication" in error_detail.lower() or "401" in error_detail or "invalid api key" in error_detail.lower():
                    error_msg = (
                        "**Groq API authentication failed.** "
                        "Check that your `GROQ_API_KEY` in `.env` is correct and active."
                    )
                elif "quota" in error_detail.lower() or "insufficient" in error_detail.lower():
                    error_msg = (
                        "**Groq API quota exceeded.** "
                        "You've used all available tokens/requests. "
                        "Wait or upgrade your Groq plan."
                    )
                elif "connection" in error_detail.lower() or "timeout" in error_detail.lower():
                    error_msg = (
                        "**Connection error.** "
                        "Could not reach the Groq API or the database. "
                        "Check your internet connection and that PostgreSQL is running."
                    )
                else:
                    error_msg = f"**Error ({error_type}):** {error_detail}"

                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })
