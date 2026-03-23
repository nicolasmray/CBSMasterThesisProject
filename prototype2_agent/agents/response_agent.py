"""Response formulation agent.

Synthesizes a final answer from all populated state fields.
Always runs last on every graph path and is the sole writer of ``final_answer``.
Does NOT call any MCP tools.

Anti-hallucination design:
--------------------------
The previous version asked the LLM to "summarize the key findings with numbers",
giving it free rein to invent, round, or misquote figures.  Even with strict
system-prompt rules, LLaMA models routinely ignore number-grounding constraints.

This version removes the problem structurally by splitting every answer into two
separate, independent parts:

**Part 1 — programmatic data block (Python only, no LLM)**
    :func:`~utils.stats.build_data_summary` formats row count, per-column
    statistics, and sample rows as plain text.  This block is written directly
    into ``final_answer``; the LLM never touches it.

**Part 2 — LLM interpretation (business insight only)**
    The model is given the user question and the column names but NOT the
    numbers.  It is asked for 2-4 sentences of business context (trends,
    anomalies, recommendations).  Without access to the figures it cannot
    hallucinate them.

This mirrors the design of ``thesisProject_langChain``, where
``AnalysisOrchestrator`` computes statistics first and the LLM only interprets.

Empty-result handling
---------------------
When ``sql_result`` is empty but a ``sql_query`` was generated, the agent
reports what happened (query ran but returned no rows, or execution failed)
instead of returning a useless "context is insufficient" message.
"""

<<<<<<< HEAD
from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
=======
from langchain_core.messages import SystemMessage, HumanMessage
>>>>>>> origin/main

from state import AgentState
from llm_config import get_llm


def _fmt_value(v) -> str:
    """Format a cell value: round floats to 2 dp with thousands separator."""
    try:
        f = float(v)
        if f == int(f):
            return f"{int(f):,}"
        return f"{f:,.2f}"
    except (TypeError, ValueError):
        return str(v) if v is not None else ""

<<<<<<< HEAD


def _format_rows(rows: list[dict]) -> str:
    """Format query rows for display in the final answer.

    - 1 row, 1 col  → plain scalar value
    - 2+ cols        → "Label: $value" per line (first col = label, second = metric)
    """
    if not rows:
        return "(no rows)"
    cols = list(rows[0].keys())

    # Single scalar
    if len(rows) == 1 and len(cols) == 1:
        return _fmt_value(next(iter(rows[0].values())))

    # Two-column: label → metric list
    if len(cols) == 2:
        label_col, value_col = cols[0], cols[1]
        return "\n\n".join(
            f"{r.get(label_col)}: `{_fmt_value(r.get(value_col))}`"
            for r in rows
        )
    
# ── Prompts ────────────────────────────────────────────────────────────────────

INTERPRETATION_PROMPT = """\
You are a Business Intelligence assistant providing executive-level insight.

The user has already been shown the exact query results and statistics from the
database.  Your job is to add business VALUE — NOT to restate numbers.

Write 2-4 sentences that:
- Identify the most notable trend, pattern, or outlier in the results.
- Explain what it means for the business in plain language.
- Optionally suggest a useful follow-up question or action.

STRICT RULES:
- Do NOT repeat or restate any specific numbers.  They are already shown above.
- Do NOT say things like "the total is X" or "there are Y customers".
- If the result is a single aggregate value (a count, a sum, etc.) with no
  meaningful trend to discuss, just confirm what was measured in one sentence.
- Do NOT include raw SQL or technical implementation details.
"""

RAG_PROMPT = """\
You are a Business Intelligence assistant.
Answer the user's question using ONLY the provided document context.
Do NOT invent numbers or facts not present in the context.
If the context is insufficient, say so clearly and suggest what data source
might have the answer.
=======
Rules:
- NEVER invent, fabricate, or hallucinate data. Use ONLY the exact values from the provided query results.
- If query results are empty or missing, say "No results were returned" — do NOT make up data.
- Do NOT include raw SQL, technical jargon, or implementation details unless the user asked.
- If query results are provided, summarize the key findings using ONLY the actual data and numbers present in the results.
- If a chart was generated, acknowledge it and describe the key business insight it shows.
- If an error occurred, explain the problem in plain language and suggest what the user could try.
- If both document context (RAG) and data results are available, weave them together.
- Keep the response concise but complete — aim for 2-5 sentences for simple queries,
  more for complex ones.
- Use bullet points or numbered lists when presenting multiple data points.
>>>>>>> origin/main
"""


# ── Agent node ─────────────────────────────────────────────────────────────────

def response_agent(state: AgentState) -> AgentState:
    """Synthesize pipeline state into a final answer with guaranteed accurate numbers.

    Response paths
    --------------
    The agent selects a path based on what the pipeline produced:

    **Path A — SQL results present** (most common):
        1. Build a programmatic fact block in Python via
           :func:`~utils.stats.build_data_summary` — row count, per-column
           stats, sample rows.  No LLM involved for this part.
        2. Call the LLM for a short business interpretation paragraph.  The
           model receives only column names and row count, NOT the numbers,
           so it cannot hallucinate figures.
        3. Concatenate: ``fact_block + "---" + llm_interpretation``.

    **Path B — SQL ran but returned no rows**:
        Report that the query returned no results and show the SQL so the user
        can diagnose the issue (wrong filter, empty table, schema mismatch).

    **Path C — Execution error**:
        Show the error and the SQL that was attempted in plain language.

    **Path D — RAG-only** (intent was "rag", no SQL ran):
        Call the LLM with the retrieved document context.

    **Path E — Nothing** (should not normally occur):
        Return a clear "no data available" message.

    Args:
        state: Current LangGraph pipeline state.  Keys read:

            - ``user_query``   — original natural-language question.
            - ``sql_query``    — generated SQL string (may be empty).
            - ``sql_result``   — list of row-dicts from execution (may be empty).
            - ``rag_context``  — retrieved document passages (may be empty).
            - ``chart_spec``   — chart metadata dict (may be empty).
            - ``plan``         — orchestrator routing plan (informational).
            - ``error``        — non-empty string when a pipeline error occurred.

    Returns:
        Partial ``AgentState`` update containing only ``final_answer``.
    """
<<<<<<< HEAD
    user_query: str = state.get("user_query", "")
    sql_query: str = state.get("sql_query", "")
    sql_result: list[dict] = state.get("sql_result", [])
    rag_context: str = state.get("rag_context", "")
    chart_spec: dict = state.get("chart_spec", {})
    error: str = state.get("error", "")

    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=os.getenv("GROQ_API_KEY"),
        # Low temperature keeps interpretation focused and non-inventive
        temperature=0.1,
    )
=======
    user_query = state.get("user_query", "")
    sql_result = state.get("sql_result", [])
    rag_context = state.get("rag_context", "")
    chart_spec = state.get("chart_spec", {})
    plan = state.get("plan", "")
    error = state.get("error", "")
    sql_query = state.get("sql_query", "")
    intent = state.get("intent", "")

    # HARD GUARD: if there's an error and no data, return a fixed message.
    # Never let the LLM generate content when there's no real data — it will hallucinate.
    if error and not sql_result and not rag_context:
        answer = (
            f"I wasn't able to retrieve the data for your question.\n\n"
            f"**Error:** {error}\n\n"
            f"You could try rephrasing your question or being more specific "
            f"about which tables or fields you're interested in."
        )
        if sql_query and intent in ("sql", "chart", "hybrid"):
            answer += f"\n\n**Attempted SQL:**\n```sql\n{sql_query}\n```"
        return {"final_answer": answer}

    # Build the context message for the LLM
    import json

    parts = [f"User question: {user_query}"]

    if plan:
        parts.append(f"Analysis plan: {plan}")

    if sql_result:
        # Limit displayed results to avoid token overflow
        display = sql_result[:20]
        parts.append(f"Query results ({len(sql_result)} rows, showing first {len(display)}):\n{json.dumps(display, indent=2, default=str)}")

    if rag_context:
        parts.append(f"Document context:\n{rag_context}")

    if chart_spec and chart_spec.get("image_base64"):
        parts.append(
            f"A {chart_spec.get('chart_type', 'chart')} chart titled "
            f"\"{chart_spec.get('title', 'Chart')}\" has been generated "
            f"(x={chart_spec.get('x', '?')}, y={chart_spec.get('y', '?')})."
        )

    # llm = ChatGroq(
    llm = get_llm("response")
>>>>>>> origin/main

    # ── Path A: SQL results present ────────────────────────────────────────────
    # This is the normal success path for any SQL query (COUNT, aggregate, or
    # multi-row).  Numbers come entirely from Python; LLM only adds context.
    if sql_result:
        # Part 1 — raw results serialised directly, no LLM involved
        facts_block = (
            f"Query returned {len(sql_result)} row(s):\n\n"
            + _format_rows(sql_result[:20])
        )

<<<<<<< HEAD
        # Part 2 — LLM interpretation
        # Deliberately give the LLM column names and row count but NOT the actual
        # values, so it can describe patterns without being tempted to restate figures.
        col_names = list(sql_result[0].keys())
        interpretation_ctx = (
            f"Question: {user_query}\n\n"
            f"The query returned {len(sql_result)} row(s).\n"
            f"Columns: {', '.join(col_names)}."
        )
        if rag_context:
            interpretation_ctx += f"\n\nAdditional business context:\n{rag_context}"
        if chart_spec and chart_spec.get("image_base64"):
            interpretation_ctx += (
                f"\n\nA {chart_spec.get('chart_type', 'chart')} chart titled "
                f'"{chart_spec.get("title", "Chart")}" was generated for this data.'
            )

        insight = llm.invoke([
            SystemMessage(content=INTERPRETATION_PROMPT),
            HumanMessage(content=interpretation_ctx),
        ]).content.strip()

        final = f"{facts_block}\n\n**Key Insights:** {insight}"
        return {"final_answer": final}

    # ── Path B: SQL ran but returned no rows ──────────────────────────────────
    # Report clearly instead of saying "context is insufficient".
    if sql_query and not error:
        msg = (
            f"The query executed successfully but returned **no results**.\n\n"
            f"This usually means the table is empty, the filters match nothing, "
            f"or there is a schema mismatch.\n\n"
            f"Query that was run:\n```sql\n{sql_query}\n```"
        )
        return {"final_answer": msg}

    # ── Path C: execution error ────────────────────────────────────────────────
    if error:
        sql_note = f"\n\nQuery attempted:\n```sql\n{sql_query}\n```" if sql_query else ""
        msg = (
            f"The query could not be completed.\n\n"
            f"**Error:** {error}"
            f"{sql_note}"
        )
        return {"final_answer": msg}

    # ── Path D: RAG-only (no SQL result, but documents were retrieved) ─────────
    if rag_context:
        response = llm.invoke([
            SystemMessage(content=RAG_PROMPT),
            HumanMessage(
                content=f"Question: {user_query}\n\nContext:\n{rag_context}"
            ),
        ])
        return {"final_answer": response.content.strip()}

    # ── Path E: nothing available ──────────────────────────────────────────────
    return {
        "final_answer": (
            "I was unable to retrieve data to answer this question.\n\n"
            "Please check that the database is connected and the MCP server "
            "is running, then try again."
        )
    }
=======
    response = llm.invoke(messages)
    answer = response.content.strip()

    # Append the exact executed SQL (from state, not LLM-generated) for sql/chart/hybrid intents
    if sql_query and intent in ("sql", "chart", "hybrid"):
        answer += f"\n\n**Executed SQL:**\n```sql\n{sql_query}\n```"

    return {"final_answer": answer}
>>>>>>> origin/main
