"""Response formulation agent.

Synthesizes a final answer from all populated state fields.
Always runs last and is the sole writer of ``final_answer``.

Anti-hallucination design
--------------------------
Python computes all numbers; the LLM only interprets.

- Small results (≤ threshold rows): all rows shown inline; LLM adds business
  context without seeing the figures.
- Large results (> threshold rows): Python extracts key facts (highest, lowest,
  average, top-3 share, total); LLM is given those pre-computed facts verbatim
  and asked to frame an insight — it cannot hallucinate numbers it wasn't given.
- No numeric facts available: LLM call is skipped entirely to prevent invention.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState
from llm_config import invoke_with_retry


_INSIGHT_THRESHOLD = 12  # rows above this get key-facts summary instead of full list


def _fmt_value(v) -> str:
    """Format a cell value: round floats to 2 dp with thousands separator."""
    try:
        f = float(v)
        if f == int(f):
            i = int(f)
            return f"{i:,}" if abs(i) >= 10_000 else str(i)
        return f"{f:,.2f}"
    except (TypeError, ValueError):
        return str(v) if v is not None else ""


# Checked first — unambiguous metrics, never dimensions (catches "prev_revenue", "total_sales" etc.)
_METRIC_KEYWORDS = ("revenue", "sales", "amount", "profit", "cost", "price", "spend", "total", "sum")
# Time/period labels — treat as dimensions not metrics
_DIM_KEYWORDS = ("year", "date", "month", "quarter", "week", "period", "day")
# Rate/change columns — averaging or summing them is meaningless
_RATE_KEYWORDS = ("pct", "percent", "growth", "change", "rate", "ratio", "diff", "delta")
# LAG/LEAD intermediates — used only for calculation, excluded from summaries entirely
_INTERMEDIATE_PREFIXES = ("prev_", "lag_", "lead_", "next_")


def _is_dimension(col: str, nums: list[float]) -> bool:
    """True when a numeric column should be treated as a label, not a metric.

    Catches: time columns (year, month…), ID/key/code columns (customerid,
    order_id, productkey…), and year-range integers (2020, 2021…).
    Explicit metric name patterns (revenue, sales…) are always False.
    """
    col_lower = col.lower()
    # Explicit metric name → always numeric, regardless of other checks
    if any(kw in col_lower for kw in _METRIC_KEYWORDS):
        return False
    if any(kw in col_lower for kw in _DIM_KEYWORDS):
        return True
    # Integer columns whose name ends with an ID/key suffix are identifiers
    is_all_int = bool(nums) and all(v == int(v) for v in nums)
    if is_all_int and (
        col_lower.endswith("id") or col_lower.endswith("key") or col_lower.endswith("code")
    ):
        return True
    # Year-like integers (e.g. column named "fiscal_year" already caught above,
    # but also catch unnamed year columns whose values are all in 1900-2100)
    return is_all_int and all(1900 <= int(v) <= 2100 for v in nums)


def _classify_columns(rows: list[dict]) -> tuple[list[str], list[str]]:
    """Return (label_cols, numeric_cols) for a result set.

    Columns matching _INTERMEDIATE_PREFIXES (prev_, lag_, lead_, next_) are
    skipped entirely — they are LAG/LEAD intermediates used only for calculation
    and produce meaningless facts (e.g. "Highest prev_revenue: …").
    """
    label_cols, numeric_cols = [], []
    for col in (rows[0] if rows else {}):
        col_lower = col.lower()
        if any(col_lower.startswith(p) for p in _INTERMEDIATE_PREFIXES):
            continue  # exclude intermediate calculation columns
        sample = [r[col] for r in rows[:20] if r.get(col) is not None]
        nums: list[float] = []
        for v in sample:
            try:
                nums.append(float(v))
            except (TypeError, ValueError):
                pass
        if sample and len(nums) / len(sample) >= 0.7 and not _is_dimension(col, nums):
            numeric_cols.append(col)
        else:
            label_cols.append(col)
    return label_cols, numeric_cols


_RANKING_WORDS = (
    "most", "top", "highest", "largest", "best", "biggest",
    "worst", "lowest", "least", "fewest", "smallest",
)


def _is_ranking_query(user_query: str) -> bool:
    q = user_query.lower()
    return any(w in q for w in _RANKING_WORDS)


def _extract_key_facts(rows: list[dict], ranked: bool = False) -> list[tuple[str, str]]:
    """Return (label, value_string) fact pairs derived purely from the data — no LLM.

    When ``ranked=True`` (ranking query), skip Lowest/Average/Total stats that
    are irrelevant when the result set is already a pre-sorted top-N list.
    """
    if not rows:
        return []

    n = len(rows)
    label_cols, numeric_cols = _classify_columns(rows)
    dim_cols = label_cols or [next(iter(rows[0]))]

    def _row_label(r: dict) -> str:
        return " / ".join(str(r[c]) for c in dim_cols if r.get(c) is not None) or "?"

    facts: list[tuple[str, str]] = []

    for nc in numeric_cols[:3]:
        vals: list[tuple[str, float]] = []
        for r in rows:
            try:
                vals.append((_row_label(r), float(r.get(nc))))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                pass
        if not vals:
            continue

        nums = [v for _, v in vals]
        total = sum(nums)
        sorted_desc = sorted(vals, key=lambda x: x[1], reverse=True)
        top_label, top_val = sorted_desc[0]
        bot_label, bot_val = sorted_desc[-1]

        is_rate = any(kw in nc.lower() for kw in _RATE_KEYWORDS)

        facts.append((f"Highest {nc}", f"{top_label} — {_fmt_value(top_val)}"))
        if not ranked:
            if n > 2:
                facts.append((f"Lowest {nc}", f"{bot_label} — {_fmt_value(bot_val)}"))
            # Average growth rate is meaningful; average of a raw total is too
            if n > 1:
                facts.append((f"Average {nc}", _fmt_value(total / len(nums))))
        # Top-3 share and Total are meaningless for rates/percentages
        if not is_rate and total > 0 and n > 3:
            seen_lbls: dict[str, float] = {}
            for lbl, val in sorted_desc:
                seen_lbls.setdefault(lbl, val)
            top3 = list(seen_lbls.items())[:3]
            pct = sum(v for _, v in top3) / total * 100
            facts.append((f"Top 3 by {nc}", f"{', '.join(l for l, _ in top3)} — {_fmt_value(pct)}% of total"))
        if not is_rate and not ranked and n > 1:
            facts.append((f"Total {nc}", _fmt_value(total)))

    return facts


def _format_key_facts(facts: list[tuple[str, str]]) -> str:
    """Render (label, value) pairs in the same style as 2-col row output."""
    return "\n\n".join(f"{label}: `{value}`" for label, value in facts)


def _chart_note(chart_spec: dict) -> str:
    """Return a one-line chart mention for the LLM context, or '' if no chart."""
    opt = (chart_spec.get("options") or [{}])[0]
    if not opt:
        return ""
    return (
        f"\n\nA {opt.get('chart_type', 'chart')} chart titled "
        f'"{opt.get("title", "Chart")}" was generated for this data.'
    )


def _format_rows(rows: list[dict]) -> str:
    """Format query rows for display in the final answer.

    - 1 row, 1 col   → plain scalar value
    - 1 row, 2+ cols → "Col: value" list (transposed, no label ambiguity)
    - 2 col           → "Label: `value`" per row
    - 3+ cols         → card-per-row: bold label heading + metric items on next line
    """
    if not rows:
        return "(no rows)"
    cols = list(rows[0].keys())

    # ── Single scalar ──────────────────────────────────────────────────────────
    if len(rows) == 1 and len(cols) == 1:
        return _fmt_value(next(iter(rows[0].values())))

    # ── Single row, multiple columns → transposed key-value list ──────────────
    if len(rows) == 1:
        return "\n\n".join(
            f"{c}: `{_fmt_value(rows[0].get(c))}`" for c in cols
        )

    # ── Two columns: label → metric ────────────────────────────────────────────
    if len(cols) == 2:
        label_col, value_col = cols[0], cols[1]
        return "\n\n".join(
            f"{r.get(label_col)}: `{_fmt_value(r.get(value_col))}`"
            for r in rows
        )

    # ── 3+ columns: bulleted row per entry, all columns styled equally ────────
    blocks = []
    for r in rows:
        items = "  ·  ".join(f"{c}: `{_fmt_value(r.get(c))}`" for c in cols)
        blocks.append(items)
    return "\n\n".join(blocks)


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

GROUNDED_INSIGHT_PROMPT = """\
You are a Business Intelligence assistant.

You have been given EXACT numerical facts computed directly from the database.
Write 1-3 plain-prose sentences of business insight that directly answer the
user's question.  Start immediately with the insight — no preamble.

STRICT RULES:
- Every number you write MUST appear verbatim in the "Computed Facts" provided.
- Do NOT round, approximate, or derive any figure not already in the facts.
- Do NOT restate all the facts — pick the ones most relevant to the question.
- Do NOT use backtick, code, or any special formatting for numbers.
- Do NOT start with phrases like "Here are...", "Based on...", or "The data shows...".
- Do NOT include raw SQL or technical column names.
"""

RAG_PROMPT = """\
You are a Business Intelligence assistant.
Answer the user's question using ONLY the provided document context.
Do NOT invent numbers or facts not present in the context.
If the context is insufficient, say so clearly and suggest what data source
might have the answer.
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
    user_query: str = state.get("user_query", "")
    sql_query: str = state.get("sql_query", "")
    sql_result: list[dict] = state.get("sql_result", [])
    rag_context: str = state.get("rag_context", "")
    chart_spec: dict = state.get("chart_spec", {})
    error: str = state.get("error", "")

    # ── Path A: SQL results present ────────────────────────────────────────────
    # Numbers always come from Python.  LLM role differs by result size:
    #   ≤ threshold  → LLM adds context only (no numbers given to it)
    #   > threshold  → Python extracts key facts with exact numbers;
    #                  LLM frames those facts verbatim (cannot hallucinate)
    if sql_result:
        n_rows = len(sql_result)
        col_names = list(sql_result[0].keys())
        large_result = n_rows > _INSIGHT_THRESHOLD

        if large_result:
            # Part 1 — programmatic key facts (Python only, no LLM).
            ranked = _is_ranking_query(user_query)
            key_facts = _extract_key_facts(sql_result, ranked=ranked)
            facts_block = (
                f"Summary ({n_rows} rows — full dataset in Raw Data block below):"
                + (f"\n\n{_format_key_facts(key_facts)}" if key_facts else "")
            )

            # Part 2 — LLM insight, but ONLY when there are computed facts to
            # ground it on.  With no numeric facts the LLM has nothing to cite
            # and will hallucinate — so skip it entirely in that case.
            if key_facts:
                plain_facts = "\n".join(f"{lbl}: {val}" for lbl, val in key_facts)
                grounded_ctx = (
                    f"Question: {user_query}\n\n"
                    f"Computed Facts (use ONLY these numbers):\n{plain_facts}"
                    + (f"\n\nAdditional business context:\n{rag_context}" if rag_context else "")
                    + _chart_note(chart_spec)
                )
                insight = invoke_with_retry("response", [
                    SystemMessage(content=GROUNDED_INSIGHT_PROMPT),
                    HumanMessage(content=grounded_ctx),
                ]).content.strip()
            else:
                insight = None

        else:
            # Part 1 — show all rows formatted (≤ threshold, fits inline)
            facts_block = (
                f"Query returned {n_rows} row(s):\n\n"
                + _format_rows(sql_result)
            )

            # Part 2 — LLM adds business context only; receives no raw numbers
            interpretation_ctx = (
                f"Question: {user_query}\n\n"
                f"The query returned {n_rows} row(s).\n"
                f"Columns: {', '.join(col_names)}."
            )
            if rag_context:
                interpretation_ctx += f"\n\nAdditional business context:\n{rag_context}"
            interpretation_ctx += _chart_note(chart_spec)
            insight = invoke_with_retry("response", [
                SystemMessage(content=INTERPRETATION_PROMPT),
                HumanMessage(content=interpretation_ctx),
            ]).content.strip()

        final = f"{facts_block}\n\n**Key Insights:** {insight}" if insight else facts_block
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
    # The reranker in rag_agent already filtered irrelevant chunks.
    # LLM synthesizes an answer strictly from the filtered content.
    # Raw chunks are passed to the UI separately (shown in an expander).
    rag_chunks: list[dict] = state.get("rag_chunks", [])
    if rag_chunks:
        # Build context from filtered chunks for the LLM
        chunk_text = "\n\n---\n\n".join(c["content"] for c in rag_chunks)

        response = invoke_with_retry("response", [
            SystemMessage(content=RAG_PROMPT),
            HumanMessage(
                content=f"Question: {user_query}\n\nContext:\n{chunk_text}"
            ),
        ])
        return {"final_answer": response.content.strip()}

    # Legacy fallback: rag_context without structured chunks
    if rag_context:
        return {"final_answer": rag_context}

    # ── Path E: nothing available ──────────────────────────────────────────────
    return {
        "final_answer": (
            "I was unable to retrieve data to answer this question.\n\n"
            "Please check that the database is connected and the MCP server "
            "is running, then try again."
        )
    }
