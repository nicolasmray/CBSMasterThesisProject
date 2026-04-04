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

import re
from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState
from llm_config import invoke_with_retry
from db.schema_snapshot import column_exists_anywhere


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
# Ranked list formatting helpers
_ID_SUFFIXES = ("id", "key", "code", "number", "num")


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


def _is_id_col(col: str) -> bool:
    cl = col.lower().replace("_", "")
    return any(cl.endswith(s) for s in _ID_SUFFIXES)


def _format_ranked_list(rows: list[dict]) -> str:
    """Numbered list for ranking results: `rank. Label — $value`.

    Hides ID columns, merges firstname+lastname, prefixes $ on currency metrics.
    """
    if not rows:
        return "(no rows)"

    label_cols, numeric_cols = _classify_columns(rows)
    # Drop ID columns — they're in the raw table below
    label_cols = [c for c in label_cols if not _is_id_col(c)]

    def _row_label(r: dict) -> str:
        # Merge firstname + lastname if both present
        cols_lower = {c.lower(): c for c in label_cols}
        if "firstname" in cols_lower and "lastname" in cols_lower:
            fn = str(r.get(cols_lower["firstname"], "")).strip()
            ln = str(r.get(cols_lower["lastname"], "")).strip()
            other = [c for c in label_cols if c not in (cols_lower["firstname"], cols_lower["lastname"])]
            parts = [f"{fn} {ln}".strip()] + [str(r.get(c, "")) for c in other]
        else:
            parts = [str(r.get(c, "")) for c in label_cols]
        return " · ".join(p for p in parts if p) or "—"

    def _fmt_metric(col: str, r: dict) -> str:
        try:
            val = float(r.get(col))  # type: ignore[arg-type]
            prefix = "$" if any(kw in col.lower() for kw in _METRIC_KEYWORDS) else ""
            return f"{prefix}{val:,.2f}" if val != int(val) else f"{prefix}{int(val):,}"
        except (TypeError, ValueError):
            return str(r.get(col, "—"))

    primary = numeric_cols[0] if numeric_cols else None
    extras = numeric_cols[1:3]

    lines = []
    for rank, r in enumerate(rows, 1):
        label = _row_label(r)
        if primary:
            val_str = _fmt_metric(primary, r)
            extra_str = ", ".join(f"{em}: `{_fmt_metric(em, r)}`" for em in extras)
            suffix = f"  ({extra_str})" if extra_str else ""
            lines.append(f"{rank}. **{label}** — `{val_str}`{suffix}")
        else:
            lines.append(f"{rank}. **{label}**")
    return "\n".join(lines)


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

    # ── 2+ columns: one line per row, all columns styled equally ─────────────
    return "\n\n".join(
        "  ·  ".join(f"{c}: `{_fmt_value(r.get(c))}`" for c in cols)
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
        # Detect results where every non-count/aggregate value is NULL or empty.
        # This happens when a view exists but its underlying data is not populated
        # (e.g. XML demographic fields). Surfacing a table of NULLs is misleading.
        non_id_cols = [
            c for c in sql_result[0].keys()
            if not any(c.lower().endswith(s) for s in ("id", "count", "total", "sum"))
        ]
        if non_id_cols and all(
            row.get(c) in (None, "", "None") for row in sql_result for c in non_id_cols
        ):
            return {
                "final_answer": (
                    "The query ran successfully but the requested data is not populated "
                    "in this database. The relevant fields exist in the schema but contain "
                    "no values — this information may not have been loaded into the system."
                )
            }

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
            # Part 1 — show all rows formatted (≤ threshold, fits inline).
            # Use ranked list whenever there is at least one label column and one
            # numeric column — this covers both explicit ranking queries ("top 10")
            # and any breakdown ("by territory", "by category") where a clean
            # label → value format is more readable than the raw col: value style.
            _lc, _nc = _classify_columns(sql_result)
            use_ranked = bool(_lc) and bool(_nc)
            formatter = _format_ranked_list if use_ranked else _format_rows
            facts_block = (
                f"Query returned {n_rows} row(s):\n\n"
                + formatter(sql_result)
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
        # UndefinedColumn: column doesn't exist in the schema
        undef_match = re.search(
            r'column\s+["\']?([a-zA-Z_][a-zA-Z0-9_.]*)["\']?\s+does not exist',
            error, re.IGNORECASE,
        )
        if undef_match:
            col_ref = undef_match.group(1).split(".")[-1]
            tables_with_col = column_exists_anywhere(col_ref)
            if tables_with_col:
                return {"final_answer": (
                    f"The column **`{col_ref}`** was not found where expected, "
                    f"but it does exist in: {', '.join(f'`{t}`' for t in tables_with_col)}. "
                    f"Try rephrasing your question to reference those tables."
                )}
            else:
                return {"final_answer": (
                    f"This information is not stored in the database — "
                    f"the field **`{col_ref}`** does not exist in any table.\n\n"
                    f"Try rephrasing your question using columns that are available in the schema. "
                    f"If you are asking about a derived metric (e.g. profit margin, revenue, cost), "
                    f"try asking directly — for example: "
                    f"*\"show me products ranked by (list price minus standard cost)\"* "
                    f"or *\"what is the revenue per product category\"*."
                )}

        # GroupingError: SQL aggregation mistake — retry already failed, give clean message
        if "groupingerror" in error.lower() or "must appear in the group by clause" in error.lower():
            return {"final_answer": (
                "This question could not be answered — the database query ran into an "
                "aggregation issue after multiple attempts.\n\n"
                "Try rephrasing your question more specifically, for example:\n"
                "- Specify which metric to aggregate (e.g. *count*, *average*, *total*)\n"
                "- Narrow the scope (e.g. *per department* instead of a full distribution)"
            )}

        # UndefinedTable: table referenced doesn't exist
        if "relation" in error.lower() and "does not exist" in error.lower():
            tbl_match = re.search(r'relation\s+"?([^"]+)"?\s+does not exist', error, re.IGNORECASE)
            tbl = tbl_match.group(1) if tbl_match else "a table"
            return {"final_answer": (
                f"The query referenced **`{tbl}`** which does not exist in the database. "
                f"Please rephrase your question."
            )}


        # Syntax error at or near "AS" (often EXTRACT/CAST/ROUND mistakes)
        if (
            "syntax error at or near \"as\"" in error.lower()
            or ("syntax error" in error.lower() and "extract" in error.lower())
            or ("syntax error" in error.lower() and "cast" in error.lower())
        ):
            return {"final_answer": (
                "There was a SQL syntax error, likely related to casting or aliasing. "
                "Common causes:\n"
                "- Use EXTRACT(YEAR FROM col)::int AS year, not (EXTRACT(YEAR FROM col) AS INT) AS year\n"
                "- Use ROUND((expr)::numeric, 2) for rounding, not ROUND(expr, 2) or ROUND(CAST(expr AS float), 2)\n"
                "- Always use ::int or ::numeric for type casts in PostgreSQL.\n\n"
                "Please review your calculation and casting expressions."
            )}

        # Numeric casting/rounding error: user-friendly message
        if (
            "round(" in error.lower() and "numeric" in error.lower()
        ) or (
            "syntax error" in error.lower() and "round" in error.lower()
        ):
            return {"final_answer": (
                "There was a SQL syntax error related to numeric casting or rounding. "
                "Please check your calculation expressions (e.g., use ROUND((expr)::numeric, 2))."
            )}

        # Generic fallback: strip psycopg2 boilerplate, show clean message
        clean_error = re.sub(r'Error executing tool run_sql_query:\s*', '', error)
        clean_error = re.sub(r'\(psycopg2\.\w+\.\w+\)\s*', '', clean_error)
        clean_error = re.sub(r'\nLINE \d+:.*', '', clean_error, flags=re.DOTALL).strip()
        return {"final_answer": (
            f"The query could not be completed.\n\n"
            f"**Error:** {clean_error}"
        )}

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
