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

import os
import sys

import plotly.io as pio
import sqlglot
import streamlit as st
import streamlit.components.v1 as components

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
    rag_chunks: list[dict] | None,
    rag_fallback: bool = False,
    chart_options: list[dict] | None = None,
    turn_key: str = "",
) -> None:
    """Render all components of one assistant response inside the current chat bubble.

    Args:
        final_answer:  Markdown-formatted LLM answer.
        sql_query:     Raw SQL string; empty string when no query was run.
        sql_result:    List of row-dicts from SQL execution; empty list when none.
        rag_chunks:    List of retrieved document chunks; None when no RAG ran.
        rag_fallback:  True when fallback search was used (below-threshold results).
        chart_options: List of up to 3 dicts with keys figure_json, chart_type, title.
        turn_key:      Unique key per turn to avoid widget ID collisions.
    """
    # 1. LLM answer
    st.markdown(final_answer)

    # 2. Raw data table (SQL flow)
    if sql_result:
        with st.expander(f"Raw Data — {len(sql_result)} rows", expanded=False):
            st.dataframe(sql_result, width="stretch")

    # 3. Retrieved document chunks (RAG flow)
    if rag_chunks:
        # Fallback notification
        if rag_fallback:
            st.warning(
                f"No documents scored above the {0.55} similarity threshold. "
                f"Showing best-effort matches from a fallback search "
                f"(highest score: {rag_chunks[0].get('score', '?')})."
            )

        label = "Retrieved Sources (fallback)" if rag_fallback else "Retrieved Sources"
        with st.expander(f"{label} — {len(rag_chunks)} chunk(s)", expanded=False):
            for i, c in enumerate(rag_chunks):
                source = c.get("source", "unknown")
                score = c.get("score", "?")
                st.markdown(f"**Chunk {i + 1}** — `{source}` (similarity: {score})")
                st.text(c.get("content", ""))
                if i < len(rag_chunks) - 1:
                    st.divider()

    # 4. SQL query
    if sql_query:
        try:
            formatted_sql = sqlglot.transpile(sql_query, pretty=True, read="postgres", write="postgres")[0]
        except Exception:
            formatted_sql = sql_query
        with st.expander("SQL Query", expanded=False):
            st.code(formatted_sql, language="sql")

    # 4. Interactive chart with type selector
    if chart_options:
        labels = [opt["chart_type"].capitalize() for opt in chart_options]
        with st.expander(chart_options[0]["title"], expanded=True):
            selected = st.radio(
                "Chart type",
                labels,
                horizontal=True,
                key=f"chart_radio_{turn_key}",
                label_visibility="collapsed",
            )
            idx = labels.index(selected)
            fig = pio.from_json(chart_options[idx]["figure_json"])
            _chart_type = chart_options[idx]["chart_type"]

            _scrollbar_css = """<style>
body { overflow-x: auto; overflow-y: hidden; margin: 0; }
::-webkit-scrollbar { height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(180,180,180,0.25); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(180,180,180,0.5); }
</style>"""
            # JavaScript injected for line charts: bold the hovered trace and dim
            # the others, then restore all on mouse-out.
            _line_hover_js = """<script>
(function() {
  var gd = document.querySelector('.plotly-graph-div');
  if (!gd) return;
  var _defaultWidths = null;
  gd.on('plotly_hover', function(eventData) {
    var n = gd.data.length;
    if (!_defaultWidths) {
      _defaultWidths = gd.data.map(function(t) {
        return (t.line && t.line.width != null) ? t.line.width : 2;
      });
    }
    var hovered = eventData.points[0].curveNumber;
    var widths = [], opacities = [];
    for (var i = 0; i < n; i++) {
      if (i === hovered) { widths.push(2); opacities.push(1.0); }
      else               { widths.push(1); opacities.push(0.6); }
    }
    Plotly.restyle(gd, {'line.width': widths, opacity: opacities});
  });
  gd.on('plotly_unhover', function() {
    if (!_defaultWidths) return;
    var n = gd.data.length;
    var widths = [], opacities = [];
    for (var i = 0; i < n; i++) {
      widths.push(_defaultWidths[i]);
      opacities.push(1.0);
    }
    Plotly.restyle(gd, {'line.width': widths, opacity: opacities});
  });
})();
</script>"""

            if fig.layout.width is not None or _chart_type == "line":
                # Wide charts → horizontal scroll wrapper.
                # Line charts → always use HTML so we can inject hover JavaScript.
                _cfg = {"responsive": False} if fig.layout.width is not None else {"responsive": True}
                chart_html = fig.to_html(
                    full_html=True,
                    include_plotlyjs="cdn",
                    config=_cfg,
                ).replace("</head>", f"{_scrollbar_css}</head>", 1)
                if _chart_type == "line":
                    chart_html = chart_html.replace("</body>", f"{_line_hover_js}</body>", 1)
                _height = fig.layout.height + 40 if fig.layout.height else 520
                components.html(
                    chart_html,
                    height=_height,
                    scrolling=fig.layout.width is not None,
                )
            else:
                st.plotly_chart(fig, width="stretch")


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
                rag_chunks=msg.get("rag_chunks"),
                rag_fallback=msg.get("rag_fallback", False),
                chart_options=msg.get("chart_options"),
                turn_key=msg.get("turn_key", str(id(msg))),
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
                rag_chunks: list[dict] | None = result.get("rag_chunks") or None
                rag_fallback: bool = result.get("rag_fallback", False)
                chart_spec: dict = result.get("chart_spec", {})

                chart_options = chart_spec.get("options") if chart_spec else None
                turn_key = str(len(st.session_state.messages))

                _render_assistant_turn(
                    final_answer=final_answer,
                    sql_query=sql_query,
                    sql_result=sql_result,
                    rag_chunks=rag_chunks,
                    rag_fallback=rag_fallback,
                    chart_options=chart_options,
                    turn_key=turn_key,
                )

                # Store raw data so the replay loop re-renders on reload
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_answer,
                    "sql_query": sql_query,
                    "sql_result": sql_result,
                    "rag_chunks": rag_chunks,
                    "rag_fallback": rag_fallback,
                    "chart_options": chart_options,
                    "turn_key": turn_key,
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
