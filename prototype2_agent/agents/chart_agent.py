"""Chart generation agent.

Receives sql_result and user_query, decides the best chart type and axes,
then generates the chart using Plotly. No MCP tools needed.
"""

import base64
import json

from langchain_core.messages import SystemMessage, HumanMessage
import plotly.graph_objects as go

from state import AgentState
from llm_config import get_llm

# ── System prompt ─────────────────────────────────────────────────────────────
CHART_SYSTEM_PROMPT = """\
You are a Chart specialist agent. You receive SQL query results (as a JSON list of dicts)
and the user's original question.

Your job:
1. Decide the best chart type: "bar", "line", "pie", "scatter", or "table".
2. Pick the best x-axis column and y-axis column from the data keys.
3. Write a short, descriptive title for the chart.

Respond with ONLY a JSON object (no markdown fences):
{"chart_type": "<type>", "x": "<column>", "y": "<column>", "title": "<title>"}

For pie charts, "x" is the labels column and "y" is the values column.
If the data is not suitable for a chart, respond: {"chart_type": "table", "x": "", "y": "", "title": "Data Table"}
"""


def generate_chart(
    data: list[dict], chart_type: str, x: str, y: str, title: str
) -> str:
    """Render a Plotly figure and return a base64-encoded PNG string.

    Args:
        data: List of row dicts from the SQL result.
        chart_type: One of "bar", "line", "pie", "scatter", "table".
        x: Column name for x-axis (or labels for pie).
        y: Column name for y-axis (or values for pie).
        title: Chart title.

    Returns:
        Base64-encoded PNG string.
    """
    x_vals = [row.get(x, "") for row in data]
    y_vals = [row.get(y, 0) for row in data]

    fig = go.Figure()

    if chart_type == "bar":
        fig.add_trace(go.Bar(x=x_vals, y=y_vals, name=y))
    elif chart_type == "line":
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines+markers", name=y))
    elif chart_type == "pie":
        fig.add_trace(go.Pie(labels=x_vals, values=y_vals))
    elif chart_type == "scatter":
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="markers", name=y))
    else:
        # Table fallback
        fig.add_trace(
            go.Table(
                header=dict(values=list(data[0].keys()) if data else []),
                cells=dict(
                    values=[
                        [row.get(k, "") for row in data]
                        for k in (data[0].keys() if data else [])
                    ]
                ),
            )
        )

    fig.update_layout(title=title, template="plotly_white")

    # Export to PNG bytes
    img_bytes = fig.to_image(format="png", width=800, height=500)
    return base64.b64encode(img_bytes).decode("utf-8")


def chart_agent(state: AgentState) -> AgentState:
    """Decide chart parameters via LLM and render a Plotly chart.

    Args:
        state: Current pipeline state with user_query and sql_result.

    Returns:
        Partial AgentState update with chart_spec containing the base64 PNG.
    """
    user_query = state["user_query"]
    sql_result = state.get("sql_result", [])

    if not sql_result:
        return {
            "chart_spec": {},
            "error": state.get("error", "") or "No data available to chart.",
        }

    # Ask the LLM to decide chart parameters
    llm = get_llm("chart")

    # Show a sample of the data (first 5 rows) to keep token usage low
    sample = sql_result[:5]
    messages = [
        SystemMessage(content=CHART_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Data sample (first {len(sample)} rows):\n{json.dumps(sample, indent=2, default=str)}\n\n"
                f"All columns: {list(sql_result[0].keys())}\n\n"
                f"User question: {user_query}"
            )
        ),
    ]

    response = llm.invoke(messages)
    content = response.content.strip()

    try:
        spec = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: render as table
        spec = {"chart_type": "table", "x": "", "y": "", "title": "Data Table"}

    chart_type = spec.get("chart_type", "bar")
    x_col = spec.get("x", "")
    y_col = spec.get("y", "")
    title = spec.get("title", "Chart")

    try:
        image_b64 = generate_chart(sql_result, chart_type, x_col, y_col, title)
    except Exception as e:
        return {
            "chart_spec": {},
            "error": f"Chart generation failed: {e}",
        }

    return {
        "chart_spec": {
            "image_base64": image_b64,
            "chart_type": chart_type,
            "x": x_col,
            "y": y_col,
            "title": title,
        }
    }
