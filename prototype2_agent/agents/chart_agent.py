"""Chart generation agent.

Receives sql_result and user_query, decides the best chart type and axes,
then generates the chart using Plotly. No MCP tools needed.
"""

import json

from langchain_core.messages import SystemMessage, HumanMessage
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from state import AgentState
from llm_config import get_llm

# ── System prompt ─────────────────────────────────────────────────────────────
CHART_SYSTEM_PROMPT = """\
You are a Chart specialist agent for business analytics. You receive SQL query results
and the user's original question.

Your job: rank the top 3 most suitable chart types and return their configurations.

Available chart types and when to use them:
- "bar"             → comparing discrete categories; best for ranked or unordered groups
                      (e.g. revenue per product, headcount per department)
- "grouped_bar"     → comparing the SAME metric across multiple series side-by-side;
                      set x to the category column, y to the metric column, group to the series column;
                      requires at least 2 distinct values in the group column
                      (e.g. sales by product AND year, headcount by department AND gender)
- "small_multiples" → one panel per category showing the same bar chart;
                      set x to the x-axis column, y to the metric column, facet to the panel column;
                      best when there are 2–6 panels and each panel has multiple x values
                      (e.g. monthly revenue per region, quarterly units sold per product line)
- "line"            → continuous trends over time with ordered x-axis;
                      if data has a "year" column and a "month" column, ALWAYS set group="year" and x="month"
                      to draw one line per year — NEVER plot multi-year data as a single line;
                      the x-axis will be rendered as Jan–Dec automatically
                      (e.g. monthly sales, daily active users, yearly trend by region)
- "area"            → same as line but emphasises volume/magnitude over time;
                      good for cumulative or stacked metrics; also supports group for multi-series
- "scatter"         → relationship/correlation between two numeric variables
                      (e.g. price vs. units sold, age vs. salary)
- "histogram"       → distribution of a single numeric column; set y to ""; reveals spread and outliers
                      (e.g. order value distribution, age distribution)
- "box"             → statistical spread — median, quartiles, outliers — for a numeric column grouped by a category;
                      use x for the category column and y for the numeric column
                      (e.g. salary by department, order size by region)
- "waterfall"       → incremental positive/negative contributions to a total; ideal for financial P&L, variance analysis
                      (e.g. revenue bridge, budget vs actual breakdown)
- "treemap"         → part-of-whole for hierarchical or categorical data; better than pie for many categories
                      (e.g. revenue share by product, cost breakdown)

Common high-value triplets for business questions:
- "sales/revenue by category"       → bar, treemap, waterfall
- "trend over time"                 → line, area, bar
- "multi-series comparison"         → grouped_bar, small_multiples, bar
- "performance by group over time"  → grouped_bar, small_multiples, line
- "distribution analysis"           → histogram, box, scatter
- "financial breakdown"             → waterfall, bar, treemap
- "correlation analysis"            → scatter, histogram, box

Chart formatting standards (always follow these):
- Title: all 3 charts must share the SAME title — a short, descriptive label for the data (e.g. "Total Sales by Product Category")
- Colors: use at most 7 distinct colors for categories; treemap is exempt and may use as many as needed
- Legend: include a legend whenever the chart has multiple categories or series
- Axes: provide human-readable axis labels (e.g. "Product Category", "Total Sales (USD)"); never leave axes unlabelled
- Y-axis: always starts at 0; the upper bound is computed automatically from the data.
- Tooltips: configure hover tooltips to show the category name and exact value on mouse-over

Respond with ONLY a JSON array of exactly 3 objects (no markdown fences).
IMPORTANT: all 3 chart_type values must be DIFFERENT — no duplicates.
- "group": required for grouped_bar (the column that defines series); set to "" for all other types.
- "facet": required for small_multiples (the column that defines panels); set to "" for all other types.
[
  {"chart_type": "<type>", "x": "<column>", "y": "<column>", "group": "<column or \"\">", "facet": "<column or \"\">", "title": "<title>", "x_label": "<label>", "y_label": "<label>"},
  {"chart_type": "<type>", "x": "<column>", "y": "<column>", "group": "<column or \"\">", "facet": "<column or \"\">", "title": "<title>", "x_label": "<label>", "y_label": "<label>"},
  {"chart_type": "<type>", "x": "<column>", "y": "<column>", "group": "<column or \"\">", "facet": "<column or \"\">", "title": "<title>", "x_label": "<label>", "y_label": "<label>"}
]
"""


# Max 7 distinct colours — colorblind-friendly palette
_PALETTE = [
    "#4C9BE8", "#E8834C", "#4CE8A0", "#E84C6B",
    "#A04CE8", "#E8D44C", "#4CDDE8",
]



def _to_label(val) -> str:
    """Normalize a value to a clean string label.

    Converts whole-number floats (e.g. 1.0, 2024.0) to ints before stringifying
    so that EXTRACT() results from PostgreSQL match category arrays and legend keys.
    """
    try:
        f = float(val)
        if f == int(f):
            return str(int(f))
        return str(f)
    except (TypeError, ValueError):
        return str(val) if val is not None else ""


def _compute_y_max(y_vals: list[float]) -> float:
    """Round the max y value up to the next clean axis boundary.

    Rounds up to the nearest 1, 2, or 5 × 10^n so the top of the chart
    always sits above the highest data point at a clean round number.
    """
    if not y_vals:
        return 1.0
    raw_max = max(y_vals)
    if raw_max <= 0:
        return 1.0
    import math
    magnitude = 10 ** math.floor(math.log10(raw_max))
    for factor in (1, 2, 3, 5, 6, 8, 10):
        candidate = factor * magnitude
        if candidate >= raw_max:
            return float(candidate)
    return float(10 * magnitude)


def _sort_key(val):
    """Sort helper: numeric strings sort as numbers, others as strings."""
    try:
        return (0, float(val))
    except (TypeError, ValueError):
        return (1, str(val))


def generate_chart(
    data: list[dict], chart_type: str, x: str, y: str, title: str,
    x_label: str = "", y_label: str = "",
    group: str = "", facet: str = "",
) -> str:
    """Render a Plotly figure and return it serialised as a JSON string."""
    x_vals = [_to_label(row.get(x, "")) for row in data]
    y_vals = []
    for row in data:
        v = row.get(y, 0)
        try:
            y_vals.append(float(v))
        except (TypeError, ValueError):
            y_vals.append(0)
    hover = "%{x}<br>%{y:,.2f}<extra></extra>"

    fig = go.Figure()

    if chart_type == "grouped_bar":
        series = list(dict.fromkeys(_to_label(row.get(group, "")) for row in data))
        for i, grp in enumerate(series):
            subset = [row for row in data if _to_label(row.get(group, "")) == grp]
            gx = [_to_label(row.get(x, "")) for row in subset]
            gy = []
            for row in subset:
                v = row.get(y, 0)
                try:
                    gy.append(float(v))
                except (TypeError, ValueError):
                    gy.append(0)
            fig.add_trace(go.Bar(
                x=gx, y=gy, name=str(grp),
                marker_color=_PALETTE[i % len(_PALETTE)],
                hovertemplate=f"{grp}<br>%{{x}}<br>%{{y:,.2f}}<extra></extra>",
            ))
        fig.update_layout(barmode="group")
    elif chart_type == "small_multiples":
        panels = list(dict.fromkeys(_to_label(row.get(facet, "")) for row in data))
        n = len(panels)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig = make_subplots(
            rows=nrows, cols=ncols,
            subplot_titles=panels,
            shared_yaxes=True,
        )
        for idx, panel in enumerate(panels):
            r = idx // ncols + 1
            c = idx % ncols + 1
            subset = [row for row in data if _to_label(row.get(facet, "")) == panel]
            px_vals = [_to_label(row.get(x, "")) for row in subset]
            py_vals = []
            for row in subset:
                v = row.get(y, 0)
                try:
                    py_vals.append(float(v))
                except (TypeError, ValueError):
                    py_vals.append(0)
            fig.add_trace(go.Bar(
                x=px_vals, y=py_vals,
                marker_color=_PALETTE[idx % len(_PALETTE)],
                hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>",
                showlegend=False,
            ), row=r, col=c)
    elif chart_type == "bar":
        # If x has duplicates (e.g. month numbers across multiple years),
        # prefix with the year column to produce unique YYYY-MM labels.
        if len(set(x_vals)) < len(x_vals):
            year_col = next(
                (c for c in (data[0].keys() if data else [])
                 if "year" in c.lower() and c != x),
                None,
            )
            if year_col:
                bar_x = [
                    f"{_to_label(row.get(year_col))}-{_to_label(row.get(x, '')).zfill(2)}"
                    for row in data
                ]
            else:
                bar_x = x_vals
        else:
            bar_x = x_vals
        pairs = sorted(zip(bar_x, y_vals), key=lambda p: p[0])
        bx, by = (list(z) for z in zip(*pairs)) if pairs else ([], [])
        fig.add_trace(go.Bar(
            x=bx, y=by, name=y,
            marker_color=_PALETTE[0],
            hovertemplate=hover,
        ))
    elif chart_type == "line":
        if group:
            series = sorted(
                list(dict.fromkeys(_to_label(row.get(group, "")) for row in data)),
                key=_sort_key,
            )
            for i, grp in enumerate(series):
                subset = sorted(
                    [row for row in data if _to_label(row.get(group, "")) == grp],
                    key=lambda r: _sort_key(r.get(x, "")),
                )
                gx = [_to_label(row.get(x, "")) for row in subset]
                gy = [float(row.get(y, 0) or 0) for row in subset]
                fig.add_trace(go.Scatter(
                    x=gx, y=gy, mode="lines+markers", name=grp,
                    line=dict(color=_PALETTE[i % len(_PALETTE)]),
                    hovertemplate=f"{grp}<br>%{{x}}<br>%{{y:,.2f}}<extra></extra>",
                ))
        else:
            pairs = sorted(zip(x_vals, y_vals), key=lambda p: _sort_key(p[0]))
            sx, sy = zip(*pairs) if pairs else ([], [])
            fig.add_trace(go.Scatter(
                x=list(sx), y=list(sy), mode="lines+markers", name=y,
                line=dict(color=_PALETTE[0]),
                hovertemplate=hover,
            ))
    elif chart_type == "area":
        if group:
            series = sorted(
                list(dict.fromkeys(_to_label(row.get(group, "")) for row in data)),
                key=_sort_key,
            )
            for i, grp in enumerate(series):
                subset = sorted(
                    [row for row in data if _to_label(row.get(group, "")) == grp],
                    key=lambda r: _sort_key(r.get(x, "")),
                )
                gx = [_to_label(row.get(x, "")) for row in subset]
                gy = [float(row.get(y, 0) or 0) for row in subset]
                fig.add_trace(go.Scatter(
                    x=gx, y=gy, mode="lines", fill="tozeroy", name=grp,
                    line=dict(color=_PALETTE[i % len(_PALETTE)]),
                    hovertemplate=f"{grp}<br>%{{x}}<br>%{{y:,.2f}}<extra></extra>",
                ))
        else:
            pairs = sorted(zip(x_vals, y_vals), key=lambda p: _sort_key(p[0]))
            sx, sy = zip(*pairs) if pairs else ([], [])
            fig.add_trace(go.Scatter(
                x=list(sx), y=list(sy), mode="lines", fill="tozeroy", name=y,
                line=dict(color=_PALETTE[0]),
                hovertemplate=hover,
            ))
    elif chart_type == "scatter":
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode="markers", name=y,
            marker=dict(color=_PALETTE[0]),
            hovertemplate=hover,
        ))
    elif chart_type == "histogram":
        fig.add_trace(go.Histogram(
            x=x_vals, name=x,
            marker_color=_PALETTE[0],
            hovertemplate="%{x}<br>Count: %{y}<extra></extra>",
        ))
    elif chart_type == "box":
        fig.add_trace(go.Box(
            x=x_vals, y=y_vals, name=y,
            marker_color=_PALETTE[0],
        ))
    elif chart_type == "waterfall":
        measures = ["relative"] * (len(y_vals) - 1) + ["total"]
        fig.add_trace(go.Waterfall(
            x=x_vals, y=y_vals, measure=measures, name=y,
            hovertemplate=hover,
        ))
    elif chart_type == "treemap":
        fig.add_trace(go.Treemap(
            labels=x_vals,
            parents=[""] * len(x_vals),
            values=y_vals,
            branchvalues="total",
            name=y,
            hovertemplate="%{label}<br>%{value:,.2f}<extra></extra>",
        ))
    else:
        fig.add_trace(go.Table(
            header=dict(values=list(data[0].keys()) if data else []),
            cells=dict(values=[
                [row.get(k, "") for row in data]
                for k in (data[0].keys() if data else [])
            ]),
        ))

    _no_cartesian = {"treemap", "small_multiples"}
    _show_legend = {"treemap", "grouped_bar"}

    layout = dict(
        title=dict(text=title, x=0.5, xanchor="center"),
        template="plotly_dark",
        showlegend=chart_type in _show_legend or (chart_type in ("line", "area") and bool(group)),
        hoverlabel=dict(bgcolor="#1e1e2e", font_size=13),
    )

    # Cartesian axis config only applies to charts that use a single x/y axis pair
    if chart_type not in _no_cartesian:
        # Detect month-number x-axis and force correct numeric category order
        all_x = [str(v) for v in x_vals if v != ""]
        _month_nums = [str(i) for i in range(1, 13)]
        if all_x and all(v in _month_nums for v in all_x):
            xaxis_extra = dict(
                categoryorder="array",
                categoryarray=_month_nums,
                tickvals=_month_nums,
                ticktext=["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"],
            )
        else:
            xaxis_extra = {}

        layout["xaxis"] = dict(
            title=x_label or x,
            tickangle=-30,
            **xaxis_extra,
        )
        # Waterfall bars stack cumulatively, so y_max must cover the running total
        if chart_type == "waterfall":
            cumulative_max = sum(v for v in y_vals if v > 0)
            effective_y_vals = [cumulative_max]
        else:
            effective_y_vals = y_vals
        layout["yaxis"] = dict(
            title=y_label or y,
            tickformat=",.0f",
            showgrid=True,
            gridwidth=1,
            range=[0, _compute_y_max(effective_y_vals)],
        )

    fig.update_layout(**layout)
    return fig.to_json()


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
        specs = json.loads(content)
        if not isinstance(specs, list):
            specs = [specs]
    except json.JSONDecodeError:
        specs = [{"chart_type": "bar", "x": "", "y": "", "title": "Chart"}]

    options = []
    seen_types = set()
    for spec in specs[:3]:
        chart_type = spec.get("chart_type", "bar")
        if chart_type in seen_types:
            continue
        seen_types.add(chart_type)
        x_col = spec.get("x", "")
        y_col = spec.get("y", "")
        title = spec.get("title", "Chart")
        x_label = spec.get("x_label", "")
        y_label = spec.get("y_label", "")
        group_col = spec.get("group", "")
        facet_col = spec.get("facet", "")

        # Validate columns exist; fall back to first/last column if not found
        actual_cols = list(sql_result[0].keys())
        if x_col not in actual_cols:
            x_col = next(
                (c for c in actual_cols
                 if not isinstance(sql_result[0].get(c), (int, float))),
                actual_cols[0],
            )
        if y_col not in actual_cols:
            y_col = actual_cols[-1]

        try:
            figure_json = generate_chart(
                sql_result, chart_type, x_col, y_col, title,
                x_label, y_label, group_col, facet_col,
            )
            options.append({"figure_json": figure_json, "chart_type": chart_type, "title": title})
        except Exception:
            continue

    if not options:
        return {"chart_spec": {}, "error": "Chart generation failed for all options."}

    return {"chart_spec": {"options": options}}
