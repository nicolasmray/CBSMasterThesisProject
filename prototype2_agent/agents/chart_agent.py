"""Chart generation agent.

Receives sql_result and user_query, decides the best chart type and axes,
then generates the chart using Plotly. No MCP tools needed.
"""

import calendar
import json
import math

from langchain_core.messages import SystemMessage, HumanMessage
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from state import AgentState
from llm_config import invoke_with_retry

# ── System prompt ─────────────────────────────────────────────────────────────
CHART_SYSTEM_PROMPT = """\
You are a Chart specialist agent for business analytics. You receive SQL query results
and the user's original question.

Your job: rank the top 2-3 most suitable chart types and return their configurations.

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
                      if data has a "year" column and a "month" column AND the query is about
                      raw values (revenue, sales, count), ALWAYS set group="year" and x="month"
                      to draw one line per year — the x-axis will be rendered as Jan–Dec automatically
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
- "donut"           → part-of-whole for a small number of categories (≤7); highlights the largest segment visually;
                      set x to the label column and y to the value column
                      (e.g. revenue share by product category, sales split by region)

Common high-value triplets for business questions:
- "sales/revenue by category"       → bar, treemap, donut
- "trend over time"                 → line, area, bar
- "multi-series comparison"         → grouped_bar, small_multiples, bar
- "performance by group over time"  → grouped_bar, small_multiples, line
- "distribution analysis"           → histogram, box, scatter
- "financial breakdown"             → waterfall, bar, treemap
- "correlation analysis"            → scatter, histogram, box
- "period-over-period growth/change"→ bar (y=pct_change), area (y=pct_change), waterfall

IMPORTANT — growth and change queries:
If the result contains a column whose name includes "pct", "change", "growth", "delta",
or "diff", the user almost certainly wants to SEE that column, not the raw value column.
In that case:
- Set y to the change/pct column (e.g. pct_change) for the primary chart option.
- Use a bar chart as the first option — bars make positive vs. negative change immediately visible.
- Use an area chart as the second option — the filled area above/below zero makes gains and losses vivid.
- A waterfall chart is a valid third option for sequential period changes; use the pct column as y (not abs_change or revenue).
- Only offer the raw value column (e.g. revenue) as a last resort, not the primary.
- Do NOT plot abs_change, revenue, or any raw metric column as y when a pct/rate column is present.
- When multiple change columns exist (e.g. abs_change AND pct_change), always prefer the pct column.
- If the data contains a "period" column (pre-formatted chronological label like "Jan'22"),
  set x="period" and group="" — do not use separate year/month columns as x or group.

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

_MONTH_NUMS = [str(i) for i in range(1, 13)]
_MONTH_ABBRS = [calendar.month_abbr[i] for i in range(1, 13)]


def _to_label(val) -> str:
    """Normalize a value to a clean string label."""
    try:
        f = float(val)
        if f == int(f):
            return str(int(f))
        return str(f)
    except (TypeError, ValueError):
        return str(val) if val is not None else ""


def _sort_key(val):
    """Sort helper: numeric strings sort as numbers, others as strings."""
    try:
        return (0, float(val))
    except (TypeError, ValueError):
        return (1, str(val))


def _aggregate(rows: list[dict], x_col: str, y_col: str) -> tuple[list, list]:
    """Sum y values for duplicate x keys and return sorted (x_vals, y_vals)."""
    agg: dict[str, float] = {}
    for row in rows:
        xk = _to_label(row.get(x_col, ""))
        try:
            agg[xk] = agg.get(xk, 0.0) + float(row.get(y_col, 0) or 0)
        except (TypeError, ValueError):
            agg.setdefault(xk, 0.0)
    pairs = sorted(agg.items(), key=lambda p: _sort_key(p[0]))
    if not pairs:
        return [], []
    xs, ys = zip(*pairs)
    return list(xs), list(ys)


def _compute_y_max(y_vals: list[float]) -> float:
    """Round the max y value up to the next clean axis boundary."""
    if not y_vals:
        return 1.0
    raw_max = max(y_vals)
    if raw_max <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(raw_max))
    for factor in (1, 2, 3, 5, 6, 8, 10):
        candidate = factor * magnitude
        if candidate >= raw_max:
            return float(candidate)
    return float(10 * magnitude)


def _add_grouped_traces(
    fig: go.Figure,
    data: list[dict],
    x: str,
    y: str,
    group: str,
    trace_type: str,  # "bar" | "line" | "area"
) -> list[float]:
    """Add one trace per group value and return all aggregated y values.

    Centralises the identical loop used by grouped_bar, line (grouped),
    and area (grouped). Returns the flat list of y values across all series
    so the caller can compute the y-axis range without re-aggregating.
    """
    series = sorted(dict.fromkeys(_to_label(r.get(group, "")) for r in data), key=_sort_key)
    all_y: list[float] = []
    for i, grp in enumerate(series):
        subset = [r for r in data if _to_label(r.get(group, "")) == grp]
        gx, gy = _aggregate(subset, x, y)
        all_y.extend(gy)
        color = _PALETTE[i % len(_PALETTE)]
        ht = f"{grp}<br>%{{x}}<br>%{{y:,.2f}}<extra></extra>"
        if trace_type == "bar":
            fig.add_trace(go.Bar(x=gx, y=gy, name=str(grp), marker_color=color, hovertemplate=ht))
        elif trace_type == "line":
            fig.add_trace(go.Scatter(x=gx, y=gy, mode="lines+markers", name=grp,
                                     line=dict(color=color), hovertemplate=ht))
        elif trace_type == "area":
            fig.add_trace(go.Scatter(x=gx, y=gy, mode="lines", fill="tozeroy", name=grp,
                                     line=dict(color=color), hovertemplate=ht))
    return all_y


def generate_chart(
    data: list[dict], chart_type: str, x: str, y: str, title: str,
    x_label: str = "", y_label: str = "",
    group: str = "", facet: str = "",
) -> str:
    """Render a Plotly figure and return it serialised as a JSON string."""
    x_vals = [_to_label(row.get(x, "")) for row in data]
    y_vals = []
    for row in data:
        try:
            y_vals.append(float(row.get(y, 0) or 0))
        except (TypeError, ValueError):
            y_vals.append(0.0)

    hover = "%{x}<br>%{y:,.2f}<extra></extra>"
    fig = go.Figure()
    effective_y_vals = y_vals  # default; overridden per chart type below
    _period_tick_labels: list[str] | None = None  # set by diverging area branch

    # ── Chart type dispatch ────────────────────────────────────────────────────
    if chart_type == "grouped_bar":
        effective_y_vals = _add_grouped_traces(fig, data, x, y, group, "bar")
        fig.update_layout(barmode="group")

    elif chart_type == "small_multiples":
        panels = list(dict.fromkeys(_to_label(row.get(facet, "")) for row in data))
        n = len(panels)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=panels, shared_yaxes=True)
        for idx, panel in enumerate(panels):
            r, c = idx // ncols + 1, idx % ncols + 1
            subset = [row for row in data if _to_label(row.get(facet, "")) == panel]
            px_vals, py_vals = _aggregate(subset, x, y)
            fig.add_trace(go.Bar(
                x=px_vals, y=py_vals,
                marker_color=_PALETTE[idx % len(_PALETTE)],
                hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>",
                showlegend=False,
            ), row=r, col=c)
            if px_vals and all(v in _MONTH_NUMS for v in px_vals):
                axis_key = "xaxis" if idx == 0 else f"xaxis{idx + 1}"
                fig.update_layout(**{axis_key: dict(
                    categoryorder="array", categoryarray=_MONTH_NUMS,
                    tickvals=_MONTH_NUMS, ticktext=_MONTH_ABBRS,
                )})

    elif chart_type == "bar":
        # Prefix duplicate x values (e.g. month numbers across years) with year
        if len(set(x_vals)) < len(x_vals):
            year_col = next(
                (c for c in (data[0].keys() if data else []) if "year" in c.lower() and c != x),
                None,
            )
            bar_x = (
                [f"{_to_label(row.get(year_col))}-{_to_label(row.get(x, '')).zfill(2)}" for row in data]
                if year_col else x_vals
            )
        else:
            bar_x = x_vals

        if x == "period":
            bx, by = list(bar_x), list(y_vals)
            bar_colors = ["#4CE8A0" if v >= 0 else "#E84C6B" for v in by]
        else:
            bx, by = list(bar_x), list(y_vals)
            bar_colors = _PALETTE[0]
        fig.add_trace(go.Bar(x=bx, y=by, name=y, marker_color=bar_colors, hovertemplate=hover))
        effective_y_vals = list(by) if by else y_vals

    elif chart_type == "line":
        if group:
            effective_y_vals = _add_grouped_traces(fig, data, x, y, group, "line")
        else:
            sx = x_vals if x == "period" else [p[0] for p in sorted(zip(x_vals, y_vals), key=lambda p: _sort_key(p[0]))]
            sy = y_vals if x == "period" else [p[1] for p in sorted(zip(x_vals, y_vals), key=lambda p: _sort_key(p[0]))]
            fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines+markers", name=y,
                                     line=dict(color=_PALETTE[0]), hovertemplate=hover))
            effective_y_vals = sy

    elif chart_type == "area":
        if group:
            effective_y_vals = _add_grouped_traces(fig, data, x, y, group, "area")
        elif x == "period":
            # Diverging fill with interpolated zero crossings so each segment's fill
            # boundary exactly follows the data line. Uses numeric x internally so
            # crossing position can be fractionally interpolated between ticks.
            nx = list(range(len(y_vals)))
            # Split into contiguous same-sign segments, inserting y=0 at crossings
            segs: list[tuple[list, list, bool]] = []
            sx, sy, pos = [nx[0]], [y_vals[0]], y_vals[0] >= 0
            for i in range(1, len(y_vals)):
                p, c = y_vals[i - 1], y_vals[i]
                if (p >= 0) != (c >= 0):
                    t = p / (p - c)                  # fraction of the way to zero
                    cx = nx[i - 1] + t               # fractional index at crossing
                    sx.append(cx); sy.append(0.0)
                    segs.append((list(sx), list(sy), pos))
                    sx, sy, pos = [cx, nx[i]], [0.0, c], c >= 0
                else:
                    sx.append(nx[i]); sy.append(c)
            segs.append((list(sx), list(sy), pos))
            shown: set[str] = set()
            for seg_x, seg_y, is_pos in segs:
                name = "Growth" if is_pos else "Decline"
                fig.add_trace(go.Scatter(
                    x=seg_x, y=seg_y, mode="lines", fill="tozeroy",
                    fillcolor="rgba(76,232,160,0.4)" if is_pos else "rgba(232,76,107,0.4)",
                    line=dict(color="#4CE8A0" if is_pos else "#E84C6B", width=1.5),
                    name=name, legendgroup=name, showlegend=name not in shown,
                    hoverinfo="skip",
                ))
                shown.add(name)
            # One invisible trace gives a single hover value per period tick
            fig.add_trace(go.Scatter(x=nx, y=list(y_vals), mode="none",
                                     showlegend=False, hovertemplate=hover))
            effective_y_vals = y_vals
            _period_tick_labels = list(x_vals)  # save labels before overwriting x_vals
            x_vals = [str(i) for i in nx]       # numeric strings so all_x check works
        else:
            pairs = sorted(zip(x_vals, y_vals), key=lambda p: _sort_key(p[0]))
            sx, sy = zip(*pairs) if pairs else ([], [])
            fig.add_trace(go.Scatter(x=list(sx), y=list(sy), mode="lines", fill="tozeroy",
                                     name=y, line=dict(color=_PALETTE[0]), hovertemplate=hover))
            effective_y_vals = list(sy)

    elif chart_type == "scatter":
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="markers", name=y,
                                 marker=dict(color=_PALETTE[0]), hovertemplate=hover))

    elif chart_type == "histogram":
        fig.add_trace(go.Histogram(x=x_vals, name=x, marker_color=_PALETTE[0],
                                   hovertemplate="%{x}<br>Count: %{y}<extra></extra>"))

    elif chart_type == "box":
        fig.add_trace(go.Box(x=x_vals, y=y_vals, name=y, marker_color=_PALETTE[0]))

    elif chart_type == "waterfall":
        total = sum(y_vals)
        fig.add_trace(go.Waterfall(
            x=list(x_vals) + ["Total"],
            y=list(y_vals) + [total],
            measure=["relative"] * len(y_vals) + ["total"],
            name=y, hovertemplate=hover,
        ))
        effective_y_vals = [sum(v for v in y_vals if v > 0)]  # y_max covers running total

    elif chart_type == "treemap":
        fig.add_trace(go.Treemap(
            labels=x_vals, parents=[""] * len(x_vals), values=y_vals,
            branchvalues="total", name=y,
            hovertemplate="%{label}<br>%{value:,.2f}<extra></extra>",
        ))

    elif chart_type == "donut":
        # Cap at top 9 by value; bucket the rest as "Other"
        if len(x_vals) > 10:
            pairs_d = sorted(zip(y_vals, x_vals), reverse=True)
            top_vals = [v for v, _ in pairs_d[:9]]
            top_labels = [lbl for _, lbl in pairs_d[:9]]
            other_val = sum(v for v, _ in pairs_d[9:])
            donut_labels = top_labels + ["Other"]
            donut_values = top_vals + [other_val]
        else:
            donut_labels, donut_values = list(x_vals), list(y_vals)
        fig.add_trace(go.Pie(
            labels=donut_labels, values=donut_values,
            hole=0.70,
            marker=dict(colors=_PALETTE),
            hovertemplate="%{label}<br>%{value:,.2f} (%{percent})<extra></extra>",
            textinfo="percent",
            textposition="outside",
            outsidetextfont=dict(size=12),
        ))

    else:
        fig.add_trace(go.Table(
            header=dict(values=list(data[0].keys()) if data else []),
            cells=dict(values=[[row.get(k, "") for row in data] for k in (data[0].keys() if data else [])]),
        ))

    # ── Layout ────────────────────────────────────────────────────────────────
    _no_cartesian = {"treemap", "small_multiples", "donut"}
    show_legend = (
        chart_type in {"treemap", "grouped_bar", "donut"}
        or (chart_type in ("line", "area") and bool(group))
        or (chart_type == "area" and x == "period")
    )
    layout = dict(
        title=dict(text=title, x=0.5, xanchor="center"),
        template="plotly_dark",
        showlegend=show_legend,
        hoverlabel=dict(bgcolor="#1e1e2e", font_size=13),
    )

    if chart_type not in _no_cartesian:
        # X-axis: month numbers → abbreviated labels; period → preserve trace order
        all_x = [v for v in x_vals if v != ""]
        if all_x and all(v in _MONTH_NUMS for v in all_x):
            xaxis_extra = dict(categoryorder="array", categoryarray=_MONTH_NUMS,
                               tickvals=_MONTH_NUMS, ticktext=_MONTH_ABBRS)
        elif x == "period":
            if _period_tick_labels:
                # Diverging area used numeric x — map indices back to period labels
                xaxis_extra = dict(
                    tickmode="array",
                    tickvals=list(range(len(_period_tick_labels))),
                    ticktext=_period_tick_labels,
                )
            else:
                xaxis_extra = dict(categoryorder="trace")
        elif chart_type == "bar":
            # Preserve SQL ORDER BY order — do not let Plotly re-sort categories
            xaxis_extra = dict(categoryorder="trace")
        else:
            xaxis_extra = {}
        # Wide bar charts: give each bar 30 px so the chart is physically wider
        # than the viewport. The UI renders it with use_container_width=False so
        # Streamlit wraps it in a native horizontal scroll container.
        _SCROLL_THRESHOLD = 30
        if chart_type == "bar" and len(all_x) > _SCROLL_THRESHOLD:
            layout["width"] = max(1000, len(all_x) * 30)
            layout["height"] = 480

        layout["xaxis"] = dict(title=x_label or x, tickangle=-30, **xaxis_extra)

        # Y-axis range: extend below zero when data has negative values
        y_min = min(effective_y_vals) if effective_y_vals else 0
        y_range = (
            [y_min * 1.1, _compute_y_max(effective_y_vals)]
            if y_min < 0
            else [0, _compute_y_max(effective_y_vals)]
        )
        layout["yaxis"] = dict(title=y_label or y, tickformat=",.0f",
                               showgrid=True, gridwidth=1, range=y_range)

    fig.update_layout(**layout)
    return fig.to_json()


def chart_agent(state: AgentState) -> AgentState:
    """Decide chart parameters via LLM and render a Plotly chart."""
    user_query = state["user_query"]
    sql_result = state.get("sql_result", [])

    if not sql_result:
        return {"chart_spec": {}, "error": state.get("error", "") or "No data available to chart."}

    if len(sql_result) == 1:
        return {"chart_spec": {}}

    # ── Period label pre-processing ────────────────────────────────────────────
    # When year+month columns exist alongside a rate/change column, inject a
    # combined "period" label (e.g. "Jan'22") as an extra column. The LLM sees
    # it in the sample and can choose it as x to get a clean sequential axis.
    _rate_kw = ("pct", "percent", "growth", "change", "rate", "ratio", "diff", "delta")
    cols = list(sql_result[0].keys())
    if "year" in cols and "month" in cols and any(any(kw in c.lower() for kw in _rate_kw) for c in cols):
        sql_result.sort(key=lambda r: (int(r.get("year", 0)), int(r.get("month", 0))))
        for row in sql_result:
            try:
                row["period"] = f"{calendar.month_abbr[int(row['month'])]}'{str(int(row['year']))[-2:]}"
            except (KeyError, ValueError, IndexError):
                row["period"] = f"{row.get('year')}-{row.get('month')}"

    # ── LLM chart selection ────────────────────────────────────────────────────
    llm = get_llm("chart")
    sample = sql_result[:5]
    messages = [
        SystemMessage(content=CHART_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Data sample (first {len(sample)} rows):\n{json.dumps(sample, indent=2, default=str)}\n\n"
            f"All columns: {list(sql_result[0].keys())}\n\n"
            f"User question: {user_query}"
        )),
    ]

    response = invoke_with_retry("chart", messages)
    content = response.content.strip()

    try:
        specs = json.loads(content)
        if not isinstance(specs, list):
            specs = [specs]
    except json.JSONDecodeError:
        specs = [{"chart_type": "bar", "x": "", "y": "", "title": "Chart"}]

    # ── Render each chart option ───────────────────────────────────────────────
    options = []
    seen_types: set[str] = set()
    actual_cols = list(sql_result[0].keys())

    for spec in specs[:3]:
        chart_type = spec.get("chart_type", "bar")
        if chart_type in seen_types:
            continue
        seen_types.add(chart_type)

        x_col = spec.get("x", "")
        y_col = spec.get("y", "")

        # Fallback if LLM picked a column that doesn't exist
        if x_col not in actual_cols:
            x_col = next((c for c in actual_cols if not isinstance(sql_result[0].get(c), (int, float))), actual_cols[0])
        if y_col not in actual_cols:
            y_col = actual_cols[-1]

        try:
            figure_json = generate_chart(
                sql_result, chart_type, x_col, y_col,
                spec.get("title", "Chart"),
                spec.get("x_label", ""), spec.get("y_label", ""),
                spec.get("group", ""), spec.get("facet", ""),
            )
            options.append({"figure_json": figure_json, "chart_type": chart_type, "title": spec.get("title", "Chart")})
        except Exception:
            continue

    if not options:
        return {"chart_spec": {}, "error": "Chart generation failed for all options."}

    return {"chart_spec": {"options": options}}
