"""Chart generation agent.

Receives sql_result and user_query, decides the best chart type and axes,
then generates the chart using Plotly. No MCP tools needed.
"""

import calendar
import colorsys
import json
import math
from collections import Counter, defaultdict

import numpy as np

from langchain_core.messages import SystemMessage, HumanMessage
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from state import AgentState
from llm_config import invoke_with_retry

# ── System prompt ─────────────────────────────────────────────────────────────
CHART_SYSTEM_PROMPT = """\
You are a Chart specialist agent for business analytics. You receive SQL query results
and the user's original question.

Your job: return 1–3 chart configurations ranked by how well they fit the data and the question.
Return only chart types that are genuinely useful for this specific dataset — do not pad to a fixed count.
A single highly-relevant chart is better than three where the last one is a poor fit.

Available chart types and when to use them:
- "bar"             → comparing discrete categories; best for ranked or unordered groups
                      (e.g. revenue per product, headcount per department)
- "grouped_bar"     → comparing the SAME metric across multiple series side-by-side;
                      set x to the category column, y to the metric column, group to the series column;
                      requires at least 2 distinct values in the group column;
                      do NOT use when the group column has more than 10 distinct values
                      (e.g. sales by product AND year, headcount by department AND gender)
- "stacked_bar"     → shows how a total is composed of parts stacked per category;
                      set x to the category, y to the metric, group to the series column;
                      best when you care about BOTH the total height AND each part's contribution;
                      do NOT use when the group column has more than 10 distinct values
                      (e.g. revenue per region stacked by product category, headcount by department stacked by role)
- "normalized_bar"  → 100% stacked bar — shows each part as a share of the whole per category;
                      set x to the category, y to the metric, group to the series column;
                      use when relative composition matters more than absolute values;
                      do NOT use when the group column has more than 10 distinct values
                      (e.g. sales mix by category, revenue share by region over time)
- "bar-line"           → bar for the primary metric, line for a secondary metric on its own right y-axis;
                      ONLY use when the two metrics have genuinely different units or scales
                      (e.g. revenue in millions vs margin %, order count vs average value);
                      do NOT use when both metrics share the same unit (e.g. both USD, both counts) —
                      use grouped_bar instead so both are directly comparable on one axis;
                      set x to the shared axis, y to the bar metric, y2 to the line metric;
                      (e.g. revenue bars + profit margin line, sales volume bars + growth rate line)
- "small_multiples" → one panel per category showing the same bar chart;
                      set x to the x-axis column, y to the metric column, facet to the panel column;
                      ONLY use when there are 2–6 distinct facet values AND each panel has multiple x values;
                      do NOT use when there are more than 6 panels or more than 10 distinct facet values
                      (e.g. monthly revenue per region, quarterly units sold per product line)
- "line"            → continuous trends over time with ordered x-axis;
                      if data has a "year" column and a "month" column AND the query is about
                      raw values (revenue, sales, count), ALWAYS set group="year" and x="month"
                      to draw one line per year — the x-axis will be rendered as Jan–Dec automatically
                      (e.g. monthly sales, daily active users, yearly trend by region)
- "area"            → same as line but emphasises volume/magnitude over time;
                      good for cumulative or stacked metrics; also supports group for multi-series
- "scatter"         → relationship/correlation between two numeric variables,
                      OR comparison across many data points;
                      do NOT use when x is a low-cardinality categorical column
                      (fewer than ~15 distinct values like product category, region name) —
                      use grouped_bar or bar instead for those comparisons
                      (e.g. price vs. units sold, age vs. salary — good scatter use cases)
- "histogram"       → distribution of a single numeric column; set y to ""; reveals spread and outliers
                      (e.g. order value distribution, age distribution)
- "box"             → statistical spread — median, quartiles, outliers — for a numeric column grouped by a category;
                      use x for the category column and y for the numeric column
                      (e.g. salary by department, order size by region)
- "waterfall"       → incremental positive/negative contributions to a total; ideal for financial P&L, variance analysis;
                      ONLY use when x is a single ordered sequence of categories or periods with NO additional group dimension —
                      do NOT use when the data has multiple category groups (the chart will be distorted)
                      (e.g. revenue bridge, budget vs actual breakdown)
- "treemap"         → part-of-whole for hierarchical or categorical data; better than pie for many categories
                      (e.g. revenue share by product, cost breakdown)
- "donut"           → part-of-whole for a small number of categories (≤7); highlights the largest segment visually;
                      set x to the label column and y to the value column
                      (e.g. revenue share by product category, sales split by region)

Common high-value triplets for business questions:
- "sales/revenue by category"       → bar, treemap, donut
- "trend over time"                 → line, area, bar
- "multi-series comparison"         → grouped_bar, stacked_bar, small_multiples
- "composition / part of whole"     → stacked_bar, normalized_bar, donut
- "performance by group over time"  → grouped_bar, small_multiples, line
- "two metrics side by side"        → bar-line, grouped_bar, bar
- "distribution analysis"           → histogram, box, scatter
- "financial breakdown"             → waterfall, bar, treemap
- "correlation analysis"            → scatter, histogram, box
- "period-over-period growth/change"→ bar (y=pct_change), area (y=pct_change), waterfall
- "growth/change broken down by category" → grouped_bar (x=period, y=pct_change, group=category), small_multiples, line

IMPORTANT — multiple metric columns (e.g. cogs + revenue, actual + budget, metric_a + metric_b):
If the SQL result contains a categorical x-axis column (like category, region, product name)
and TWO OR MORE numeric metric columns of the same unit (e.g. both in USD, both counts),
ALWAYS use grouped_bar as the FIRST option (x=category col, y=one metric col, group will auto-pivot),
NOT a plain bar. A plain bar can only show one metric at a time, which hides the comparison.
Do NOT suggest scatter for this pattern — scatter is for continuous numeric x-axes and correlation analysis,
not for categorical comparisons with a small number of groups.

IMPORTANT — growth and change queries:
If the result contains a column whose name includes "pct", "change", "growth", "delta",
or "diff", the user almost certainly wants to SEE that column, not the raw value column.
In that case:
- Set y to the change/pct column (e.g. pct_change) for the primary chart option.
- If the data also has a category/group dimension (e.g. product category, region, territory),
  use grouped_bar as the first option with x=period col, y=pct_change, group=category col.
  This shows each category's change side by side per period — far more informative than a flat bar.
- If there is NO category dimension, use a plain bar chart as the first option.
- Use an area chart as the second option — the filled area above/below zero makes gains and losses vivid.
- A waterfall chart is a valid third option for sequential period changes; use the pct column as y (not abs_change or revenue).
- Only offer the raw value column (e.g. revenue) as a last resort, not the primary.
- Do NOT plot abs_change, revenue, or any raw metric column as y when a pct/rate column is present.
- When multiple change columns exist (e.g. abs_change AND pct_change), always prefer the pct column.
- If the data contains a "period" column (pre-formatted chronological label like "Jan'22"),
  set x="period" and group="" — do not use separate year/month columns as x or group.

Chart formatting standards (always follow these):
- Title: all charts must share the SAME title — a short, descriptive label for the data (e.g. "Total Sales by Product Category")
- Colors: use at most 7 distinct colors for categories; treemap is exempt and may use as many as needed
- Legend: include a legend whenever the chart has multiple categories or series
- Axes: provide human-readable axis labels (e.g. "Product Category", "Total Sales (USD)"); never leave axes unlabelled
- Y-axis: always starts at 0; the upper bound is computed automatically from the data.
- Tooltips: configure hover tooltips to show the category name and exact value on mouse-over

Respond with ONLY a JSON array of 1–3 objects (no markdown fences).
IMPORTANT: all chart_type values must be DIFFERENT — no duplicates.
- "group": required for grouped_bar, stacked_bar, normalized_bar; set to "" for all other types.
- "facet": required for small_multiples; set to "" for all other types.
- "y2": required for bar-line (the line metric column); set to "" for all other types.
- "y2_label": human-readable label for the y2 axis; set to "" for all other types.
[
  {"chart_type": "<type>", "x": "<col>", "y": "<col>", "y2": "<col or \"\">", "group": "<col or \"\">", "facet": "<col or \"\">", "title": "<title>", "x_label": "<label>", "y_label": "<label>", "y2_label": "<label or \"\">"},
  ...
]
"""


# 10-colour palette — original 7 + 3 new distinct colours
_PALETTE = [
    "#4C9BE8", "#E8834C", "#4CE8A0", "#E84C6B",
    "#A04CE8", "#E8D44C", "#4CDDE8",
    "#E8A84C", "#84E84C", "#E84CB8",
]


def _get_palette(n: int) -> list[str]:
    """Return n visually distinct hex colours.

    For n ≤ 7 the hand-picked palette is used.  For larger n, evenly-spaced
    HSL hues are generated so every series/panel gets a unique colour.
    """
    if n <= len(_PALETTE):
        return _PALETTE[:n]
    colours: list[str] = []
    for i in range(n):
        h = i / n
        r, g, b = colorsys.hls_to_rgb(h, 0.55, 0.70)
        colours.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return colours

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


def _aggregate(rows: list[dict], x_col: str, y_col: str,
               preserve_order: bool = False, mean: bool = False) -> tuple[list, list]:
    """Aggregate y values for duplicate x keys and return (x_vals, y_vals).

    preserve_order: keep first-occurrence order instead of sorting (use for period labels).
    mean: average instead of sum — use for rates, percentages, averages.
    """
    agg: dict[str, float] = {}
    counts: dict[str, int] = {}
    order: list[str] = []
    for row in rows:
        xk = _to_label(row.get(x_col, ""))
        try:
            val = float(row.get(y_col, 0) or 0)
        except (TypeError, ValueError):
            val = 0.0
        if xk not in agg:
            agg[xk] = 0.0
            counts[xk] = 0
            order.append(xk)
        agg[xk] += val
        counts[xk] += 1
    if not agg:
        return [], []
    keys = order if preserve_order else [k for k, _ in sorted(agg.items(), key=lambda p: _sort_key(p[0]))]
    vals = [agg[k] / counts[k] if mean else agg[k] for k in keys]
    return keys, vals


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
    palette = _get_palette(len(series))
    # For line/area traces the data has already been sorted chronologically;
    # preserve that order instead of letting _aggregate re-sort alphabetically
    # (which breaks period labels like "Jan'22", "Feb'22", …).
    _preserve = trace_type in ("line", "area") or x in ("period", "__period__")
    all_y: list[float] = []
    for i, grp in enumerate(series):
        subset = [r for r in data if _to_label(r.get(group, "")) == grp]
        gx, gy = _aggregate(subset, x, y, preserve_order=_preserve)
        all_y.extend(gy)
        color = palette[i]
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
    y2: str = "", y2_label: str = "",
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

    # ── Multi-year month normalisation ────────────────────────────────────────
    # When year + month columns exist and span >1 calendar year, replace the
    # plain month x-axis with "Jan '25" style labels so each point is unique
    # and chronological order is preserved across all chart types.
    _cols = list(data[0].keys()) if data else []
    if "year" in _cols and "month" in _cols and x == "month" and group != "year":
        data = sorted(data, key=lambda r: (int(r.get("year", 0) or 0), int(r.get("month", 0) or 0)))
        _uniq_years = set(int(r.get("year", 0) or 0) for r in data)
        if len(_uniq_years) > 1:
            def _mk_period(r) -> str:
                mo = int(r.get("month", 0) or 0)
                yr = int(r.get("year", 0) or 0)
                abbr = calendar.month_abbr[mo] if 1 <= mo <= 12 else str(mo)
                return f"{abbr} '{str(yr)[-2:]}"
            data = [{**r, "__period__": _mk_period(r)} for r in data]
            x = "__period__"
            x_vals = [row["__period__"] for row in data]
            y_vals = []
            for _r in data:
                try:
                    y_vals.append(float(_r.get(y, 0) or 0))
                except (TypeError, ValueError):
                    y_vals.append(0.0)

    # ── Chart type dispatch ────────────────────────────────────────────────────
    if chart_type == "grouped_bar":
        effective_y_vals = _add_grouped_traces(fig, data, x, y, group, "bar")
        fig.update_layout(barmode="group")

    elif chart_type == "small_multiples":
        # Sort data chronologically so period labels appear in the right order
        panels = list(dict.fromkeys(_to_label(row.get(facet, "")) for row in data))

        # Cap at 6 panels — keep the top panels by total y value
        _MAX_PANELS = 6
        if len(panels) > _MAX_PANELS:
            panel_totals: dict[str, float] = {}
            for row in data:
                pk = _to_label(row.get(facet, ""))
                try:
                    panel_totals[pk] = panel_totals.get(pk, 0.0) + float(row.get(y, 0) or 0)
                except (TypeError, ValueError):
                    pass
            panels = sorted(panels, key=lambda p: panel_totals.get(p, 0.0), reverse=True)[:_MAX_PANELS]
            title = f"{title} (top {_MAX_PANELS} by total {y_label or y})"

        n = len(panels)
        ncols = min(3, n)
        nrows = math.ceil(n / ncols)
        panel_h = 300
        fig = make_subplots(
            rows=nrows, cols=ncols,
            subplot_titles=panels,
            shared_yaxes="all",   # same scale across every panel
            vertical_spacing=0.18,
            horizontal_spacing=0.06,
        )
        _sm_palette = _get_palette(len(panels))
        for idx, panel in enumerate(panels):
            r, c = idx // ncols + 1, idx % ncols + 1
            subset = [row for row in data if _to_label(row.get(facet, "")) == panel]
            px_vals, py_vals = _aggregate(subset, x, y, preserve_order=x in ("period", "__period__"))
            fig.add_trace(go.Bar(
                x=px_vals, y=py_vals,
                marker_color=_sm_palette[idx],
                hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>",
                showlegend=False,
            ), row=r, col=c)
            if px_vals and all(v in _MONTH_NUMS for v in px_vals):
                axis_key = "xaxis" if idx == 0 else f"xaxis{idx + 1}"
                fig.update_layout(**{axis_key: dict(
                    categoryorder="array", categoryarray=_MONTH_NUMS,
                    tickvals=_MONTH_NUMS, ticktext=_MONTH_ABBRS,
                )})
        fig.update_layout(height=panel_h * nrows + 80)

    elif chart_type == "bar":
        # Always aggregate: collapse duplicate x values into one bar per label.
        # Duplicate x values (e.g. the same period appearing once per territory)
        # produce stacked bars in a single Plotly trace even with barmode="group".
        # Aggregating (summing) gives one clean bar per x label.
        # Preserve insertion order for period labels so the axis stays chronological.
        bx, by = _aggregate(data, x, y, preserve_order=x in ("period", "__period__"))
        if x in ("period", "__period__"):
            bar_colors = ["#4CE8A0" if v >= 0 else "#E84C6B" for v in by]
        else:
            n_bars = len(bx)
            highlight_n = 10 if n_bars > 30 else 0
            highlight_color = _PALETTE[0]  # blue for top N
            base_color = "#6A6E78"          # brighter grey for the rest
            if highlight_n > 0:
                # Get indices of top N bars by value
                by_array = np.array(by)
                # Get indices of top N values (descending)
                top_indices = np.argsort(by_array)[-highlight_n:][::-1]
                highlight_set = set(top_indices)
                bar_colors = [
                    highlight_color if i in highlight_set else base_color
                    for i in range(n_bars)
                ]
            else:
                bar_colors = [highlight_color for _ in range(n_bars)]
        fig.add_trace(go.Bar(x=bx, y=by, name=y, marker_color=bar_colors, hovertemplate=hover))
        fig.update_layout(barmode="group")
        effective_y_vals = list(by) if by else y_vals

    elif chart_type == "line":
        # Ensure chronological order: if both year and month columns exist sort by
        # (year, month) first, then rebuild x_vals/y_vals from the sorted data.
        # The chart_agent pre-sort only fires when rate columns are present, so
        # plain "monthly revenue" queries arrive unsorted here.
        if group:
            effective_y_vals = _add_grouped_traces(fig, data, x, y, group, "line")
        else:
            _presorted = x in ("period", "__period__")
            pairs = sorted(zip(x_vals, y_vals), key=lambda p: _sort_key(p[0]))
            sx = x_vals if _presorted else [p[0] for p in pairs]
            sy = y_vals if _presorted else [p[1] for p in pairs]
            # Guard: if x is purely categorical (non-numeric strings, non-temporal),
            # a connected line is misleading — render as a bar instead.
            _all_str_x = sx and all(
                v not in _MONTH_NUMS and not v.isdigit()
                for v in sx if v != ""
            )
            if _all_str_x and x not in ("period", "__period__"):
                bar_colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(sx))]
                fig.add_trace(go.Bar(x=sx, y=sy, name=y, marker_color=bar_colors, hovertemplate=hover))
                fig.update_layout(barmode="group")
            else:
                fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines+markers", name=y,
                                         line=dict(color=_PALETTE[0]), hovertemplate=hover))
            effective_y_vals = sy

    elif chart_type == "area":
        if group:
            effective_y_vals = _add_grouped_traces(fig, data, x, y, group, "area")
        elif x in ("period", "__period__"):
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
        # Parse x as numeric for a continuous axis.
        raw_scatter: list[tuple[float, float]] = []
        for row, yv in zip(data, y_vals):
            try:
                raw_scatter.append((float(row.get(x, 0) or 0), yv))
            except (TypeError, ValueError):
                pass

        # Aggregate duplicate x values by averaging y — collapses vertical stacks
        # into one representative point per x value, revealing the true relationship.
        agg_sums: dict[float, float] = defaultdict(float)
        agg_counts: dict[float, int] = defaultdict(int)
        for xv, yv in raw_scatter:
            agg_sums[xv] += yv
            agg_counts[xv] += 1
        scatter_x = sorted(agg_sums)
        scatter_y = [agg_sums[xv] / agg_counts[xv] for xv in scatter_x]

        fig.add_trace(go.Scatter(
            x=scatter_x, y=scatter_y, mode="markers", name=y,
            marker=dict(
                color=scatter_y,
                colorscale=[[0, "#FFFFFF"], [1, "#4CE8A0"]],  # white → bright green
                size=[max(6, min(20, agg_counts[xv] ** 0.5 * 3)) for xv in scatter_x],
                opacity=0.9,
                showscale=False,
            ),
            hovertemplate="%{x:,.2f}<br>Avg %{y:,.2f}<extra></extra>",
        ))

        # Add OLS trend line if there are enough points and a real correlation exists
        if len(scatter_x) >= 5:
            sx_arr = np.array(scatter_x, dtype=float)
            sy_arr = np.array(scatter_y, dtype=float)
            # Only draw the line when r² > 0.05 (weak or stronger relationship)
            coeffs = np.polyfit(sx_arr, sy_arr, 1)
            y_pred = np.polyval(coeffs, sx_arr)
            ss_res = np.sum((sy_arr - y_pred) ** 2)
            ss_tot = np.sum((sy_arr - sy_arr.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            if r2 > 0.05:
                trend_x = [float(sx_arr.min()), float(sx_arr.max())]
                trend_y = [float(np.polyval(coeffs, trend_x[0])),
                           float(np.polyval(coeffs, trend_x[1]))]
                direction = "↑" if coeffs[0] > 0 else "↓"
                # Insert at index 0 so the line renders behind the scatter dots
                fig.add_trace(go.Scatter(
                    x=trend_x, y=trend_y, mode="lines",
                    name=f"Trend (r²={r2:.2f} {direction})",
                    line=dict(color="#E84C6B", width=2, dash="solid"),
                    hoverinfo="skip",
                ), row=None, col=None)
                # Move trend line to back by reordering traces
                fig.data = (fig.data[-1],) + fig.data[:-1]

        effective_y_vals = scatter_y

    elif chart_type == "histogram":
        # Choose bin count: sqrt rule capped at 20 so unique values don't each get their own bin
        nbins = min(20, max(5, round(len(x_vals) ** 0.5)))
        fig.add_trace(go.Histogram(x=x_vals, name=x, marker_color=_PALETTE[0],
                                   nbinsx=nbins,
                                   hovertemplate="%{x}<br>Count: %{y}<extra></extra>"))

    elif chart_type == "box":
        x_counts = Counter(x_vals)
        if max(x_counts.values(), default=1) <= 1:
            # All x values unique — no meaningful grouping. Render a single box
            # over all y values to show the statistical distribution.
            fig.add_trace(go.Box(y=y_vals, name=y_label or y, marker_color=_PALETTE[0],
                                 boxmean="sd",
                                 hovertemplate="%{y:,.2f}<extra></extra>"))
        else:
            fig.add_trace(go.Box(x=x_vals, y=y_vals, name=y, marker_color=_PALETTE[0]))

    elif chart_type == "waterfall":
        # Aggregate by x while preserving the data's original row order (which
        # matches SQL ORDER BY).  _aggregate sorts alphabetically, which breaks
        # chronological sequences like Jan'22, Feb'22, …
        _wf_seen: dict[str, float] = {}
        _wf_order: list[str] = []
        for _wf_row in data:
            _wf_k = _to_label(_wf_row.get(x, ""))
            _wf_v = 0.0
            try:
                _wf_v = float(_wf_row.get(y, 0) or 0)
            except (TypeError, ValueError):
                pass
            if _wf_k not in _wf_seen:
                _wf_seen[_wf_k] = 0.0
                _wf_order.append(_wf_k)
            _wf_seen[_wf_k] += _wf_v
        wf_x = _wf_order
        wf_y = [_wf_seen[k] for k in _wf_order]
        total = sum(wf_y)
        fig.add_trace(go.Waterfall(
            x=list(wf_x) + ["Total"],
            y=list(wf_y) + [total],
            measure=["relative"] * len(wf_y) + ["total"],
            name=y, hovertemplate=hover,
        ))
        effective_y_vals = [sum(v for v in wf_y if v > 0)]  # y_max covers running total

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

    elif chart_type in ("stacked_bar", "normalized_bar"):
        series = sorted(dict.fromkeys(_to_label(r.get(group, "")) for r in data), key=_sort_key)
        palette = _get_palette(len(series))
        # For normalized_bar pre-compute total per x so we can express each value as %
        if chart_type == "normalized_bar":
            x_totals: dict[str, float] = {}
            for row in data:
                xk = _to_label(row.get(x, ""))
                try:
                    x_totals[xk] = x_totals.get(xk, 0.0) + abs(float(row.get(y, 0) or 0))
                except (TypeError, ValueError):
                    pass
        all_stack_y: list[float] = []
        for i, grp in enumerate(series):
            subset = [r for r in data if _to_label(r.get(group, "")) == grp]
            gx, gy = _aggregate(subset, x, y, preserve_order=x in ("period", "__period__"))
            if chart_type == "normalized_bar":
                gy = [v / x_totals[xk] * 100 if x_totals.get(xk) else 0 for xk, v in zip(gx, gy)]
            all_stack_y.extend(gy)
            suffix = "%" if chart_type == "normalized_bar" else ""
            ht = f"{grp}<br>%{{x}}<br>%{{y:,.1f}}{suffix}<extra></extra>"
            fig.add_trace(go.Bar(x=gx, y=gy, name=str(grp), marker_color=palette[i], hovertemplate=ht))
        fig.update_layout(barmode="stack")
        effective_y_vals = [100] if chart_type == "normalized_bar" else all_stack_y

    elif chart_type == "bar-line":
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        bx, by = _aggregate(data, x, y, preserve_order=x in ("period", "__period__"))
        fig.add_trace(
            go.Bar(x=bx, y=by, name=y_label or y, marker_color=_PALETTE[0], hovertemplate=hover),
            secondary_y=False,
        )
        if y2:
            l2x, l2y = _aggregate(data, x, y2, preserve_order=x in ("period", "__period__"), mean=True)
            fig.add_trace(
                go.Scatter(x=l2x, y=l2y, mode="lines+markers", name=y2_label or y2,
                           line=dict(color=_PALETTE[3], width=2),
                           hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>"),
                secondary_y=True,
            )
            _y2_max = _compute_y_max(l2y) if l2y else 1.0
            fig.update_yaxes(title_text=y2_label or y2, secondary_y=True,
                             tickformat=",.2f", showgrid=False,
                             range=[0, _y2_max])
        effective_y_vals = by

    else:
        fig.add_trace(go.Table(
            header=dict(values=list(data[0].keys()) if data else []),
            cells=dict(values=[[row.get(k, "") for row in data] for k in (data[0].keys() if data else [])]),
        ))

    # ── Layout ────────────────────────────────────────────────────────────────
    _no_cartesian = {"treemap", "small_multiples", "donut"}
    show_legend = (
        chart_type in {"treemap", "grouped_bar", "stacked_bar", "normalized_bar", "donut", "bar-line"}
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
        elif x in ("period", "__period__"):
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
        _SCROLL_THRESHOLD = 40
        if chart_type == "bar" and len(set(all_x)) > _SCROLL_THRESHOLD:
            layout["width"] = max(1000, len(set(all_x)) * 30)
            layout["height"] = 480
        elif chart_type in ("grouped_bar", "stacked_bar", "normalized_bar"):
            _unique_x = len(set(all_x))
            if _unique_x > _SCROLL_THRESHOLD:
                _n_groups = len(set(_to_label(r.get(group, "")) for r in data)) if group else 1
                _px_per_group = max(30, _n_groups * 20) if chart_type == "grouped_bar" else 30
                layout["width"] = max(1000, _unique_x * _px_per_group)
                layout["height"] = 480

        layout["xaxis"] = dict(title=x_label or x, tickangle=-30, **xaxis_extra)

        # Y-axis: fixed 0–100 for normalized bar; auto-scaled for everything else.
        if chart_type == "normalized_bar":
            yaxis_cfg = dict(title="% of Total", tickformat=",.0f", showgrid=True,
                             gridwidth=1, range=[0, 100])
        else:
            _y_abs_max = max(abs(v) for v in effective_y_vals) if effective_y_vals else 1
            _tick_fmt = ",.2f" if _y_abs_max < 10 else (",.1f" if _y_abs_max < 100 else ",.0f")
            yaxis_cfg = dict(title=y_label or y, tickformat=_tick_fmt, showgrid=True, gridwidth=1)
            if chart_type != "histogram":
                y_min = min(effective_y_vals) if effective_y_vals else 0
                y_range = (
                    [y_min * 1.1, _compute_y_max(effective_y_vals)]
                    if y_min < 0
                    else [0, _compute_y_max(effective_y_vals)]
                )
                yaxis_cfg["range"] = y_range
        layout["yaxis"] = yaxis_cfg

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

    _CAT_LIMIT = 10
    for spec in specs[:3]:
        chart_type = spec.get("chart_type", "bar")
        if chart_type in seen_types:
            continue

        # Skip chart types where the group/facet dimension exceeds _CAT_LIMIT
        if chart_type in ("grouped_bar", "stacked_bar", "normalized_bar"):
            _grp_col = spec.get("group", "")
            if _grp_col and len({_to_label(r.get(_grp_col, "")) for r in sql_result}) > _CAT_LIMIT:
                continue
        if chart_type == "small_multiples":
            _fct_col = spec.get("facet", "")
            if _fct_col and len({_to_label(r.get(_fct_col, "")) for r in sql_result}) > _CAT_LIMIT:
                continue

        seen_types.add(chart_type)

        x_col = spec.get("x", "")
        y_col = spec.get("y", "")
        y2_col = spec.get("y2", "")
        group_col = spec.get("group", "")

        # Fallback if LLM picked a column that doesn't exist
        if x_col not in actual_cols:
            x_col = next((c for c in actual_cols if not isinstance(sql_result[0].get(c), (int, float))), actual_cols[0])
        if y_col not in actual_cols:
            y_col = actual_cols[-1]
        if y2_col and y2_col not in actual_cols:
            y2_col = ""

        # Auto-pivot wide → long for grouped/stacked/normalized bar when the
        # data has multiple numeric metric columns but no categorical group column.
        # E.g. {category, cogs, revenue} → {category, value, metric} so each
        # metric becomes its own series rather than only one being shown.
        _render_data = sql_result
        if chart_type in ("grouped_bar", "stacked_bar", "normalized_bar"):
            def _is_numeric_col(col):
                for r in sql_result:
                    v = r.get(col)
                    if v is None:
                        continue
                    try:
                        float(v)
                        return True
                    except (TypeError, ValueError):
                        return False
                return False

            _grp_missing = not group_col or group_col not in actual_cols
            _grp_same_as_x = group_col == x_col
            if _grp_missing or _grp_same_as_x:
                # Find all numeric columns except x_col, excluding ratio/percentage
                # derived columns that don't share the same unit as the primary metric.
                _ratio_keywords = (
                    "percent", "pct", "_rate", "rate_", "_ratio", "ratio_",
                    "_margin", "margin_", "_share", "share_", "_factor",
                )
                def _is_ratio_col(col: str) -> bool:
                    cl = col.lower()
                    return any(kw in cl for kw in _ratio_keywords)

                _numeric_cols = [
                    c for c in actual_cols
                    if c != x_col and _is_numeric_col(c) and not _is_ratio_col(c)
                ]
                # If the LLM-specified y_col is present and numeric, further restrict
                # to only columns whose magnitude is within 3 orders of the y_col median.
                if y_col and y_col in actual_cols and _is_numeric_col(y_col):
                    def _col_magnitude(col):
                        vals = [float(r[col]) for r in sql_result if r.get(col) is not None]
                        return max(abs(v) for v in vals) if vals else 0
                    _y_mag = _col_magnitude(y_col)
                    if _y_mag > 0:
                        _numeric_cols = [
                            c for c in _numeric_cols
                            if _col_magnitude(c) >= _y_mag / 1000
                        ]
                if len(_numeric_cols) >= 2:
                    _render_data = [
                        {x_col: r[x_col], "value": r[c], "metric": c}
                        for r in sql_result for c in _numeric_cols
                    ]
                    y_col = "value"
                    group_col = "metric"

        try:
            figure_json = generate_chart(
                _render_data, chart_type, x_col, y_col,
                spec.get("title", "Chart"),
                spec.get("x_label", ""), spec.get("y_label", ""),
                group_col, spec.get("facet", ""),
                y2_col, spec.get("y2_label", ""),
            )
            options.append({"figure_json": figure_json, "chart_type": chart_type, "title": spec.get("title", "Chart")})
        except Exception:
            continue

    if not options:
        return {"chart_spec": {}, "error": "Chart generation failed for all options."}

    return {"chart_spec": {"options": options}}
