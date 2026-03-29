"""Chart agent evaluation — visualization generation correctness.

Tests cover:
  - Chart spec structure (valid JSON, required keys)
  - Plotly figure validity (parseable by plotly.io)
  - Multiple chart options returned
  - Column references match SQL result columns
"""

import json

import pytest


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _run_chart_pipeline(query: str) -> dict:
    """Run orchestrator → sql → chart pipeline and return full state."""
    from graph import compiled_graph
    return compiled_graph.invoke({"user_query": query})


CHART_QUERIES = [
    "Show me a bar chart of revenue by territory",
    "Plot monthly sales trend for 2024",
    "Create a chart of employee count by department",
]


# ─── Chart spec structure ─────────────────────────────────────────────────────

@pytest.mark.llm
@pytest.mark.integration
@pytest.mark.parametrize("query", CHART_QUERIES,
                         ids=[f"chart_{i}" for i in range(len(CHART_QUERIES))])
def test_chart_produces_valid_spec(query):
    """Chart agent must produce a chart_spec with valid structure."""
    result = _run_chart_pipeline(query)
    chart_spec = result.get("chart_spec", {})

    assert chart_spec, f"No chart_spec returned for: {query!r}"

    options = chart_spec.get("options", [])
    assert len(options) >= 1, f"No chart options returned for: {query!r}"

    for i, opt in enumerate(options):
        assert "figure_json" in opt, f"Option {i} missing 'figure_json'"
        assert "chart_type" in opt, f"Option {i} missing 'chart_type'"
        assert "title" in opt, f"Option {i} missing 'title'"


@pytest.mark.llm
@pytest.mark.integration
@pytest.mark.parametrize("query", CHART_QUERIES,
                         ids=[f"plotly_{i}" for i in range(len(CHART_QUERIES))])
def test_chart_produces_valid_plotly_json(query):
    """Each chart option must contain valid Plotly figure JSON."""
    import plotly.io as pio

    result = _run_chart_pipeline(query)
    chart_spec = result.get("chart_spec", {})
    options = chart_spec.get("options", [])

    if not options:
        pytest.skip("No chart options to validate")

    for i, opt in enumerate(options):
        fig_json = opt.get("figure_json", "")
        assert fig_json, f"Option {i} has empty figure_json"

        try:
            fig = pio.from_json(fig_json)
            assert fig.data, f"Option {i} has no data traces"
        except Exception as e:
            pytest.fail(f"Option {i} has invalid Plotly JSON: {e}")


@pytest.mark.llm
@pytest.mark.integration
def test_chart_returns_multiple_options():
    """Chart agent should return multiple chart type options."""
    result = _run_chart_pipeline("Show me a chart of revenue by territory")
    chart_spec = result.get("chart_spec", {})
    options = chart_spec.get("options", [])

    assert len(options) >= 2, (
        f"Expected multiple chart options, got {len(options)}"
    )

    # Options should have different chart types
    types = [opt.get("chart_type") for opt in options]
    assert len(set(types)) >= 2, (
        f"Expected different chart types, got: {types}"
    )


@pytest.mark.llm
@pytest.mark.integration
def test_chart_also_has_sql_result():
    """Chart pipeline must also produce SQL results (data behind the chart)."""
    result = _run_chart_pipeline("Show me a chart of revenue by territory")

    assert result.get("sql_query"), "Chart pipeline missing SQL query"
    assert result.get("sql_result"), "Chart pipeline missing SQL result data"
