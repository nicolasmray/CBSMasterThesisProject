"""SQL agent evaluation — query generation, validation, and execution correctness.

Tests cover:
  - SQL syntax validity (sqlglot parsing)
  - Correct table/column references (schema compliance)
  - Forbidden pattern avoidance (CURRENT_DATE, TO_CHAR, etc.)
  - Result correctness (non-empty results, value checks)
  - Revenue formula correctness
  - DeepEval SQL quality assessment
"""

import re

import pytest

from datasets import SQL_AGENT_CASES


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _run_sql_agent(query: str) -> dict:
    """Run the SQL agent and return the full state update."""
    from agents.sql_agent import sql_agent
    return sql_agent({"user_query": query, "plan": "", "retry_count": 0})


def _normalize_sql(sql: str) -> str:
    """Lowercase and collapse whitespace for pattern matching."""
    return re.sub(r"\s+", " ", sql.lower().strip())


# ─── SQL generation & syntax validation ───────────────────────────────────────

@pytest.mark.sql
@pytest.mark.llm
@pytest.mark.parametrize(
    "case",
    SQL_AGENT_CASES,
    ids=[c["description"] for c in SQL_AGENT_CASES],
)
def test_sql_generates_valid_query(case):
    """SQL agent must produce a parseable SQL query."""
    import sqlglot

    result = _run_sql_agent(case["query"])
    sql = result.get("sql_query", "")

    assert sql, f"SQL agent returned empty query for: {case['query']!r}"

    # Must parse without error
    try:
        sqlglot.parse_one(sql, dialect="postgres")
    except sqlglot.errors.ParseError as e:
        pytest.fail(f"Invalid SQL syntax: {e}\nQuery: {sql}")


@pytest.mark.sql
@pytest.mark.llm
@pytest.mark.parametrize(
    "case",
    SQL_AGENT_CASES,
    ids=[c["description"] for c in SQL_AGENT_CASES],
)
def test_sql_references_correct_tables(case):
    """Generated SQL must reference the expected tables."""
    if "expected_tables" not in case:
        pytest.skip("No expected_tables defined")

    result = _run_sql_agent(case["query"])
    sql_lower = _normalize_sql(result.get("sql_query", ""))

    for table in case["expected_tables"]:
        assert table.lower() in sql_lower, (
            f"Expected table '{table}' not found in SQL.\n"
            f"Query: {case['query']!r}\n"
            f"SQL: {result.get('sql_query', '')}"
        )


@pytest.mark.sql
@pytest.mark.llm
@pytest.mark.parametrize(
    "case",
    SQL_AGENT_CASES,
    ids=[c["description"] for c in SQL_AGENT_CASES],
)
def test_sql_references_correct_columns(case):
    """Generated SQL must reference expected columns."""
    if "expected_columns" not in case:
        pytest.skip("No expected_columns defined")

    result = _run_sql_agent(case["query"])
    sql_lower = _normalize_sql(result.get("sql_query", ""))

    for col in case["expected_columns"]:
        assert col.lower() in sql_lower, (
            f"Expected column '{col}' not found in SQL.\n"
            f"Query: {case['query']!r}\n"
            f"SQL: {result.get('sql_query', '')}"
        )


@pytest.mark.sql
@pytest.mark.llm
@pytest.mark.parametrize(
    "case",
    SQL_AGENT_CASES,
    ids=[c["description"] for c in SQL_AGENT_CASES],
)
def test_sql_avoids_forbidden_patterns(case):
    """Generated SQL must NOT contain forbidden patterns."""
    if "forbidden_patterns" not in case:
        pytest.skip("No forbidden_patterns defined")

    result = _run_sql_agent(case["query"])
    sql_upper = result.get("sql_query", "").upper()

    for pattern in case["forbidden_patterns"]:
        assert pattern.upper() not in sql_upper, (
            f"Forbidden pattern '{pattern}' found in SQL.\n"
            f"Query: {case['query']!r}\n"
            f"SQL: {result.get('sql_query', '')}"
        )


# ─── SQL execution & result correctness ──────────────────────────────────────

@pytest.mark.sql
@pytest.mark.llm
@pytest.mark.integration
@pytest.mark.parametrize(
    "case",
    SQL_AGENT_CASES,
    ids=[c["description"] for c in SQL_AGENT_CASES],
)
def test_sql_returns_results(case):
    """SQL agent must return non-empty results (or meet min_rows)."""
    result = _run_sql_agent(case["query"])
    sql_result = result.get("sql_result", [])
    error = result.get("error", "")
    min_rows = case.get("min_rows", 1)

    assert not error, (
        f"SQL agent returned error: {error}\n"
        f"Query: {case['query']!r}\n"
        f"SQL: {result.get('sql_query', '')}"
    )
    assert len(sql_result) >= min_rows, (
        f"Expected >= {min_rows} rows, got {len(sql_result)}.\n"
        f"Query: {case['query']!r}\n"
        f"SQL: {result.get('sql_query', '')}"
    )


@pytest.mark.sql
@pytest.mark.llm
@pytest.mark.integration
@pytest.mark.parametrize(
    "case",
    [c for c in SQL_AGENT_CASES if "result_check" in c],
    ids=[c["description"] for c in SQL_AGENT_CASES if "result_check" in c],
)
def test_sql_result_custom_validation(case):
    """Custom result validation callable."""
    result = _run_sql_agent(case["query"])
    sql_result = result.get("sql_result", [])

    assert case["result_check"](sql_result), (
        f"Custom result check failed.\n"
        f"Query: {case['query']!r}\n"
        f"Result: {sql_result[:5]}"
    )


# ─── Revenue formula correctness ─────────────────────────────────────────────

@pytest.mark.sql
@pytest.mark.llm
def test_sql_revenue_formula():
    """Revenue queries must use unitprice * orderqty * (1 - unitpricediscount)."""
    result = _run_sql_agent("What is the total revenue for 2024?")
    sql_lower = _normalize_sql(result.get("sql_query", ""))

    # Must contain all three components of the revenue formula
    assert "unitprice" in sql_lower, "Missing 'unitprice' in revenue query"
    assert "orderqty" in sql_lower, "Missing 'orderqty' in revenue query"
    assert "unitpricediscount" in sql_lower, "Missing 'unitpricediscount' in revenue query"

    # Must NOT use a non-existent linetotal column
    assert "linetotal" not in sql_lower, (
        "SQL uses 'linetotal' which does not exist in salesorderdetail"
    )


@pytest.mark.sql
@pytest.mark.llm
def test_sql_time_anchor_uses_max_date():
    """Relative time queries must anchor to MAX(date), not CURRENT_DATE/NOW()."""
    result = _run_sql_agent("Show me the last 6 months of sales")
    sql_upper = result.get("sql_query", "").upper()

    assert "CURRENT_DATE" not in sql_upper and "NOW()" not in sql_upper, (
        "SQL uses CURRENT_DATE/NOW() instead of anchoring to MAX(orderdate)"
    )
    assert "MAX" in sql_upper, (
        "SQL does not anchor to MAX(date) for relative time range"
    )


# ─── SQL agent retry behaviour ───────────────────────────────────────────────

@pytest.mark.sql
@pytest.mark.llm
def test_sql_agent_handles_bad_query_gracefully():
    """SQL agent should not crash on an impossible query; it should return an error."""
    result = _run_sql_agent(
        "Show me the quantum entanglement coefficient of each product"
    )
    # It should either produce a result or a clean error, not crash
    assert "sql_query" in result or "error" in result


# ─── DeepEval: LLM-graded SQL quality ────────────────────────────────────────

@pytest.mark.sql
@pytest.mark.llm
@pytest.mark.parametrize("case", SQL_AGENT_CASES[:5],
                         ids=[f"deepeval_{c['description']}" for c in SQL_AGENT_CASES[:5]])
def test_sql_deepeval_quality(case):
    """DeepEval LLM-graded assessment of SQL generation quality."""
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import GEval
    from groq_judge import get_judge_model
    from score_recorder import record_and_assert

    metric = GEval(
        name="SQL Query Quality",
        criteria=(
            "The generated SQL query correctly answers the user's question. "
            "It uses fully-qualified table names (schema.table), correct JOINs, "
            "appropriate aggregation, and valid PostgreSQL syntax. "
            "It derives revenue from unitprice * orderqty * (1 - unitpricediscount) "
            "and anchors relative dates to MAX(date) instead of CURRENT_DATE."
        ),
        evaluation_params=["input", "actual_output"],
        threshold=0.6,
        model=get_judge_model(),
    )

    result = _run_sql_agent(case["query"])

    test_case = LLMTestCase(
        input=case["query"],
        actual_output=(
            f"SQL: {result.get('sql_query', 'NONE')}\n"
            f"Rows returned: {len(result.get('sql_result', []))}\n"
            f"Error: {result.get('error', 'none')}"
        ),
    )
    record_and_assert(test_case, [metric], test_name=f"sql_quality_{case['description']}")


# ─── Aggregate pass rate ─────────────────────────────────────────────────────

@pytest.mark.sql
@pytest.mark.llm
@pytest.mark.integration
def test_sql_agent_overall_success_rate():
    """At least 70% of SQL test cases must execute without errors."""
    successes = 0
    for case in SQL_AGENT_CASES:
        result = _run_sql_agent(case["query"])
        if not result.get("error") and result.get("sql_result"):
            successes += 1

    rate = successes / len(SQL_AGENT_CASES)
    print(f"\nSQL agent success rate: {successes}/{len(SQL_AGENT_CASES)} = {rate:.1%}")

    assert rate >= 0.70, (
        f"SQL agent success rate {rate:.1%} is below 70% threshold. "
        f"Got {successes}/{len(SQL_AGENT_CASES)} successful."
    )
