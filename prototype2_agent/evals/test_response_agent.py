"""Response agent evaluation — anti-hallucination and answer quality.

Tests cover:
  - SQL path: numbers in answer come from data, not LLM invention
  - RAG path: answer grounded in retrieved context
  - Error path: errors are reported clearly
  - Empty result path: handled gracefully
  - DeepEval hallucination detection
"""

import re

import pytest


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _run_response_agent(state: dict) -> dict:
    """Run the response agent with a pre-populated state."""
    from agents.response_agent import response_agent
    return response_agent(state)


# ─── Path A: SQL results — anti-hallucination ────────────────────────────────

@pytest.mark.llm
def test_response_sql_path_contains_data():
    """Response for SQL results must include the actual data values."""
    state = {
        "user_query": "How many customers do we have?",
        "sql_query": "SELECT COUNT(*) as total FROM sales.customer",
        "sql_result": [{"total": 19820}],
        "rag_context": "",
        "rag_chunks": [],
        "chart_spec": {},
        "error": "",
    }
    result = _run_response_agent(state)
    answer = result.get("final_answer", "")

    # The exact number must appear (formatted as 19,820 or 19820)
    assert "19" in answer and "820" in answer, (
        f"Answer does not contain the actual count (19820).\n"
        f"Answer: {answer[:300]}"
    )


@pytest.mark.llm
def test_response_sql_path_multirow():
    """Response for multi-row SQL results must include a table or list."""
    state = {
        "user_query": "Revenue by territory",
        "sql_query": "SELECT territory, revenue FROM ...",
        "sql_result": [
            {"territory": "Northwest", "revenue": 5000000.50},
            {"territory": "Northeast", "revenue": 3500000.25},
            {"territory": "Southwest", "revenue": 2100000.75},
        ],
        "rag_context": "",
        "rag_chunks": [],
        "chart_spec": {},
        "error": "",
    }
    result = _run_response_agent(state)
    answer = result.get("final_answer", "")

    # Must contain territory names from the data
    assert "Northwest" in answer, "Missing 'Northwest' from SQL result"
    assert "Northeast" in answer, "Missing 'Northeast' from SQL result"
    assert "3 row" in answer.lower(), "Missing row count"


@pytest.mark.llm
def test_response_does_not_invent_numbers():
    """LLM interpretation must not contain specific numbers from the result."""
    state = {
        "user_query": "Total revenue for 2024",
        "sql_query": "SELECT SUM(...) as total_revenue",
        "sql_result": [{"total_revenue": 45678912.34}],
        "rag_context": "",
        "rag_chunks": [],
        "chart_spec": {},
        "error": "",
    }
    result = _run_response_agent(state)
    answer = result.get("final_answer", "")

    # Split at "Key Insights" — the LLM interpretation part
    parts = answer.split("Key Insights:")
    if len(parts) >= 2:
        insight = parts[1]
        # The LLM insight should NOT contain the exact number
        assert "45678912" not in insight.replace(",", ""), (
            "LLM interpretation restated the exact number — "
            "anti-hallucination design violated"
        )


# ─── Path B: Empty results ───────────────────────────────────────────────────

@pytest.mark.unit
def test_response_empty_result_reports_clearly():
    """Empty SQL result must be reported clearly with the query shown."""
    state = {
        "user_query": "Revenue for product XYZ123",
        "sql_query": "SELECT * FROM sales WHERE product = 'XYZ123'",
        "sql_result": [],
        "rag_context": "",
        "rag_chunks": [],
        "chart_spec": {},
        "error": "",
    }
    result = _run_response_agent(state)
    answer = result.get("final_answer", "")

    assert "no result" in answer.lower() or "no rows" in answer.lower(), (
        f"Empty result not clearly reported.\nAnswer: {answer[:300]}"
    )
    assert "XYZ123" in answer or "sql" in answer.lower(), (
        "Answer does not show the query context for debugging"
    )


# ─── Path C: Error handling ──────────────────────────────────────────────────

@pytest.mark.unit
def test_response_error_path():
    """Errors must be surfaced clearly to the user."""
    state = {
        "user_query": "Some query",
        "sql_query": "SELECT invalid_col FROM nonexistent_table",
        "sql_result": [],
        "rag_context": "",
        "rag_chunks": [],
        "chart_spec": {},
        "error": "relation 'nonexistent_table' does not exist",
    }
    result = _run_response_agent(state)
    answer = result.get("final_answer", "")

    assert "error" in answer.lower() or "could not" in answer.lower(), (
        f"Error not surfaced in answer.\nAnswer: {answer[:300]}"
    )


# ─── Path D: RAG-only ────────────────────────────────────────────────────────

@pytest.mark.llm
def test_response_rag_path():
    """RAG-only path must produce a meaningful answer from chunks."""
    state = {
        "user_query": "What is the PII policy?",
        "sql_query": "",
        "sql_result": [],
        "rag_context": "PII fields include passwordhash and nationalidnumber.",
        "rag_chunks": [
            {"content": "PII fields include passwordhash and nationalidnumber. Never expose these.", "score": 0.85, "source": "company_policies.txt"},
        ],
        "chart_spec": {},
        "error": "",
    }
    result = _run_response_agent(state)
    answer = result.get("final_answer", "")

    assert answer and len(answer) > 20, "RAG path produced empty answer"
    # Should reference the actual content
    assert "pii" in answer.lower() or "password" in answer.lower(), (
        f"RAG answer doesn't reference the provided context.\nAnswer: {answer[:300]}"
    )


# ─── Path E: Nothing available ───────────────────────────────────────────────

@pytest.mark.unit
def test_response_nothing_available():
    """When nothing is available, a clear fallback message is shown."""
    state = {
        "user_query": "Some query",
        "sql_query": "",
        "sql_result": [],
        "rag_context": "",
        "rag_chunks": [],
        "chart_spec": {},
        "error": "",
    }
    result = _run_response_agent(state)
    answer = result.get("final_answer", "")

    assert "unable" in answer.lower() or "no data" in answer.lower(), (
        f"No fallback message shown.\nAnswer: {answer[:300]}"
    )


# ─── DeepEval: hallucination detection ───────────────────────────────────────

@pytest.mark.llm
def test_response_no_hallucination_deepeval():
    """DeepEval hallucination metric on a synthetic SQL result."""
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import HallucinationMetric
    from groq_judge import get_judge_model
    from score_recorder import record_and_assert

    state = {
        "user_query": "How many employees are there?",
        "sql_query": "SELECT COUNT(*) as headcount FROM humanresources.employee WHERE currentflag = true",
        "sql_result": [{"headcount": 290}],
        "rag_context": "",
        "rag_chunks": [],
        "chart_spec": {},
        "error": "",
    }
    result = _run_response_agent(state)

    test_case = LLMTestCase(
        input="How many employees are there?",
        actual_output=result.get("final_answer", ""),
        context=["The database has 290 active employees (currentflag = true)."],
    )
    record_and_assert(test_case, [HallucinationMetric(threshold=0.5, model=get_judge_model())],
                      test_name="response_hallucination_check")


@pytest.mark.llm
def test_response_quality_deepeval():
    """DeepEval general quality assessment of the response agent."""
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import GEval
    from groq_judge import get_judge_model
    from score_recorder import record_and_assert

    metric = GEval(
        name="Answer Quality",
        criteria=(
            "The answer is clear, accurate, and helpful for a business user. "
            "It shows actual data values and provides useful interpretation. "
            "Numbers are not fabricated — they match the SQL result."
        ),
        evaluation_params=["input", "actual_output"],
        threshold=0.6,
        model=get_judge_model(),
    )

    state = {
        "user_query": "Top 3 products by revenue",
        "sql_query": "SELECT name, SUM(...) as revenue FROM ...",
        "sql_result": [
            {"name": "Mountain-200 Black", "revenue": 4400592.80},
            {"name": "Road-250 Red", "revenue": 3578270.50},
            {"name": "Touring-1000 Blue", "revenue": 2451318.90},
        ],
        "rag_context": "",
        "rag_chunks": [],
        "chart_spec": {},
        "error": "",
    }
    result = _run_response_agent(state)

    test_case = LLMTestCase(
        input="Top 3 products by revenue",
        actual_output=result.get("final_answer", ""),
    )
    record_and_assert(test_case, [metric], test_name="response_answer_quality")
