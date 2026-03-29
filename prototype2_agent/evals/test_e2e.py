"""End-to-end pipeline evaluation — full user flows through the compiled graph.

Tests cover:
  - SQL flow: query → data → formatted answer
  - RAG flow: query → retrieval → synthesized answer
  - Chart flow: query → SQL → chart generation
  - Hybrid flow: query → SQL + RAG → combined answer
  - Error resilience: bad queries don't crash the pipeline
  - DeepEval overall quality assessment
"""

import pytest

from datasets import E2E_CASES


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _run_pipeline(query: str) -> dict:
    """Run the full compiled graph and return the result state."""
    from graph import compiled_graph
    return compiled_graph.invoke({"user_query": query})


# ─── Intent routing correctness (e2e) ────────────────────────────────────────

@pytest.mark.e2e
@pytest.mark.llm
@pytest.mark.integration
@pytest.mark.parametrize(
    "case",
    E2E_CASES,
    ids=[c["description"] for c in E2E_CASES],
)
def test_e2e_intent_routing(case):
    """Pipeline routes to the correct agent path."""
    result = _run_pipeline(case["query"])
    actual_intent = result.get("intent", "")

    if "expected_intent" in case:
        assert actual_intent == case["expected_intent"], (
            f"Expected intent '{case['expected_intent']}', got '{actual_intent}'.\n"
            f"Query: {case['query']!r}"
        )


# ─── Pipeline produces expected outputs ───────────────────────────────────────

@pytest.mark.e2e
@pytest.mark.llm
@pytest.mark.integration
@pytest.mark.parametrize(
    "case",
    E2E_CASES,
    ids=[c["description"] for c in E2E_CASES],
)
def test_e2e_produces_final_answer(case):
    """Every pipeline run must produce a non-empty final_answer."""
    result = _run_pipeline(case["query"])
    answer = result.get("final_answer", "")

    assert answer and len(answer) > 20, (
        f"Pipeline produced empty/trivial answer.\n"
        f"Query: {case['query']!r}\n"
        f"Answer: {answer!r}"
    )


@pytest.mark.e2e
@pytest.mark.llm
@pytest.mark.integration
@pytest.mark.parametrize(
    "case",
    E2E_CASES,
    ids=[c["description"] for c in E2E_CASES],
)
def test_e2e_expected_artifacts(case):
    """Pipeline must produce the expected artifacts (SQL, RAG, chart)."""
    result = _run_pipeline(case["query"])

    if case.get("should_have_sql"):
        assert result.get("sql_query"), (
            f"Expected SQL query but got none.\nQuery: {case['query']!r}"
        )

    if case.get("should_have_rag"):
        rag_chunks = result.get("rag_chunks", [])
        rag_context = result.get("rag_context", "")
        assert rag_chunks or rag_context, (
            f"Expected RAG context but got none.\nQuery: {case['query']!r}"
        )

    if case.get("should_have_chart"):
        chart_spec = result.get("chart_spec", {})
        assert chart_spec and chart_spec.get("options"), (
            f"Expected chart but got none.\nQuery: {case['query']!r}"
        )


@pytest.mark.e2e
@pytest.mark.llm
@pytest.mark.integration
@pytest.mark.parametrize(
    "case",
    [c for c in E2E_CASES if c.get("answer_must_contain_number")],
    ids=[c["description"] for c in E2E_CASES if c.get("answer_must_contain_number")],
)
def test_e2e_answer_contains_numbers(case):
    """SQL-based answers must contain actual numeric data."""
    import re

    result = _run_pipeline(case["query"])
    answer = result.get("final_answer", "")

    numbers = re.findall(r"\d[\d,]*\.?\d*", answer)
    assert numbers, (
        f"Answer contains no numbers for a data query.\n"
        f"Query: {case['query']!r}\n"
        f"Answer: {answer[:300]}"
    )


@pytest.mark.e2e
@pytest.mark.llm
@pytest.mark.integration
@pytest.mark.parametrize(
    "case",
    [c for c in E2E_CASES if "answer_keywords" in c],
    ids=[c["description"] for c in E2E_CASES if "answer_keywords" in c],
)
def test_e2e_answer_contains_keywords(case):
    """RAG-based answers must contain expected keywords."""
    result = _run_pipeline(case["query"])
    answer = result.get("final_answer", "").lower()

    for kw in case["answer_keywords"]:
        assert kw.lower() in answer, (
            f"Expected keyword '{kw}' not in answer.\n"
            f"Query: {case['query']!r}\n"
            f"Answer: {answer[:300]}"
        )


# ─── Error resilience ────────────────────────────────────────────────────────

@pytest.mark.e2e
@pytest.mark.llm
def test_e2e_handles_impossible_query():
    """Pipeline must not crash on impossible queries."""
    result = _run_pipeline("What is the quantum spin of product ID 42?")

    assert "final_answer" in result, "Pipeline crashed — no final_answer"
    answer = result["final_answer"]
    assert len(answer) > 10, "Pipeline returned empty answer for edge case"


@pytest.mark.e2e
@pytest.mark.llm
def test_e2e_handles_empty_query():
    """Pipeline must handle empty/trivial queries gracefully."""
    result = _run_pipeline("hello")

    assert "final_answer" in result, "Pipeline crashed on trivial input"


# ─── DeepEval: overall answer quality ─────────────────────────────────────────

@pytest.mark.e2e
@pytest.mark.llm
@pytest.mark.integration
@pytest.mark.parametrize("case", E2E_CASES[:3],
                         ids=[f"quality_{c['description']}" for c in E2E_CASES[:3]])
def test_e2e_deepeval_quality(case):
    """DeepEval quality assessment of full pipeline output."""
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import GEval
    from groq_judge import get_judge_model
    from score_recorder import record_and_assert

    metric = GEval(
        name="End-to-End Answer Quality",
        criteria=(
            "The answer is helpful, accurate, and addresses the user's question. "
            "For data questions, it includes actual numbers from the database. "
            "For policy questions, it references the correct documents. "
            "The answer is clear and suitable for a business audience."
        ),
        evaluation_params=["input", "actual_output"],
        threshold=0.6,
        model=get_judge_model(),
    )

    result = _run_pipeline(case["query"])

    test_case = LLMTestCase(
        input=case["query"],
        actual_output=result.get("final_answer", "No answer produced"),
    )
    record_and_assert(test_case, [metric], test_name=f"e2e_quality_{case['description']}")
