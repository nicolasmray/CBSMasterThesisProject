"""Orchestrator agent evaluation — intent classification accuracy.

Tests that the orchestrator correctly routes user queries to the right
agent (sql, rag, chart, hybrid). Uses both pytest parametrize and
DeepEval for structured LLM-graded evaluation.
"""

import pytest

from datasets import ORCHESTRATOR_CASES


# ─── Pytest parametrized: exact intent match ──────────────────────────────────

@pytest.mark.llm
@pytest.mark.parametrize(
    "case",
    ORCHESTRATOR_CASES,
    ids=[f"{c['expected_intent']}_{i}" for i, c in enumerate(ORCHESTRATOR_CASES)],
)
def test_orchestrator_intent_classification(case, llm_orchestrator):
    """Verify the orchestrator classifies intent correctly."""
    from agents.orchestrator import orchestrator_agent

    state = orchestrator_agent({"user_query": case["query"]})

    actual_intent = state.get("intent", "")
    expected_intent = case["expected_intent"]

    assert actual_intent == expected_intent, (
        f"Query: {case['query']!r}\n"
        f"Expected intent: {expected_intent!r}, got: {actual_intent!r}\n"
        f"Plan: {state.get('plan', '')!r}"
    )


@pytest.mark.llm
@pytest.mark.parametrize(
    "case",
    ORCHESTRATOR_CASES,
    ids=[f"{c['expected_intent']}_{i}" for i, c in enumerate(ORCHESTRATOR_CASES)],
)
def test_orchestrator_produces_plan(case):
    """Verify the orchestrator always produces a non-empty plan."""
    from agents.orchestrator import orchestrator_agent

    state = orchestrator_agent({"user_query": case["query"]})
    plan = state.get("plan", "")

    assert plan and len(plan) > 10, (
        f"Orchestrator produced empty or trivial plan for: {case['query']!r}"
    )


# ─── DeepEval: LLM-graded routing quality ────────────────────────────────────

@pytest.mark.llm
@pytest.mark.parametrize("case", ORCHESTRATOR_CASES[:6],
                         ids=[f"deepeval_{i}" for i in range(6)])
def test_orchestrator_deepeval_routing(case):
    """DeepEval LLM-graded assessment of routing quality."""
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import GEval
    from groq_judge import get_judge_model
    from score_recorder import record_and_assert
    from agents.orchestrator import orchestrator_agent

    metric = GEval(
        name="Routing Correctness",
        criteria=(
            "The AI assistant correctly classified the user query intent. "
            "SQL queries about data/metrics/counts should be 'sql'. "
            "Questions about policies/definitions/documents should be 'rag'. "
            "Requests for charts/visualizations should be 'chart'. "
            "Questions needing both data and documents should be 'hybrid'."
        ),
        evaluation_params=["input", "actual_output", "expected_output"],
        threshold=0.7,
        model=get_judge_model(),
    )

    state = orchestrator_agent({"user_query": case["query"]})

    test_case = LLMTestCase(
        input=case["query"],
        actual_output=f"intent={state.get('intent')}, plan={state.get('plan')}",
        expected_output=f"intent={case['expected_intent']}",
    )
    record_and_assert(test_case, [metric], test_name=f"orchestrator_routing_{case['expected_intent']}")


# ─── Aggregate accuracy metric ───────────────────────────────────────────────

@pytest.mark.llm
def test_orchestrator_accuracy_above_threshold():
    """Overall orchestrator accuracy must be >= 80%."""
    from agents.orchestrator import orchestrator_agent

    correct = 0
    for case in ORCHESTRATOR_CASES:
        state = orchestrator_agent({"user_query": case["query"]})
        if state.get("intent") == case["expected_intent"]:
            correct += 1

    accuracy = correct / len(ORCHESTRATOR_CASES)
    print(f"\nOrchestrator accuracy: {correct}/{len(ORCHESTRATOR_CASES)} = {accuracy:.1%}")

    assert accuracy >= 0.80, (
        f"Orchestrator accuracy {accuracy:.1%} is below 80% threshold. "
        f"Got {correct}/{len(ORCHESTRATOR_CASES)} correct."
    )
