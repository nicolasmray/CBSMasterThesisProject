"""Performance and latency benchmarking.

Tests cover:
  - Individual agent latency thresholds
  - Schema loading speed (cached vs uncached)
  - Semantic search latency
  - Full pipeline latency
  - Token usage estimation
"""

import time

import pytest

from datasets import PERF_THRESHOLDS


# ─── Schema loading ──────────────────────────────────────────────────────────

@pytest.mark.perf
@pytest.mark.unit
def test_schema_load_latency(timer):
    """Cached schema load must be under threshold."""
    from db.schema_snapshot import get_compact_schema

    # Prime the cache
    get_compact_schema()

    # Measure cached access
    with timer() as t:
        schema = get_compact_schema()

    assert schema, "Schema is empty"
    assert t.elapsed < PERF_THRESHOLDS["schema_load_max_seconds"], (
        f"Schema load took {t.elapsed:.3f}s "
        f"(threshold: {PERF_THRESHOLDS['schema_load_max_seconds']}s)"
    )


# ─── Semantic search latency ─────────────────────────────────────────────────

@pytest.mark.perf
@pytest.mark.integration
def test_semantic_search_latency(timer):
    """Semantic search must complete within threshold."""
    from db.vector_store import semantic_search

    with timer() as t:
        results = semantic_search("revenue calculation")

    print(f"\nSemantic search: {t.elapsed:.2f}s, {len(results)} chunks")

    assert t.elapsed < PERF_THRESHOLDS["semantic_search_max_seconds"], (
        f"Semantic search took {t.elapsed:.2f}s "
        f"(threshold: {PERF_THRESHOLDS['semantic_search_max_seconds']}s)"
    )


# ─── Orchestrator latency ────────────────────────────────────────────────────

@pytest.mark.perf
@pytest.mark.llm
def test_orchestrator_latency(timer):
    """Orchestrator must classify intent within threshold."""
    from agents.orchestrator import orchestrator_agent

    with timer() as t:
        result = orchestrator_agent({"user_query": "How many customers do we have?"})

    print(f"\nOrchestrator: {t.elapsed:.2f}s, intent={result.get('intent')}")

    assert t.elapsed < PERF_THRESHOLDS["orchestrator_max_seconds"], (
        f"Orchestrator took {t.elapsed:.2f}s "
        f"(threshold: {PERF_THRESHOLDS['orchestrator_max_seconds']}s)"
    )


# ─── SQL agent latency ───────────────────────────────────────────────────────

@pytest.mark.perf
@pytest.mark.llm
@pytest.mark.integration
def test_sql_agent_latency(timer):
    """SQL agent must generate and execute within threshold."""
    from agents.sql_agent import sql_agent

    with timer() as t:
        result = sql_agent({
            "user_query": "How many customers do we have?",
            "plan": "Count customers",
            "retry_count": 0,
        })

    rows = len(result.get("sql_result", []))
    error = result.get("error", "")
    print(f"\nSQL agent: {t.elapsed:.2f}s, {rows} rows, error={error!r}")

    assert t.elapsed < PERF_THRESHOLDS["sql_agent_max_seconds"], (
        f"SQL agent took {t.elapsed:.2f}s "
        f"(threshold: {PERF_THRESHOLDS['sql_agent_max_seconds']}s)"
    )


# ─── RAG agent latency ───────────────────────────────────────────────────────

@pytest.mark.perf
@pytest.mark.llm
@pytest.mark.integration
def test_rag_agent_latency(timer):
    """RAG agent must retrieve and rerank within threshold."""
    from agents.rag_agent import rag_agent

    with timer() as t:
        result = rag_agent({"user_query": "What is the PII policy?"})

    chunks = len(result.get("rag_chunks", []))
    print(f"\nRAG agent: {t.elapsed:.2f}s, {chunks} filtered chunks")

    assert t.elapsed < PERF_THRESHOLDS["rag_agent_max_seconds"], (
        f"RAG agent took {t.elapsed:.2f}s "
        f"(threshold: {PERF_THRESHOLDS['rag_agent_max_seconds']}s)"
    )


# ─── Full pipeline latency ───────────────────────────────────────────────────

@pytest.mark.perf
@pytest.mark.llm
@pytest.mark.integration
@pytest.mark.slow
def test_e2e_pipeline_latency(timer):
    """Full pipeline must complete within threshold."""
    from graph import compiled_graph

    with timer() as t:
        result = compiled_graph.invoke({"user_query": "How many customers do we have?"})

    answer_len = len(result.get("final_answer", ""))
    print(f"\nFull pipeline: {t.elapsed:.2f}s, answer length={answer_len}")

    assert t.elapsed < PERF_THRESHOLDS["e2e_max_seconds"], (
        f"Full pipeline took {t.elapsed:.2f}s "
        f"(threshold: {PERF_THRESHOLDS['e2e_max_seconds']}s)"
    )


# ─── Latency breakdown (informational) ───────────────────────────────────────

@pytest.mark.perf
@pytest.mark.llm
@pytest.mark.integration
@pytest.mark.slow
def test_latency_breakdown():
    """Measure and report latency for each pipeline stage."""
    timings = {}

    # Orchestrator
    from agents.orchestrator import orchestrator_agent
    start = time.perf_counter()
    orch_result = orchestrator_agent({"user_query": "Total revenue for 2024"})
    timings["orchestrator"] = time.perf_counter() - start

    # SQL agent
    from agents.sql_agent import sql_agent
    start = time.perf_counter()
    sql_result = sql_agent({
        "user_query": "Total revenue for 2024",
        "plan": orch_result.get("plan", ""),
        "retry_count": 0,
    })
    timings["sql_agent"] = time.perf_counter() - start

    # Response agent
    from agents.response_agent import response_agent
    state = {
        "user_query": "Total revenue for 2024",
        **orch_result,
        **sql_result,
        "rag_context": "",
        "rag_chunks": [],
        "chart_spec": {},
    }
    start = time.perf_counter()
    resp_result = response_agent(state)
    timings["response_agent"] = time.perf_counter() - start

    total = sum(timings.values())
    timings["total"] = total

    print("\n┌─── Latency Breakdown ───────────────────")
    for stage, elapsed in timings.items():
        pct = (elapsed / total * 100) if total > 0 else 0
        print(f"│ {stage:20s}: {elapsed:6.2f}s ({pct:5.1f}%)")
    print("└─────────────────────────────────────────")

    # All stages should complete
    assert resp_result.get("final_answer"), "Pipeline produced no answer"
