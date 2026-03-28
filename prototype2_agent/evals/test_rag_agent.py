"""RAG agent evaluation — retrieval quality, faithfulness, and relevancy.

Tests cover:
  - Retrieval precision: correct chunks are returned for known queries
  - Source accuracy: chunks come from the expected source file
  - Similarity threshold: irrelevant queries return few/no chunks
  - Reranker effectiveness: noise filtering works
  - DeepEval RAGAS-style metrics: faithfulness, answer relevancy, contextual recall
"""

import pytest

from datasets import RAG_RETRIEVAL_CASES


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _search(query: str) -> list[dict]:
    """Run semantic_search directly (no MCP)."""
    from mcp_server.tools.rag_tools import semantic_search
    return semantic_search(query)


def _run_rag_agent(query: str) -> dict:
    """Run the full RAG agent (retrieval + reranker)."""
    from agents.rag_agent import rag_agent
    return rag_agent({"user_query": query})


# ─── Retrieval precision tests ────────────────────────────────────────────────

@pytest.mark.rag
@pytest.mark.integration
@pytest.mark.parametrize(
    "case",
    RAG_RETRIEVAL_CASES,
    ids=[c["description"] for c in RAG_RETRIEVAL_CASES],
)
def test_retrieval_returns_chunks(case):
    """Semantic search must return at least min_chunks for known queries."""
    chunks = _search(case["query"])
    min_chunks = case.get("min_chunks", 1)

    assert len(chunks) >= min_chunks, (
        f"Expected >= {min_chunks} chunks, got {len(chunks)}.\n"
        f"Query: {case['query']!r}"
    )


@pytest.mark.rag
@pytest.mark.integration
@pytest.mark.parametrize(
    "case",
    RAG_RETRIEVAL_CASES,
    ids=[c["description"] for c in RAG_RETRIEVAL_CASES],
)
def test_retrieval_content_relevance(case):
    """Top retrieved chunks must contain expected keywords."""
    chunks = _search(case["query"])
    if not chunks:
        pytest.fail(f"No chunks returned for: {case['query']!r}")

    # Combine all chunk content for checking
    all_content = " ".join(c["content"].lower() for c in chunks)

    for keyword in case["must_contain"]:
        assert keyword.lower() in all_content, (
            f"Keyword '{keyword}' not found in any retrieved chunk.\n"
            f"Query: {case['query']!r}\n"
            f"Chunks returned: {len(chunks)}"
        )


@pytest.mark.rag
@pytest.mark.integration
@pytest.mark.parametrize(
    "case",
    RAG_RETRIEVAL_CASES,
    ids=[c["description"] for c in RAG_RETRIEVAL_CASES],
)
def test_retrieval_correct_source(case):
    """At least one chunk must come from the expected source file."""
    if "expected_source" not in case:
        pytest.skip("No expected_source defined")

    chunks = _search(case["query"])
    sources = [c.get("source", "") for c in chunks]

    assert any(case["expected_source"] in s for s in sources), (
        f"Expected source '{case['expected_source']}' not found.\n"
        f"Query: {case['query']!r}\n"
        f"Sources found: {sources}"
    )


# ─── Similarity threshold and noise filtering ────────────────────────────────

@pytest.mark.rag
@pytest.mark.integration
def test_irrelevant_query_returns_few_chunks():
    """Completely unrelated queries should return 0 or very few chunks."""
    chunks = _search("quantum physics string theory dark matter")

    assert len(chunks) <= 2, (
        f"Irrelevant query returned {len(chunks)} chunks (expected <= 2). "
        f"Threshold may be too low."
    )


@pytest.mark.rag
@pytest.mark.integration
def test_similarity_scores_are_reasonable():
    """Top chunks for relevant queries should have scores > 0.6."""
    chunks = _search("How is revenue calculated?")

    assert chunks, "No chunks returned for a known-good query"
    top_score = chunks[0]["score"]
    assert top_score > 0.6, (
        f"Top chunk score {top_score} is too low for a highly relevant query"
    )


# ─── Reranker effectiveness ──────────────────────────────────────────────────

@pytest.mark.rag
@pytest.mark.llm
def test_reranker_filters_noise():
    """Reranker should reduce chunk count by filtering irrelevant results."""
    # This query will match many chunks on "AdventureWorks" keyword
    raw_chunks = _search("AdventureWorks revenue calculation")
    state = _run_rag_agent("How is revenue calculated in AdventureWorks?")
    filtered_chunks = state.get("rag_chunks", [])

    # Reranker should return fewer than raw retrieval (filtering noise)
    # or at least not more
    assert len(filtered_chunks) <= len(raw_chunks), (
        f"Reranker returned more chunks ({len(filtered_chunks)}) "
        f"than raw retrieval ({len(raw_chunks)})"
    )


@pytest.mark.rag
@pytest.mark.llm
def test_reranker_keeps_relevant_chunks():
    """Reranker must keep genuinely relevant chunks."""
    state = _run_rag_agent("How is revenue calculated?")
    filtered_chunks = state.get("rag_chunks", [])

    assert filtered_chunks, "Reranker filtered out all chunks for a relevant query"

    # At least one chunk should mention the revenue formula components
    all_content = " ".join(c["content"].lower() for c in filtered_chunks)
    assert "unitprice" in all_content or "revenue" in all_content, (
        "Reranker kept chunks but none mention revenue/unitprice"
    )


@pytest.mark.rag
@pytest.mark.llm
def test_reranker_returns_empty_for_irrelevant():
    """Reranker should return few/no chunks for completely irrelevant queries."""
    state = _run_rag_agent("What is the speed of light in a vacuum?")
    filtered_chunks = state.get("rag_chunks", [])

    assert len(filtered_chunks) <= 2, (
        f"Reranker kept {len(filtered_chunks)} chunks for an irrelevant query"
    )


# ─── DeepEval RAGAS-style metrics ────────────────────────────────────────────

@pytest.mark.rag
@pytest.mark.llm
@pytest.mark.parametrize("case", RAG_RETRIEVAL_CASES[:5],
                         ids=[f"faithfulness_{c['description']}" for c in RAG_RETRIEVAL_CASES[:5]])
def test_rag_faithfulness(case):
    """RAG answer must be faithful to retrieved context (no hallucination)."""
    from deepeval import assert_test
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import FaithfulnessMetric
    from groq_judge import get_judge_model

    state = _run_rag_agent(case["query"])
    chunks = state.get("rag_chunks", [])

    if not chunks:
        pytest.skip("No chunks retrieved")

    from llm_config import get_llm
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = get_llm("response")
    chunk_text = "\n\n---\n\n".join(c["content"] for c in chunks)
    response = llm.invoke([
        SystemMessage(content="Answer using ONLY the provided context. Do not invent facts."),
        HumanMessage(content=f"Question: {case['query']}\n\nContext:\n{chunk_text}"),
    ])
    answer = response.content.strip()

    test_case = LLMTestCase(
        input=case["query"],
        actual_output=answer,
        retrieval_context=[c["content"] for c in chunks],
    )
    assert_test(test_case, [FaithfulnessMetric(threshold=0.5, model=get_judge_model())])


@pytest.mark.rag
@pytest.mark.llm
@pytest.mark.parametrize("case", RAG_RETRIEVAL_CASES[:5],
                         ids=[f"relevancy_{c['description']}" for c in RAG_RETRIEVAL_CASES[:5]])
def test_rag_answer_relevancy(case):
    """RAG answer must be relevant to the original question."""
    from deepeval import assert_test
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import AnswerRelevancyMetric
    from groq_judge import get_judge_model

    state = _run_rag_agent(case["query"])
    chunks = state.get("rag_chunks", [])

    if not chunks:
        pytest.skip("No chunks retrieved")

    from llm_config import get_llm
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = get_llm("response")
    chunk_text = "\n\n---\n\n".join(c["content"] for c in chunks)
    response = llm.invoke([
        SystemMessage(content="Answer using ONLY the provided context."),
        HumanMessage(content=f"Question: {case['query']}\n\nContext:\n{chunk_text}"),
    ])

    test_case = LLMTestCase(
        input=case["query"],
        actual_output=response.content.strip(),
        retrieval_context=[c["content"] for c in chunks],
    )
    assert_test(test_case, [AnswerRelevancyMetric(threshold=0.5, model=get_judge_model())])


@pytest.mark.rag
@pytest.mark.llm
@pytest.mark.parametrize("case", RAG_RETRIEVAL_CASES[:5],
                         ids=[f"ctx_relevancy_{c['description']}" for c in RAG_RETRIEVAL_CASES[:5]])
def test_rag_contextual_relevancy(case):
    """Retrieved chunks must be contextually relevant to the query."""
    from deepeval import assert_test
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import ContextualRelevancyMetric
    from groq_judge import get_judge_model

    chunks = _search(case["query"])

    if not chunks:
        pytest.skip("No chunks retrieved")

    test_case = LLMTestCase(
        input=case["query"],
        actual_output="placeholder",
        retrieval_context=[c["content"] for c in chunks],
    )
    assert_test(test_case, [ContextualRelevancyMetric(threshold=0.4, model=get_judge_model())])


# ─── Aggregate retrieval success rate ─────────────────────────────────────────

@pytest.mark.rag
@pytest.mark.integration
def test_rag_retrieval_overall_success_rate():
    """At least 80% of RAG test cases must retrieve relevant content."""
    successes = 0
    for case in RAG_RETRIEVAL_CASES:
        chunks = _search(case["query"])
        all_content = " ".join(c["content"].lower() for c in chunks)

        if all(kw.lower() in all_content for kw in case["must_contain"]):
            successes += 1

    rate = successes / len(RAG_RETRIEVAL_CASES)
    print(f"\nRAG retrieval success rate: {successes}/{len(RAG_RETRIEVAL_CASES)} = {rate:.1%}")

    assert rate >= 0.80, (
        f"RAG retrieval success rate {rate:.1%} is below 80% threshold."
    )
