"""RAG specialist agent.

Retrieves relevant document chunks from the pgvector store, then uses an LLM
to filter out noise. Includes a fallback mechanism: if no chunks pass the
similarity threshold, a second search retrieves the top 10 closest chunks
regardless of score, then the LLM filters those for relevance.

The user is notified when the fallback is activated.
"""

import json

from langchain_core.messages import SystemMessage, HumanMessage

from state import AgentState
from db.vector_store import semantic_search, semantic_search_no_threshold
from llm_config import invoke_with_retry, RAG_SIMILARITY_THRESHOLD

# ── Reranker prompt ───────────────────────────────────────────────────────────
RERANKER_PROMPT = """\
You are a relevance filter. Given a user question and a list of retrieved
document chunks, your job is to select ONLY the chunks that are genuinely
relevant to answering the question.

A chunk is relevant if it contains information that directly helps answer
the question. A chunk is NOT relevant just because it shares common words
(like a company name) with the question.

Return a JSON array of the chunk indices (0-based) that are relevant.
Example: [0, 2, 5]

If NONE of the chunks are relevant, return an empty array: []

Return ONLY the JSON array — no explanation, no markdown fences.
"""


def _rerank(user_query: str, candidates: list[dict]) -> list[dict]:
    """Send candidates to the LLM reranker and return only relevant chunks."""
    chunks_for_llm = "\n\n".join(
        f"[Chunk {i}] (score: {c['score']}, source: {c['source']})\n{c['content']}"
        for i, c in enumerate(candidates)
    )

    response = invoke_with_retry("rag", [
        SystemMessage(content=RERANKER_PROMPT),
        HumanMessage(content=(
            f"Question: {user_query}\n\n"
            f"Chunks:\n{chunks_for_llm}"
        )),
    ])

    # Parse the selected indices
    raw = response.content.strip().strip("`").strip()
    if raw.lower().startswith("json"):
        raw = raw[4:].strip()
    try:
        selected_indices = json.loads(raw)
        if not isinstance(selected_indices, list):
            selected_indices = list(range(len(candidates)))
    except (json.JSONDecodeError, TypeError):
        selected_indices = list(range(len(candidates)))

    return [
        candidates[i] for i in selected_indices
        if isinstance(i, int) and 0 <= i < len(candidates)
    ]


def rag_agent(state: AgentState) -> AgentState:
    """Retrieve and filter document chunks from pgvector.

    Flow:
      1. Primary search with similarity threshold (>= 0.55).
      2. If results found → LLM reranker filters noise → return.
      3. If 0 results → FALLBACK: retrieve top 10 regardless of score.
      4. LLM reranker filters the fallback results.
      5. Set rag_fallback=True so the UI can notify the user.

    Args:
        state: Current pipeline state with user_query.

    Returns:
        Partial AgentState with rag_chunks, rag_context, and rag_fallback.
    """
    user_query = state["user_query"]

    # ── Step 1: Primary search (with threshold) ──────────────────────────────
    candidates = semantic_search(user_query)

    if candidates:
        # Step 2: LLM reranker on threshold-passing chunks
        filtered = _rerank(user_query, candidates)

        if filtered:
            rag_context = "\n\n---\n\n".join(
                f"[{c['source']} | score: {c['score']}]\n{c['content']}"
                for c in filtered
            )
            return {
                "rag_chunks": filtered,
                "rag_context": rag_context,
                "rag_fallback": False,
            }

    # ── Step 3: Fallback search (no threshold, top 10) ───────────────────────
    fallback_candidates = semantic_search_no_threshold(user_query, top_k=10)

    if not fallback_candidates:
        return {
            "rag_chunks": [],
            "rag_context": "No relevant documents found.",
            "rag_fallback": False,
        }

    # Step 4: LLM reranker on fallback chunks
    filtered = _rerank(user_query, fallback_candidates)

    if filtered:
        # Record the highest score from fallback results for the user notification
        top_fallback_score = fallback_candidates[0]["score"]
        rag_context = (
            f"[Fallback search activated — no chunks scored above {RAG_SIMILARITY_THRESHOLD} threshold. "
            f"Best match score: {top_fallback_score}]\n\n"
            + "\n\n---\n\n".join(
                f"[{c['source']} | score: {c['score']}]\n{c['content']}"
                for c in filtered
            )
        )
        return {
            "rag_chunks": filtered,
            "rag_context": rag_context,
            "rag_fallback": True,
        }

    # Fallback also found nothing relevant after reranking
    return {
        "rag_chunks": [],
        "rag_context": "No relevant documents found.",
        "rag_fallback": False,
    }
