"""RAG specialist agent.

Retrieves relevant document chunks from the pgvector store, then uses an LLM
to filter out noise (e.g. chunks that match on common words like "AdventureWorks"
but are not actually relevant to the question). The LLM only selects — it does
not synthesize or rephrase. Raw chunks are passed through to the response agent.
"""

import json

from langchain_core.messages import SystemMessage, HumanMessage

from state import AgentState
from mcp_server.tools.rag_tools import semantic_search
from llm_config import get_llm

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


def rag_agent(state: AgentState) -> AgentState:
    """Retrieve and filter document chunks from pgvector.

    1. Semantic search retrieves candidate chunks (wide net).
    2. LLM reranker filters out irrelevant chunks (keyword noise).
    3. Raw filtered chunks are returned — no synthesis, no rephrasing.

    Args:
        state: Current pipeline state with user_query.

    Returns:
        Partial AgentState with rag_chunks (raw filtered) and rag_context (formatted).
    """
    user_query = state["user_query"]

    # Step 1: Retrieve candidate chunks
    candidates = semantic_search(user_query)

    if not candidates:
        return {"rag_chunks": [], "rag_context": "No relevant documents found."}

    # Step 2: LLM reranker — select only truly relevant chunks
    chunks_for_llm = "\n\n".join(
        f"[Chunk {i}] (score: {c['score']}, source: {c['source']})\n{c['content']}"
        for i, c in enumerate(candidates)
    )

    llm = get_llm("rag")
    response = llm.invoke([
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
        # Fallback: keep all candidates if parsing fails
        selected_indices = list(range(len(candidates)))

    # Filter to valid indices
    filtered = [
        candidates[i] for i in selected_indices
        if isinstance(i, int) and 0 <= i < len(candidates)
    ]

    # Build context string for downstream agents (hybrid path)
    if filtered:
        rag_context = "\n\n---\n\n".join(
            f"[{c['source']} | score: {c['score']}]\n{c['content']}"
            for c in filtered
        )
    else:
        rag_context = "No relevant documents found."

    return {"rag_chunks": filtered, "rag_context": rag_context}
