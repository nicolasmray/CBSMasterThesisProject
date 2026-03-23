"""RAG specialist agent.

Answers questions by retrieving relevant document chunks from the vector store
via the MCP server's semantic_search tool.
"""

import asyncio
<<<<<<< HEAD
import concurrent.futures
import os
=======
>>>>>>> origin/main

from langchain_core.messages import SystemMessage, HumanMessage
from mcp.client.stdio import stdio_client

from state import AgentState
from mcp_client import get_server_params, call_tool
from llm_config import get_llm

# ── System prompt ─────────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """\
You are a RAG (Retrieval-Augmented Generation) specialist agent.

You have been given context passages retrieved from a document store.
Use ONLY the provided context to answer the user's question.
If the context does not contain enough information, say so clearly.
Do not make up facts. Cite specific passages when possible.

Respond with a clear, concise answer suitable for a business audience.
"""


async def _run_rag(state: AgentState) -> AgentState:
    """Async implementation: search documents via MCP and synthesize an answer."""
    user_query = state["user_query"]

    # Connect to MCP server and perform semantic search
    server_params = get_server_params()
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with mcp_client_session(read_stream, write_stream) as session:
            await session.initialize()

            # Retrieve relevant document chunks
            chunks = await call_tool(
                session, "semantic_search", {"query": user_query, "top_k": 5}
            )

    # Build context string from retrieved chunks
    if chunks and isinstance(chunks, list):
        rag_context = "\n\n---\n\n".join(str(c) for c in chunks)
    else:
        rag_context = "No relevant documents found."

    # Ask the LLM to synthesize an answer from the context
    llm = get_llm("rag")

    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Context:\n{rag_context}\n\n"
                f"Question: {user_query}"
            )
        ),
    ]

    response = llm.invoke(messages)
    return {"rag_context": response.content.strip()}


# We need an import for the MCP client session context manager
from mcp import ClientSession


class mcp_client_session:
    """Async context manager that wraps ClientSession initialization."""

    def __init__(self, read_stream, write_stream):
        self._session = ClientSession(read_stream, write_stream)

    async def __aenter__(self):
        await self._session.__aenter__()
        return self._session

    async def __aexit__(self, *args):
        await self._session.__aexit__(*args)


def rag_agent(state: AgentState) -> AgentState:
    """Retrieve document context via MCP semantic_search and synthesize an answer.

    Args:
        state: Current pipeline state with user_query.

    Returns:
        Partial AgentState update with rag_context.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _run_rag(state)).result()
