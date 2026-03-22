"""RAG tool implementations — imported by the MCP server, never by agents directly.

Lazy embedding initialisation
------------------------------
``OllamaEmbeddings`` is intentionally NOT instantiated at module import time.
Creating it at import would cause the MCP server to attempt a connection to
Ollama the moment the module is loaded — crashing the entire server process
(with a TaskGroup exception visible to the client) whenever Ollama is not
running or the model is not yet loaded.

Instead, :func:`_get_embeddings` creates the instance on the first RAG call
and caches it in ``_embeddings``.  SQL queries work without Ollama being
available at all; only the ``semantic_search`` and ``embed_and_store`` tools
require it, and they will surface a clear error if Ollama is unreachable.
"""

from __future__ import annotations

import json

from sqlalchemy import text

from db.connection import get_engine

# Populated lazily on first call to a RAG tool — NOT at import time.
_embeddings = None


def _get_embeddings():
    """Return the shared OllamaEmbeddings instance, creating it on first call.

    Raises:
        ImportError: If ``langchain_community`` is not installed.
        Exception: If Ollama is unreachable or the model is not available.
    """
    global _embeddings
    if _embeddings is None:
        from langchain_community.embeddings import OllamaEmbeddings  # noqa: PLC0415
        _embeddings = OllamaEmbeddings(model="bge-large-en-v1.5")
    return _embeddings


def semantic_search(query: str, top_k: int = 5) -> list[str]:
    """Embed the query and retrieve the closest document chunks from pgvector.

    Args:
        query: The natural-language search query.
        top_k: Number of results to return.

    Returns:
        List of document content strings ranked by cosine similarity.
    """
    query_embedding = _get_embeddings().embed_query(query)
    embedding_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT content, 1 - (embedding <=> :emb::vector) AS score
                FROM documents
                ORDER BY embedding <=> :emb::vector
                LIMIT :k
            """),
            {"emb": embedding_literal, "k": top_k},
        ).fetchall()

    return [row[0] for row in rows]


def embed_and_store(content: str, metadata: dict) -> bool:
    """Embed a text chunk and store it in the documents table.

    Args:
        content: The text content to embed and store.
        metadata: JSON-serializable metadata dict.

    Returns:
        True on success.
    """
    embedding = _get_embeddings().embed_documents([content])[0]
    embedding_literal = "[" + ",".join(str(x) for x in embedding) + "]"

    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO documents (content, metadata, embedding)
                VALUES (:content, :metadata, :embedding::vector)
            """),
            {
                "content": content,
                "metadata": json.dumps(metadata),
                "embedding": embedding_literal,
            },
        )
        conn.commit()
    return True
