"""RAG tool implementations — imported by the MCP server and by ingest.

Embedding model is configured centrally in llm_config.py and initialized
lazily on first call via get_embeddings(). SQL queries work without Ollama
being available; only semantic_search and embed_and_store require it.
"""

from __future__ import annotations

import json

from sqlalchemy import text

from db.connection import get_engine
from llm_config import get_embeddings


def semantic_search(query: str, top_k: int = 5) -> list[str]:
    """Embed the query and retrieve the closest document chunks from pgvector.

    Args:
        query: The natural-language search query.
        top_k: Number of results to return.

    Returns:
        List of document content strings ranked by cosine similarity.
    """
    query_embedding = get_embeddings().embed_query(query)
    embedding_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT content, 1 - (embedding <=> CAST(:emb AS vector)) AS score
                FROM rag_chunks
                ORDER BY embedding <=> CAST(:emb AS vector)
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
    embedding = get_embeddings().embed_documents([content])[0]
    embedding_literal = "[" + ",".join(str(x) for x in embedding) + "]"

    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO rag_chunks (content, metadata, embedding)
                VALUES (:content, :metadata, CAST(:embedding AS vector))
            """),
            {
                "content": content,
                "metadata": json.dumps(metadata),
                "embedding": embedding_literal,
            },
        )
        conn.commit()
    return True
