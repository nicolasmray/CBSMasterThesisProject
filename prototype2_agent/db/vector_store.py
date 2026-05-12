"""Vector store access — pgvector semantic search and chunk ingestion."""

from __future__ import annotations

import json

from sqlalchemy import text

from db.connection import get_engine
from llm_config import get_embeddings, RAG_TOP_K, RAG_SIMILARITY_THRESHOLD


def semantic_search(query: str, top_k: int | None = None,
                    min_score: float | None = None) -> list[dict]:
    """Embed the query and retrieve the closest document chunks from pgvector.

    Uses a similarity threshold to dynamically filter results — only chunks
    above the threshold are returned, up to top_k. Settings default to the
    values in llm_config.py.

    Args:
        query: The natural-language search query.
        top_k: Max chunks to return (default: RAG_TOP_K from llm_config).
        min_score: Minimum cosine similarity 0-1 (default: RAG_SIMILARITY_THRESHOLD).

    Returns:
        List of dicts with 'content', 'score', and 'source' keys,
        ranked by similarity (highest first).
    """
    if top_k is None:
        top_k = RAG_TOP_K
    if min_score is None:
        min_score = RAG_SIMILARITY_THRESHOLD

    query_embedding = get_embeddings().embed_query(query)
    embedding_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT content,
                       1 - (embedding <=> CAST(:emb AS vector)) AS score,
                       metadata->>'source' AS source
                FROM rag_chunks
                WHERE 1 - (embedding <=> CAST(:emb AS vector)) >= :min_score
                ORDER BY embedding <=> CAST(:emb AS vector)
                LIMIT :k
            """),
            {"emb": embedding_literal, "k": top_k, "min_score": min_score},
        ).fetchall()

    return [{"content": row[0], "score": round(row[1], 3), "source": row[2]}
            for row in rows]


def semantic_search_no_threshold(query: str, top_k: int = 10) -> list[dict]:
    """Fallback search: return the top_k closest chunks regardless of score.

    Used when the primary threshold-based search returns 0 results.
    Returns the same dict format as semantic_search but without any
    minimum score filter.
    """
    query_embedding = get_embeddings().embed_query(query)
    embedding_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT content,
                       1 - (embedding <=> CAST(:emb AS vector)) AS score,
                       metadata->>'source' AS source
                FROM rag_chunks
                ORDER BY embedding <=> CAST(:emb AS vector)
                LIMIT :k
            """),
            {"emb": embedding_literal, "k": top_k},
        ).fetchall()

    return [{"content": row[0], "score": round(row[1], 3), "source": row[2]}
            for row in rows]


def get_ingested_sources() -> set[str]:
    """Return the set of source filenames already present in rag_chunks.

    Used by the ingestion pipeline to skip files that have already been embedded.
    Belongs here (L2B) rather than in the pipeline script because it is a raw
    data-access query against the vector store table.
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT DISTINCT metadata->>'source' FROM rag_chunks "
                "WHERE metadata->>'source' IS NOT NULL"
            )
        ).fetchall()
    return {row[0] for row in rows}


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
