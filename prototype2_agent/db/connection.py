"""PostgreSQL + pgvector connection and table setup."""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


def get_connection_string() -> str:
    """Build the PostgreSQL connection string from environment variables."""
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "prototype2")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "")
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


def get_engine():
    """Create and return a SQLAlchemy engine."""
    return create_engine(get_connection_string())


def init_pgvector():
    """Enable the pgvector extension and create the rag_chunks table with HNSW index.

    This is idempotent — safe to call on every startup.
    Uses 'rag_chunks' to avoid collision with the AdventureWorks 'documents' table.
    """
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS rag_chunks (
                id SERIAL PRIMARY KEY,
                content TEXT,
                metadata JSONB,
                embedding VECTOR(1024)
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx
            ON rag_chunks USING hnsw (embedding vector_cosine_ops)
        """))
        conn.commit()
