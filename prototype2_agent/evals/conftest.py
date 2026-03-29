"""Shared pytest fixtures for the evaluation suite.

Sets up sys.path, loads .env, and provides reusable fixtures for
database connections, LLM instances, schema snapshots, and the
compiled LangGraph pipeline.
"""

import os
import sys
import time

import pytest
from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────────────────────────
EVALS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(EVALS_DIR, "..")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, EVALS_DIR)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


# ── LangSmith tracing (reads from .env) ───────────────────────────────────────
# LANGCHAIN_TRACING_V2=true enables tracing. LANGCHAIN_ENDPOINT must point
# to the correct regional API (e.g. https://eu.api.smith.langchain.com).


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def db_engine():
    """SQLAlchemy engine connected to the test database."""
    from db.connection import get_engine
    engine = get_engine()
    # Quick smoke test
    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return engine


@pytest.fixture(scope="session")
def schema_snapshot():
    """The flat schema dict (schema.table -> columns)."""
    from db.schema_snapshot import load_schema_snapshot
    return load_schema_snapshot()


@pytest.fixture(scope="session")
def compact_schema():
    """The compact schema string used by the SQL agent."""
    from db.schema_snapshot import get_compact_schema
    return get_compact_schema()


@pytest.fixture(scope="session")
def compiled_graph():
    """The compiled LangGraph pipeline."""
    from graph import compiled_graph
    return compiled_graph


@pytest.fixture(scope="session")
def llm_orchestrator():
    """LLM instance for the orchestrator agent."""
    from llm_config import get_llm
    return get_llm("orchestrator")


@pytest.fixture(scope="session")
def llm_sql():
    """LLM instance for the SQL agent."""
    from llm_config import get_llm
    return get_llm("sql")


@pytest.fixture(scope="session")
def llm_rag():
    """LLM instance for the RAG agent."""
    from llm_config import get_llm
    return get_llm("rag")


@pytest.fixture(scope="session")
def llm_response():
    """LLM instance for the response agent."""
    from llm_config import get_llm
    return get_llm("response")


@pytest.fixture(scope="session")
def embedding_model():
    """The shared embedding model instance."""
    from llm_config import get_embeddings
    return get_embeddings()


class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start


@pytest.fixture
def timer():
    """Provides a Timer context manager for latency tests."""
    return Timer
