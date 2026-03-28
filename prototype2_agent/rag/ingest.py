"""RAG document ingestion.

Provides two entry points:
  1. CLI:  python -m rag.ingest <file_path>       — ingest a single file via MCP.
  2. API:  ingest_knowledge_base()                 — called by main.py on startup.
           Scans knowledge_base/ for new files, embeds them directly (no MCP).
"""

import asyncio
import json
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from sqlalchemy import text

from mcp_client import get_server_params, call_tool

KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge_base")
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".md"}


class _MCPSession:
    """Async context manager wrapping ClientSession."""

    def __init__(self, read_stream, write_stream):
        self._session = ClientSession(read_stream, write_stream)

    async def __aenter__(self):
        await self._session.__aenter__()
        return self._session

    async def __aexit__(self, *args):
        await self._session.__aexit__(*args)


def load_documents(file_path: str):
    """Load documents from a file path using the appropriate LangChain loader."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def chunk_documents(docs) -> list:
    """Split documents into chunks suitable for bge-large-en-v1.5."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
    )
    return splitter.split_documents(docs)


# ── Startup auto-ingest (direct, no MCP) ─────────────────────────────────────

def _get_ingested_sources() -> set[str]:
    """Query the documents table for all distinct source filenames already ingested."""
    from db.connection import get_engine

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT DISTINCT metadata->>'source' FROM rag_chunks WHERE metadata->>'source' IS NOT NULL")
        ).fetchall()
    return {row[0] for row in rows}


def ingest_knowledge_base():
    """Scan knowledge_base/ and embed any files not yet in the documents table.

    Calls embed_and_store directly (no MCP subprocess) for speed.
    """
    if not os.path.isdir(KNOWLEDGE_BASE_DIR):
        print("  No knowledge_base/ directory found — skipping.")
        return

    # Collect candidate files
    candidates = []
    for fname in sorted(os.listdir(KNOWLEDGE_BASE_DIR)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            candidates.append(fname)

    if not candidates:
        print("  No supported files in knowledge_base/ — skipping.")
        return

    # Check which files are already ingested
    try:
        already_ingested = _get_ingested_sources()
    except Exception:
        already_ingested = set()

    new_files = [f for f in candidates if f not in already_ingested]

    if not new_files:
        print(f"  All {len(candidates)} knowledge base file(s) already ingested.")
        return

    # Lazy import — only needed when we actually embed
    from mcp_server.tools.rag_tools import embed_and_store

    for fname in new_files:
        file_path = os.path.join(KNOWLEDGE_BASE_DIR, fname)
        print(f"  Ingesting {fname}...")

        docs = load_documents(file_path)
        chunks = chunk_documents(docs)
        print(f"    {len(chunks)} chunks to embed.")

        for i, chunk in enumerate(chunks):
            metadata = {
                **(chunk.metadata if hasattr(chunk, "metadata") else {}),
                "source": fname,
                "chunk_index": i,
            }
            embed_and_store(chunk.page_content, metadata)

        print(f"    Done ({len(chunks)} chunks stored).")

    print(f"  Ingested {len(new_files)} new file(s).")


# ── CLI single-file ingest (via MCP) ─────────────────────────────────────────

async def ingest_file(file_path: str):
    """Load, chunk, and store a document via the MCP embed_and_store tool."""
    print(f"Loading {file_path}...")
    docs = load_documents(file_path)
    print(f"Loaded {len(docs)} page(s)/section(s).")

    chunks = chunk_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    server_params = get_server_params()
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with _MCPSession(read_stream, write_stream) as session:
            await session.initialize()

            for i, chunk in enumerate(chunks):
                metadata = {
                    "source": file_path,
                    "chunk_index": i,
                    **(chunk.metadata if hasattr(chunk, "metadata") else {}),
                }
                result = await call_tool(
                    session,
                    "embed_and_store",
                    {"text": chunk.page_content, "meta": metadata},
                )
                print(f"  Stored chunk {i + 1}/{len(chunks)}: {result}")

    print("Ingestion complete.")


def main():
    """CLI entry-point for document ingestion."""
    if len(sys.argv) < 2:
        print("Usage: python -m rag.ingest <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.isfile(file_path):
        print(f"Error: file not found: {file_path}")
        sys.exit(1)

    asyncio.run(ingest_file(file_path))


if __name__ == "__main__":
    main()
