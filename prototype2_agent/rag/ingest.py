"""RAG document ingestion pipeline.

Provides two entry points:
  1. CLI:  python -m rag.ingest <file_path>  — ingest a single file.
  2. API:  ingest_knowledge_base()            — called by main.py on startup.
           Scans knowledge_base/ for new files and embeds them.

Both paths call db.vector_store functions directly (no MCP subprocess).
MCP is used only for SQL execution; document embedding does not require
subprocess isolation.
"""

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from db.vector_store import embed_and_store, get_ingested_sources

KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge_base")
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".md"}


def load_documents(file_path: str):
    """Load documents from a file path using the appropriate LangChain loader."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def chunk_documents(docs) -> list:
    """Split documents into chunks suitable for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
    )
    return splitter.split_documents(docs)


# ── Startup auto-ingest ───────────────────────────────────────────────────────

def ingest_knowledge_base():
    """Scan knowledge_base/ and embed any files not yet in rag_chunks.

    Skips files whose source filename is already present in the vector store.
    """
    if not os.path.isdir(KNOWLEDGE_BASE_DIR):
        print("  No knowledge_base/ directory found — skipping.")
        return

    candidates = [
        fname for fname in sorted(os.listdir(KNOWLEDGE_BASE_DIR))
        if os.path.splitext(fname)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if not candidates:
        print("  No supported files in knowledge_base/ — skipping.")
        return

    try:
        already_ingested = get_ingested_sources()
    except Exception:
        already_ingested = set()

    new_files = [f for f in candidates if f not in already_ingested]

    if not new_files:
        print(f"  All {len(candidates)} knowledge base file(s) already ingested.")
        return

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


# ── CLI single-file ingest ────────────────────────────────────────────────────

def ingest_file(file_path: str):
    """Load, chunk, and store a single document directly via db.vector_store."""
    print(f"Loading {file_path}...")
    docs = load_documents(file_path)
    print(f"Loaded {len(docs)} page(s)/section(s).")

    chunks = chunk_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    fname = os.path.basename(file_path)
    for i, chunk in enumerate(chunks):
        metadata = {
            **(chunk.metadata if hasattr(chunk, "metadata") else {}),
            "source": fname,
            "chunk_index": i,
        }
        embed_and_store(chunk.page_content, metadata)
        print(f"  Stored chunk {i + 1}/{len(chunks)}")

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

    ingest_file(file_path)


if __name__ == "__main__":
    main()
