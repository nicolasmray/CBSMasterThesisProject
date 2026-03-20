"""RAG document ingestion script.

Accepts a file path (PDF or plain text) as a CLI argument, chunks the document,
embeds each chunk via Ollama bge-large-en-v1.5, and stores them in pgvector
through the MCP server's embed_and_store tool.

Usage:
    python -m rag.ingest <file_path>
"""

import asyncio
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from mcp_client import get_server_params, call_tool


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
    """Load documents from a file path using the appropriate LangChain loader.

    Args:
        file_path: Path to a PDF or text file.

    Returns:
        List of LangChain Document objects.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def chunk_documents(docs) -> list:
    """Split documents into chunks suitable for bge-large-en-v1.5.

    Args:
        docs: List of LangChain Document objects.

    Returns:
        List of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
    )
    return splitter.split_documents(docs)


async def ingest_file(file_path: str):
    """Load, chunk, and store a document via the MCP embed_and_store tool.

    Args:
        file_path: Path to the source file.
    """
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
