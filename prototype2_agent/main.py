"""Entrypoint: initializes the database, captures schema snapshot, and launches the Streamlit UI.

Usage:
    python main.py
"""

import os
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()


def main():
    """Initialize infrastructure and launch the Streamlit app."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)

    # 1. Initialize pgvector extension and documents table
    print("Initializing database and pgvector extension...")
    from db.connection import init_pgvector

    try:
        init_pgvector()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Warning: Database initialization failed: {e}")
        print("Make sure PostgreSQL is running and .env is configured correctly.")

    # 2. Capture schema snapshot
    print("Capturing schema snapshot...")
    from db.schema_snapshot import capture_schema_snapshot

    try:
        schema = capture_schema_snapshot()
        table_count = len(schema)
        print(f"Schema snapshot saved ({table_count} tables).")
    except Exception as e:
        print(f"Warning: Schema snapshot failed: {e}")

    # 2b. Capture FK snapshot (declared constraints + preserved manual entries)
    print("Capturing FK snapshot...")
    from db.fk_snapshot import capture_fk_snapshot

    try:
        fks = capture_fk_snapshot()
        fk_count = sum(len(cols) for cols in fks.values())
        print(f"FK snapshot saved ({fk_count} relationships across {len(fks)} tables).")
    except Exception as e:
        print(f"Warning: FK snapshot failed: {e}")

    # 3. Ingest new knowledge base documents (skips already-ingested files)
    print("Checking knowledge base for new documents...")
    from rag.ingest import ingest_knowledge_base

    try:
        ingest_knowledge_base()
    except Exception as e:
        print(f"Warning: Knowledge base ingestion failed: {e}")
        print("RAG features may be limited. Ensure Ollama is running with bge-large-en-v1.5.")

    # 4. Launch Streamlit UI
    print("Launching Streamlit UI...")
    ui_path = os.path.join(project_root, "ui", "app.py")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", ui_path, "--server.headless", "true"],
        cwd=project_root,
    )


if __name__ == "__main__":
    main()
