"""Centralized LLM configuration for all agents.

Change the model for any agent (or all at once) from this single file.
No need to touch individual agent files.

Supported providers:
  - "ollama"  → local via Ollama (free, no API key)
  - "groq"    → Groq cloud API (requires GROQ_API_KEY in .env)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── Provider: "ollama" or "groq" ────────────────────────────────────────────
PROVIDER = "groq"

# ─── Model settings per provider ─────────────────────────────────────────────
OLLAMA_MODEL = "llama3.1"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ─── Per-agent overrides (set to None to use the defaults above) ─────────────
AGENT_MODELS = {
    "orchestrator": None,   # e.g. "llama3.1" or "mistral"
    "rag":          None,
    "sql":          None,
    "chart":        None,
    "response":     None,
}

# ─── Embedding model (always local via Ollama — free, no API tokens) ─────────
EMBEDDING_MODEL = "mxbai-embed-large"

# ─── RAG retrieval settings ──────────────────────────────────────────────────
RAG_TOP_K = 10                 # max chunks to retrieve from pgvector
RAG_SIMILARITY_THRESHOLD = 0.55  # min cosine similarity to include a chunk (0-1)

# ─── Per-agent temperature overrides (set to None to use default) ────────────
AGENT_TEMPERATURES = {
    "orchestrator": 0,
    "rag":          0,
    "sql":          0,
    "chart":        0,
    "response":     0,
}


_embeddings = None


def get_embeddings():
    """Return the shared OllamaEmbeddings instance, creating it on first call.

    Always uses a local Ollama model (free, no API tokens, specialized for embeddings).
    """
    global _embeddings
    if _embeddings is None:
        from langchain_ollama import OllamaEmbeddings
        _embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return _embeddings


def get_llm(agent_name: str):
    """Return the configured LLM instance for a given agent.

    Args:
        agent_name: One of "orchestrator", "rag", "sql", "chart", "response".

    Returns:
        A LangChain chat model instance (ChatOllama or ChatGroq).
    """
    temperature = AGENT_TEMPERATURES.get(agent_name, 0)
    if temperature is None:
        temperature = 0

    model_override = AGENT_MODELS.get(agent_name)

    if PROVIDER == "groq":
        from langchain_groq import ChatGroq
        model = model_override or GROQ_MODEL
        return ChatGroq(
            model=model,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=temperature,
        )
    else:  # "ollama" (default)
        from langchain_ollama import ChatOllama
        model = model_override or OLLAMA_MODEL
        return ChatOllama(
            model=model,
            temperature=temperature,
        )
