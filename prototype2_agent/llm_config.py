"""Centralized LLM configuration for all agents.

Change the model for any agent (or all at once) from this single file.
No need to touch individual agent files.

Supported providers:
  - "ollama"  → local via Ollama (free, no API key)
  - "groq"    → Groq cloud API (requires GROQ_API_KEY in .env)

Groq API key rotation:
  Set GROQ_API_KEYS as a comma-separated list in .env to enable automatic
  failover when a key hits rate limits. Falls back to GROQ_API_KEY if
  GROQ_API_KEYS is not set.
"""

import os
import threading

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


# ─── Groq API key rotation ───────────────────────────────────────────────────

class GroqKeyPool:
    """Thread-safe rotating pool of Groq API keys.

    Loads keys from GROQ_API_KEYS (comma-separated) or falls back to GROQ_API_KEY.
    When a key hits a rate limit, call rotate() to switch to the next one.
    """

    def __init__(self):
        keys_str = os.getenv("GROQ_API_KEYS", "")
        if keys_str:
            self._keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        else:
            single = os.getenv("GROQ_API_KEY", "")
            self._keys = [single] if single else []
        self._index = 0
        self._lock = threading.Lock()
        self._exhausted: set[int] = set()

    @property
    def current_key(self) -> str:
        if not self._keys:
            return ""
        with self._lock:
            return self._keys[self._index % len(self._keys)]

    def rotate(self) -> str | None:
        """Move to the next key. Returns the new key, or None if all exhausted."""
        with self._lock:
            self._exhausted.add(self._index % len(self._keys))
            if len(self._exhausted) >= len(self._keys):
                return None  # all keys exhausted
            # Find next non-exhausted key
            for _ in range(len(self._keys)):
                self._index = (self._index + 1) % len(self._keys)
                if self._index not in self._exhausted:
                    return self._keys[self._index]
            return None

    def reset(self):
        """Reset exhaustion state (e.g. after a cool-down period)."""
        with self._lock:
            self._exhausted.clear()

    @property
    def total_keys(self) -> int:
        return len(self._keys)

    @property
    def available_keys(self) -> int:
        return len(self._keys) - len(self._exhausted)


_key_pool = GroqKeyPool()


def get_groq_key() -> str:
    """Return the current active Groq API key."""
    return _key_pool.current_key


def rotate_groq_key() -> str | None:
    """Switch to the next Groq API key. Returns new key or None if all exhausted."""
    new_key = _key_pool.rotate()
    if new_key:
        # Also update the env var so any code reading os.getenv picks it up
        os.environ["GROQ_API_KEY"] = new_key
    return new_key


def get_key_pool() -> GroqKeyPool:
    """Return the key pool instance for status checks."""
    return _key_pool


# ─── Embeddings ───────────────────────────────────────────────────────────────

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


# ─── LLM with auto-retry on rate limit ───────────────────────────────────────

def get_llm(agent_name: str):
    """Return the configured LLM instance for a given agent.

    Uses the current key from the rotation pool. If the pool has multiple keys,
    callers should catch rate-limit errors and call rotate_groq_key() + retry.

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
            api_key=get_groq_key(),
            temperature=temperature,
        )
    else:  # "ollama" (default)
        from langchain_ollama import ChatOllama
        model = model_override or OLLAMA_MODEL
        return ChatOllama(
            model=model,
            temperature=temperature,
        )


def invoke_with_retry(agent_name: str, messages: list, max_retries: int = 3):
    """Invoke an LLM with automatic key rotation on rate limit errors.

    Args:
        agent_name: Agent name for get_llm().
        messages: LangChain message list.
        max_retries: Max rotation attempts before raising.

    Returns:
        The LLM response.

    Raises:
        The last rate-limit error if all keys are exhausted.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            llm = get_llm(agent_name)
            return llm.invoke(messages)
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate_limit" in error_str or "rate limit" in error_str:
                last_error = e
                new_key = rotate_groq_key()
                if new_key is None:
                    raise  # all keys exhausted
                pool = get_key_pool()
                print(f"  [Key rotation] Rate limited, switched to key {pool._index + 1}/{pool.total_keys}")
                continue
            raise  # not a rate limit error
    raise last_error
