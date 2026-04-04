"""Promptfoo provider that runs the agent pipeline with Groq LLaMA 3.3 70B.

Same pipeline as the default Groq provider but uses the larger 70B model
for side-by-side comparison against LLaMA 4 Scout.
"""

import json
import os
import sys

EVALS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(EVALS_DIR, "..")
VENV_SITE = os.path.join(PROJECT_ROOT, ".venv", "lib")

# Add venv site-packages
for d in os.listdir(VENV_SITE) if os.path.isdir(VENV_SITE) else []:
    sp = os.path.join(VENV_SITE, d, "site-packages")
    if os.path.isdir(sp) and sp not in sys.path:
        sys.path.insert(0, sp)

sys.path.insert(0, PROJECT_ROOT)

# Load env vars manually
env_path = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

# Patch llm_config to use LLaMA 3.3 70B on Groq BEFORE any agent imports
import llm_config
llm_config.PROVIDER = "groq"
llm_config.GROQ_MODEL = "llama-3.3-70b-versatile"


def call_api(prompt, options=None, context=None):
    """Called by Promptfoo — runs pipeline with Groq LLaMA 3.3 70B."""
    from graph import compiled_graph

    # Extract query
    query = ""
    if prompt and prompt.strip() and "{{" not in prompt:
        query = prompt.strip()
    if not query and context and isinstance(context, dict):
        query = context.get("vars", {}).get("query", "")
    if not query and context and isinstance(context, dict):
        query = context.get("test", {}).get("vars", {}).get("query", "")

    if not query:
        return {"output": "Error: no query received", "error": "Empty prompt"}

    try:
        result = compiled_graph.invoke({"user_query": query})
        return {
            "output": result.get("final_answer", "No answer"),
            "metadata": {
                "intent": result.get("intent", ""),
                "sql_query": result.get("sql_query", ""),
                "row_count": len(result.get("sql_result", [])),
                "has_rag": bool(result.get("rag_chunks")),
                "has_chart": bool(result.get("chart_spec", {}).get("options")),
                "error": result.get("error", ""),
                "model": "groq:llama-3.3-70b-versatile",
            },
        }
    except Exception as e:
        return {
            "output": f"Error: {type(e).__name__}: {str(e)[:200]}",
            "error": str(e)[:500],
        }
