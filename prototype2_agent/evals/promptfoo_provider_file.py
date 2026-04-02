"""Promptfoo Python file provider.

Promptfoo calls call_api(prompt, options, context) directly via Python import.
This avoids the exec/stdin issues with shell providers.

See: https://www.promptfoo.dev/docs/providers/python/
"""

import json
import os
import sys

EVALS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(EVALS_DIR, "..")
VENV_SITE = os.path.join(PROJECT_ROOT, ".venv", "lib")

# Add venv site-packages so imports work even when Promptfoo uses system Python
for d in os.listdir(VENV_SITE) if os.path.isdir(VENV_SITE) else []:
    sp = os.path.join(VENV_SITE, d, "site-packages")
    if os.path.isdir(sp) and sp not in sys.path:
        sys.path.insert(0, sp)

sys.path.insert(0, PROJECT_ROOT)

# Load env vars manually (no dotenv dependency needed)
env_path = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


def call_api(prompt, options=None, context=None):
    """Called by Promptfoo for each test case.

    Args:
        prompt: The rendered prompt string (our query)
        options: Provider config from promptfooconfig.yaml
        context: Test context including vars

    Returns:
        dict with 'output' key (string) and optional 'metadata'
    """
    from graph import compiled_graph

    # Extract query from all possible locations Promptfoo might put it
    query = ""

    # 1. Rendered prompt (works when Promptfoo resolves {{query}})
    if prompt and prompt.strip() and "{{" not in prompt:
        query = prompt.strip()

    # 2. context.vars.query (top-level vars)
    if not query and context and isinstance(context, dict):
        query = context.get("vars", {}).get("query", "")

    # 3. context.test.vars.query (test-level vars)
    if not query and context and isinstance(context, dict):
        test = context.get("test", {})
        if isinstance(test, dict):
            query = test.get("vars", {}).get("query", "")

    # 4. Render the template ourselves from context
    if not query and prompt and "{{query}}" in prompt and context and isinstance(context, dict):
        q = context.get("vars", {}).get("query", "") or context.get("test", {}).get("vars", {}).get("query", "")
        if q:
            query = q

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
            },
        }
    except Exception as e:
        return {
            "output": f"Error: {type(e).__name__}: {str(e)[:200]}",
            "error": str(e)[:500],
        }
