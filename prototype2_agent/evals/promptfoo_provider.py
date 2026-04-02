#!/usr/bin/env python3
"""Promptfoo custom provider — calls our agent pipeline and returns the result.

Promptfoo invokes this script with a prompt via stdin/env, and we return
the agent's response as JSON on stdout.

Usage by promptfoo:
    python evals/promptfoo_provider.py "What is the total revenue for 2024?"
"""

import json
import os
import sys

# Setup paths
EVALS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(EVALS_DIR, "..")
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def call_agent(query: str) -> dict:
    """Run the full pipeline and return structured output."""
    from graph import compiled_graph

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
            "metadata": {"error": str(e)[:500]},
        }


if __name__ == "__main__":
    # Promptfoo exec providers receive the prompt on stdin
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = sys.stdin.read().strip()

    # Promptfoo may wrap the query in JSON — extract if so
    if query.startswith("{"):
        try:
            data = json.loads(query)
            query = data.get("prompt", data.get("query", query))
        except json.JSONDecodeError:
            pass

    # Log to stderr for debugging (won't interfere with stdout JSON)
    print(f"[provider] Query: {query[:80]}", file=sys.stderr)

    result = call_agent(query)
    print(json.dumps(result))
