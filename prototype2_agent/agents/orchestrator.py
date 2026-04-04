"""Orchestrator / router agent.

Interprets the user query, classifies intent, and writes a high-level plan.
Does NOT call any MCP tools — it purely routes.
"""

from langchain_core.messages import SystemMessage, HumanMessage

from state import AgentState
from llm_config import invoke_with_retry

# ── System prompt (easy to tune) ──────────────────────────────────────────────
ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the Orchestrator of a multi-agent Business Intelligence assistant.

Your ONLY job is to:
1. Read the user's question.
2. Classify the intent into exactly ONE of these four categories:
   • "rag"    — the question is about documents, policies, definitions, or knowledge base content.
   • "sql"    — the question requires querying structured data in a database (metrics, counts, aggregations, lookups).
   • "chart"  — the question explicitly asks for a chart, graph, visualization, or plot AND requires data from the database.
   • "hybrid" — the question needs BOTH database data AND document/knowledge-base context to answer properly.
              Use "hybrid" when the question involves a derived or non-obvious metric whose formula
              must be looked up — e.g. profit margin, gross margin, COGS, cost of goods sold,
              cost vs revenue, CLV, LTV, churn rate, retention rate, conversion rate,
              average order value, inventory turnover, days on hand.
              These require a formula from the knowledge base AND a database query to compute them.
3. Write a short plan (1-3 sentences) describing how downstream agents should handle the request.

Respond with ONLY a JSON object (no markdown fences) with exactly two keys:
{"intent": "<rag|sql|chart|hybrid>", "plan": "<your plan>"}
"""


def orchestrator_agent(state: AgentState) -> AgentState:
    """Classify the user query intent and produce a routing plan.

    Args:
        state: Current pipeline state containing at least user_query.

    Returns:
        Partial AgentState update with intent and plan.
    """
    messages = [
        SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
        HumanMessage(content=state["user_query"]),
    ]

    response = invoke_with_retry("orchestrator", messages)
    content = response.content.strip()

    # Parse the JSON response from the LLM
    import json

    try:
        parsed = json.loads(content)
        intent = parsed.get("intent", "sql")
        plan = parsed.get("plan", "")
    except json.JSONDecodeError:
        # Fallback: try to extract intent from raw text
        lower = content.lower()
        if "rag" in lower:
            intent = "rag"
        elif "chart" in lower:
            intent = "chart"
        elif "hybrid" in lower:
            intent = "hybrid"
        else:
            intent = "sql"
        plan = content

    # Validate intent
    if intent not in ("rag", "sql", "chart", "hybrid"):
        intent = "sql"

    return {"intent": intent, "plan": plan}
