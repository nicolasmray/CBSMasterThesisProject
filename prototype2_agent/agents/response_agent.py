"""Response formulation agent.

Synthesizes a final, business-friendly answer from all populated state fields.
Always runs last on every graph path and is the sole writer of final_answer.
Does NOT call any MCP tools.
"""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from state import AgentState

load_dotenv()

# ── System prompt ─────────────────────────────────────────────────────────────
RESPONSE_SYSTEM_PROMPT = """\
You are the Response Agent — the final step in a multi-agent BI pipeline.

Your job is to write a clear, professional, business-friendly answer that a
non-technical business manager would understand.

Rules:
- Do NOT include raw SQL, technical jargon, or implementation details unless the user asked.
- If query results are provided, summarize the key findings with numbers.
- If a chart was generated, acknowledge it and describe the key business insight it shows.
- If an error occurred, explain the problem in plain language and suggest what the user could try.
- If both document context (RAG) and data results are available, weave them together.
- Keep the response concise but complete — aim for 2-5 sentences for simple queries,
  more for complex ones.
- Use bullet points or numbered lists when presenting multiple data points.
"""


def response_agent(state: AgentState) -> AgentState:
    """Synthesize all state into a final business-friendly answer.

    Reads: user_query, sql_result, rag_context, chart_spec, plan, error.
    Writes: final_answer.

    Args:
        state: Current pipeline state.

    Returns:
        Partial AgentState update with final_answer.
    """
    user_query = state.get("user_query", "")
    sql_result = state.get("sql_result", [])
    rag_context = state.get("rag_context", "")
    chart_spec = state.get("chart_spec", {})
    plan = state.get("plan", "")
    error = state.get("error", "")

    # Build the context message for the LLM
    parts = [f"User question: {user_query}"]

    if plan:
        parts.append(f"Analysis plan: {plan}")

    if error:
        parts.append(f"Error encountered: {error}")

    if sql_result:
        import json

        # Limit displayed results to avoid token overflow
        display = sql_result[:20]
        parts.append(f"Query results ({len(sql_result)} rows, showing first {len(display)}):\n{json.dumps(display, indent=2, default=str)}")

    if rag_context:
        parts.append(f"Document context:\n{rag_context}")

    if chart_spec and chart_spec.get("image_base64"):
        parts.append(
            f"A {chart_spec.get('chart_type', 'chart')} chart titled "
            f"\"{chart_spec.get('title', 'Chart')}\" has been generated "
            f"(x={chart_spec.get('x', '?')}, y={chart_spec.get('y', '?')})."
        )

    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
    )

    messages = [
        SystemMessage(content=RESPONSE_SYSTEM_PROMPT),
        HumanMessage(content="\n\n".join(parts)),
    ]

    response = llm.invoke(messages)
    return {"final_answer": response.content.strip()}
