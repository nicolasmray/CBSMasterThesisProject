"""Shared agent state definition for the multi-agent BI assistant."""

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """Shared state passed between all agents in the LangGraph pipeline.

    Each agent reads only the fields it needs and writes back a partial update.
    """

    user_query: str
    intent: str  # "rag" | "sql" | "chart" | "hybrid"
    plan: str
    schema_context: str  # relevant table/column names passed to sql_agent
    sql_query: str
    sql_result: list[dict]
    rag_context: str  # LLM-synthesized interpretation (for hybrid path)
    rag_chunks: list[dict]  # raw retrieved chunks: [{content, score, source}, ...]
    rag_fallback: bool  # True when fallback search was used (below-threshold results)
    chart_spec: dict  # contains Plotly figure JSON under key "figure_json"
    final_answer: str
    error: str
    retry_count: int
