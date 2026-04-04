"""LangGraph StateGraph wiring all agents and conditional edges.

Routes:
  START → orchestrator → conditional_router:
    "rag"    → rag_agent → response_agent → END
    "sql"    → sql_agent → response_agent → END
    "chart"  → sql_agent → chart_agent → response_agent → END
    "hybrid" → rag_agent → sql_agent → chart_agent → response_agent → END
"""

from langgraph.graph import StateGraph, END

from state import AgentState
from agents.orchestrator import orchestrator_agent
from agents.rag_agent import rag_agent
from agents.sql_agent import sql_agent
from agents.chart_agent import chart_agent
from agents.response_agent import response_agent


def _route_by_intent(state: AgentState) -> str:
    """Conditional edge: route based on orchestrator-determined intent.

    Args:
        state: Current pipeline state with intent field set.

    Returns:
        Name of the next node to execute.
    """
    intent = state.get("intent", "sql")
    if intent in ("rag", "hybrid"):
        return "rag_agent"
    else:  # "sql", "chart", or fallback
        return "sql_agent"


def _route_after_sql(state: AgentState) -> str:
    """Route after SQL agent: go to chart_agent, rag_agent, or response_agent.

    Args:
        state: Current pipeline state with intent field.

    Returns:
        Name of the next node.
    """
    intent = state.get("intent", "sql")
    if intent in ("chart", "hybrid"):
        return "chart_agent"
    else:
        return "response_agent"


def build_graph() -> StateGraph:
    """Construct and compile the LangGraph StateGraph.

    Returns:
        A compiled LangGraph graph ready for .invoke().
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("orchestrator", orchestrator_agent)
    graph.add_node("rag_agent", rag_agent)
    graph.add_node("sql_agent", sql_agent)
    graph.add_node("chart_agent", chart_agent)
    graph.add_node("response_agent", response_agent)

    # Edges: START → orchestrator
    graph.set_entry_point("orchestrator")

    # Conditional routing from orchestrator
    graph.add_conditional_edges(
        "orchestrator",
        _route_by_intent,
        {
            "rag_agent": "rag_agent",
            "sql_agent": "sql_agent",
        },
    )

    # RAG agent: pure rag → response; hybrid → sql
    graph.add_conditional_edges(
        "rag_agent",
        lambda s: "sql_agent" if s.get("intent") == "hybrid" else "response_agent",
        {"sql_agent": "sql_agent", "response_agent": "response_agent"},
    )

    # SQL agent conditionally goes to chart, rag (hybrid), or response
    graph.add_conditional_edges(
        "sql_agent",
        _route_after_sql,
        {
            "chart_agent": "chart_agent",
            "rag_agent": "rag_agent",
            "response_agent": "response_agent",
        },
    )

    # Chart agent always goes to response
    graph.add_edge("chart_agent", "response_agent")

    # Response agent always ends
    graph.add_edge("response_agent", END)

    return graph.compile()


# Pre-compiled graph instance for import
compiled_graph = build_graph()
