"""Streamlit chat UI for the multi-agent BI assistant.

Provides a conversational interface that sends user queries through the
LangGraph pipeline and displays results including text answers and charts.
"""

import base64
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

from graph import compiled_graph

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BI Assistant",
    page_icon="📊",
    layout="wide",
)

st.title("Multi-Agent BI Assistant")
st.caption("Ask questions about your data or documents — powered by LLaMA 4 Scout")

# ── Session state for conversation history ────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chart_b64"):
            chart_bytes = base64.b64decode(msg["chart_b64"])
            st.image(chart_bytes, caption=msg.get("chart_title", "Chart"))

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a question about your business data...")

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run the agent pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = compiled_graph.invoke({"user_query": user_input})
                final_answer = result.get("final_answer", "I wasn't able to generate an answer.")
                chart_spec = result.get("chart_spec", {})

                st.markdown(final_answer)

                chart_b64 = None
                chart_title = None
                if chart_spec and chart_spec.get("image_base64"):
                    chart_b64 = chart_spec["image_base64"]
                    chart_title = chart_spec.get("title", "Chart")
                    chart_bytes = base64.b64decode(chart_b64)
                    st.image(chart_bytes, caption=chart_title)

                # Store in history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_answer,
                    "chart_b64": chart_b64,
                    "chart_title": chart_title,
                })

            except Exception as e:
                # Unwrap nested exceptions (TaskGroup, ExceptionGroup) to surface the real error
                root_cause = e
                while hasattr(root_cause, 'exceptions') and root_cause.exceptions:
                    root_cause = root_cause.exceptions[0]
                while root_cause.__cause__:
                    root_cause = root_cause.__cause__

                error_type = type(root_cause).__name__
                error_detail = str(root_cause)

                # Provide user-friendly messages for common errors
                if "rate_limit" in error_detail.lower() or "429" in error_detail:
                    error_msg = (
                        "**Rate limit reached on the Groq API.** "
                        "You've exceeded the free-tier request limit. "
                        "Wait a minute and try again, or upgrade your Groq plan."
                    )
                elif "authentication" in error_detail.lower() or "401" in error_detail or "invalid api key" in error_detail.lower():
                    error_msg = (
                        "**Groq API authentication failed.** "
                        "Check that your `GROQ_API_KEY` in `.env` is correct and active."
                    )
                elif "quota" in error_detail.lower() or "insufficient" in error_detail.lower():
                    error_msg = (
                        "**Groq API quota exceeded.** "
                        "You've used all available tokens/requests. "
                        "Wait or upgrade your Groq plan."
                    )
                elif "connection" in error_detail.lower() or "timeout" in error_detail.lower():
                    error_msg = (
                        "**Connection error.** "
                        "Could not reach the Groq API or the database. "
                        "Check your internet connection and that PostgreSQL is running."
                    )
                else:
                    error_msg = f"**Error ({error_type}):** {error_detail}"

                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })
