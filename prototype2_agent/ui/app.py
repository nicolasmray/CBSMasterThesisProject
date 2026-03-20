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
                error_msg = f"An error occurred: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })
