"""Streamlit demo for Persona-RAG.

Wraps the same LangGraph that powers the Telegram bot. Reviewers without
Telegram can paste a message and see what the persona would reply.
"""

from __future__ import annotations

import asyncio

import streamlit as st

from persona_rag.config import get_settings
from persona_rag.graph.compile import build_graph

settings = get_settings()
st.set_page_config(page_title=f"Persona-RAG: {settings.PERSONA_NAME}", page_icon="💬")
st.title(f"Chat with {settings.PERSONA_NAME}")
st.caption(settings.PERSONA_DESCRIPTION or "Retrieval-augmented persona demo")

if "history" not in st.session_state:
    st.session_state.history = []
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

for entry in st.session_state.history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

prompt = st.chat_input("Say something...")
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("..."):
            final = asyncio.run(
                st.session_state.graph.ainvoke(
                    {
                        "user_id": settings.ADMIN_TELEGRAM_ID,
                        "chat_id": 0,
                        "incoming": prompt,
                    }
                )
            )
        reply = final.get("reply") or "(no reply — auth check skipped you?)"
        st.markdown(reply)
        st.session_state.history.append({"role": "assistant", "content": reply})

    with st.expander("retrieval debug"):
        for r in final.get("retrieved", []):
            st.write(f"**{r.score:.3f}** — {r.turn.your_reply[:200]}")
