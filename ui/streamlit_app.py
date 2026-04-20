from __future__ import annotations

import contextlib
import io
import sys
import time
import uuid
from pathlib import Path
from typing import Callable

import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


@st.cache_resource
def load_agent() -> Callable[[str, str], str]:
    from app.agent import ask

    return ask


def create_new_thread() -> str:
    return str(uuid.uuid4())


def reset_conversation() -> None:
    st.session_state.messages = []
    st.session_state.thread_id = create_new_thread()
    st.rerun()


def run_agent(prompt: str, thread_id: str) -> str:
    ask = load_agent()
    captured_output = io.StringIO()

    try:
        with contextlib.redirect_stdout(captured_output), contextlib.redirect_stderr(captured_output):
            response = ask(prompt, thread_id=thread_id)
    except Exception:
        return "⚠️ AI is thinking... try again"

    cleaned_response = str(response).strip()
    return cleaned_response or "⚠️ AI is thinking... try again"


def animate_markdown(message: str, delay: float = 0.01) -> None:
    placeholder = st.empty()
    rendered = ""
    for line in message.splitlines():
        rendered = f"{rendered}\n{line}".strip()
        placeholder.markdown(rendered)
        time.sleep(delay)
    if not message:
        placeholder.markdown("")


st.set_page_config(
    page_title="AI Sales Assistant (E-commerce)",
    page_icon="🛍️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 960px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .assistant-shell {
        border: 1px solid rgba(120, 120, 120, 0.15);
        border-radius: 24px;
        padding: 1.2rem 1.2rem 0.4rem 1.2rem;
        background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(247,249,252,0.98));
        box-shadow: 0 20px 60px rgba(15, 23, 42, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = create_new_thread()


with st.sidebar:
    st.title("AI Sales Assistant")
    st.write(
        "A production-ready e-commerce support chatbot powered by Agentic AI, retrieval-augmented generation, "
        "conversation memory, and simple tools."
    )
    st.subheader("Topics Covered")
    st.write(
        "- Products and sizing\n"
        "- Delivery timelines\n"
        "- Returns and refunds\n"
        "- Payment methods\n"
        "- Offers and policies"
    )
    st.code(st.session_state.thread_id, language=None)
    if st.button("New Chat", use_container_width=True):
        reset_conversation()


st.title("🛍️ AI Sales Assistant (E-commerce)")
st.caption("Ask about products, delivery, returns, or anything!")
st.markdown('<div class="assistant-shell">', unsafe_allow_html=True)

if not st.session_state.messages:
    st.info("Start a conversation to test retrieval, memory, and tool use.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


prompt = st.chat_input("Type your message...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = run_agent(prompt, st.session_state.thread_id)
        animate_markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.toast("Response generated")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")
st.caption("Built with ❤️ using Agentic AI")
