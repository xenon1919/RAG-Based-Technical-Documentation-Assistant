"""
Streamlit Frontend for the RAG Technical Documentation Assistant.

Provides a chat-style UI that connects to the FastAPI backend.

Run this AFTER starting the FastAPI server:
    Terminal 1: uvicorn app.main:app --reload
    Terminal 2: streamlit run streamlit_app.py
"""

import uuid

import httpx
import streamlit as st

API_BASE = "http://localhost:8000"
REQUEST_TIMEOUT = 90  # seconds — LLM calls can take a while


# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Documentation Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session State Initialization ──────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_available" not in st.session_state:
    st.session_state.api_available = False


# ── Helper Functions ──────────────────────────────────────────────────────────


def check_health() -> dict | None:
    """Ping the API health endpoint. Returns health data or None on failure."""
    try:
        resp = httpx.get(f"{API_BASE}/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def ingest_docs(directory: str | None = None, text: str | None = None) -> dict | None:
    """Trigger document ingestion via the API."""
    payload: dict = {}
    if directory:
        payload["directory"] = directory
    if text:
        payload["text"] = text
        payload["source_name"] = "pasted_content"
    try:
        resp = httpx.post(f"{API_BASE}/ingest", json=payload, timeout=120)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def list_docs() -> list[str]:
    """Fetch list of indexed sources from the API."""
    try:
        resp = httpx.get(f"{API_BASE}/documents", timeout=10)
        if resp.status_code == 200:
            return resp.json().get("sources", [])
    except Exception:
        pass
    return []


def send_feedback(question: str, answer: str, sentiment: str) -> None:
    """Send feedback to the API (fire and forget)."""
    try:
        httpx.post(
            f"{API_BASE}/feedback",
            json={
                "question": question,
                "answer": answer,
                "feedback": sentiment,
                "session_id": st.session_state.session_id,
            },
            timeout=5,
        )
    except Exception:
        pass


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📚 RAG Assistant")
    st.caption("Technical Documentation Q&A")

    st.divider()

    # ── API Health ────────────────────────────────────────────────────────────
    st.subheader("System Status")
    health = check_health()

    if health:
        st.session_state.api_available = True
        st.success("API Connected", icon="✅")

        stats = health.get("collection_stats", {})
        col1, col2 = st.columns(2)
        col1.metric("Chunks", stats.get("document_count", 0))
        col2.metric("Model", health.get("model", "N/A").split("-")[-1])
    else:
        st.session_state.api_available = False
        st.error("API Offline — Start the FastAPI server first", icon="❌")
        st.code("uvicorn app.main:app --reload")

    st.divider()

    # ── Document Ingestion ────────────────────────────────────────────────────
    st.subheader("Ingest Documents")

    ingest_mode = st.radio("Ingest mode:", ["Default (./docs)", "Custom directory", "Paste text"])

    if ingest_mode == "Default (./docs)":
        if st.button("Ingest ./docs folder", use_container_width=True, type="primary"):
            with st.spinner("Ingesting documents..."):
                result = ingest_docs()
            if result and "error" not in result:
                st.success(result.get("message", "Done!"))
            else:
                st.error(str(result))

    elif ingest_mode == "Custom directory":
        custom_dir = st.text_input("Directory path:", placeholder="/path/to/your/docs")
        if st.button("Ingest directory", use_container_width=True) and custom_dir:
            with st.spinner("Ingesting..."):
                result = ingest_docs(directory=custom_dir)
            if result and "error" not in result:
                st.success(result.get("message", "Done!"))
            else:
                st.error(str(result))

    else:  # Paste text
        pasted = st.text_area("Paste documentation here:", height=150)
        if st.button("Ingest text", use_container_width=True) and pasted:
            with st.spinner("Ingesting..."):
                result = ingest_docs(text=pasted)
            if result and "error" not in result:
                st.success(result.get("message", "Done!"))
            else:
                st.error(str(result))

    st.divider()

    # ── Indexed Sources ───────────────────────────────────────────────────────
    st.subheader("Indexed Sources")
    if st.button("Refresh", use_container_width=True):
        sources = list_docs()
        if sources:
            for src in sources:
                st.caption(f"📄 {src}")
        else:
            st.info("No documents indexed yet")

    st.divider()

    # ── Session Management ────────────────────────────────────────────────────
    st.subheader("Session")
    st.caption(f"ID: `{st.session_state.session_id[:12]}...`")
    if st.button("New Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Built with LangGraph + Gemini + ChromaDB")


# ── Main Chat Area ────────────────────────────────────────────────────────────

st.title("RAG Technical Documentation Assistant")
st.caption(
    "Ask questions about **FastAPI**, **LangChain**, and **LangGraph** documentation. "
    "The system retrieves relevant chunks, grades them for relevance, and generates "
    "a grounded answer with citations."
)

# ── Display chat history ──────────────────────────────────────────────────────

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("meta"):
            meta = msg["meta"]

            # Sources expander
            sources = meta.get("sources", [])
            if sources:
                with st.expander(f"📎 Sources ({len(sources)})", expanded=False):
                    for src in sources:
                        st.caption(f"• {src}")

            # Metadata badges
            cols = st.columns(4)
            cols[0].caption(f"🏷️ {meta.get('question_type', 'N/A')}")
            grounded = meta.get("is_grounded", False)
            cols[1].caption(f"{'✅' if grounded else '⚠️'} {'Grounded' if grounded else 'Unverified'}")
            score = meta.get("hallucination_score", 0)
            cols[2].caption(f"🎯 Score: {score:.2f}")
            cols[3].caption(f"⏱️ {meta.get('time_ms', 0):.0f}ms")

            # Feedback buttons
            if msg.get("question"):
                fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 6])
                if fb_col1.button("👍", key=f"pos_{i}"):
                    send_feedback(msg["question"], msg["content"], "positive")
                    st.toast("Thanks for the positive feedback!")
                if fb_col2.button("👎", key=f"neg_{i}"):
                    send_feedback(msg["question"], msg["content"], "negative")
                    st.toast("Thanks! We'll work on improving this.")

# ── Chat Input ────────────────────────────────────────────────────────────────

if prompt := st.chat_input(
    "Ask about FastAPI, LangChain, LangGraph...",
    disabled=not st.session_state.api_available,
):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer from API
    with st.chat_message("assistant"):
        with st.spinner("Searching documentation and generating answer..."):
            try:
                resp = httpx.post(
                    f"{API_BASE}/query",
                    json={
                        "question": prompt,
                        "session_id": st.session_state.session_id,
                    },
                    timeout=REQUEST_TIMEOUT,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "No answer generated.")
                    sources = data.get("sources", [])

                    st.markdown(answer)

                    if sources:
                        with st.expander(f"📎 Sources ({len(sources)})", expanded=False):
                            for src in sources:
                                st.caption(f"• {src}")

                    meta = {
                        "question_type": data.get("question_type", "N/A"),
                        "is_grounded": data.get("is_grounded", False),
                        "hallucination_score": data.get("hallucination_score", 0),
                        "time_ms": data.get("processing_time_ms", 0),
                        "sources": sources,
                    }

                    cols = st.columns(4)
                    cols[0].caption(f"🏷️ {meta['question_type']}")
                    grounded = meta["is_grounded"]
                    cols[1].caption(f"{'✅' if grounded else '⚠️'} {'Grounded' if grounded else 'Unverified'}")
                    cols[2].caption(f"🎯 Score: {meta['hallucination_score']:.2f}")
                    cols[3].caption(f"⏱️ {meta['time_ms']:.0f}ms")

                    if data.get("retry_count", 0) > 0:
                        st.info(f"💡 Required {data['retry_count']} query rewrite(s) to find relevant docs")

                    # Store in history with metadata
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "question": prompt,
                            "meta": meta,
                        }
                    )

                else:
                    error_detail = resp.json().get("detail", resp.text)
                    st.error(f"API Error {resp.status_code}: {error_detail}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Error: {error_detail}"}
                    )

            except httpx.TimeoutException:
                msg = "Request timed out. The LLM may be slow. Try again or simplify your question."
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})

            except httpx.ConnectError:
                msg = "Cannot connect to API. Ensure `uvicorn app.main:app --reload` is running."
                st.error(msg)

            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
