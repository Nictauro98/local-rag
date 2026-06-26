import os
import time

import httpx
import streamlit as st

API_BASE = os.getenv("RAG_API_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _upload(file) -> dict:
    resp = httpx.post(
        f"{API_BASE}/documents/upload",
        files={"file": (file.name, file.getvalue(), file.type or "application/octet-stream")},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _poll_status(filename: str, timeout: int = 120) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = httpx.get(f"{API_BASE}/documents/{filename}/status", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data["status"] in ("completed", "failed"):
            return data
        time.sleep(2)
    return {"filename": filename, "status": "timeout", "chunks": None}


def _list_documents() -> list[str]:
    resp = httpx.get(f"{API_BASE}/documents", timeout=10)
    resp.raise_for_status()
    return resp.json()["documents"]


def _query(question: str, evaluate: bool = False, max_retries: int = 2) -> dict:
    resp = httpx.post(
        f"{API_BASE}/query",
        json={"question": question, "evaluate": evaluate, "max_retries": max_retries},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def _get_history(limit: int = 50) -> list[dict]:
    resp = httpx.get(f"{API_BASE}/query/history", params={"limit": limit}, timeout=10)
    resp.raise_for_status()
    return resp.json()["items"]


# ---------------------------------------------------------------------------
# Eval panel helpers
# ---------------------------------------------------------------------------

_GREEN = "#28a745"
_YELLOW = "#ffc107"
_RED = "#dc3545"


def _score_color(score: float, warn: float = 0.5, good: float = 0.7) -> str:
    if score >= good:
        return _GREEN
    if score >= warn:
        return _YELLOW
    return _RED


def _render_eval_panel(ev: dict) -> None:
    flagged = ev.get("flagged", False)
    retries = ev.get("retries", 0)

    if flagged:
        st.error(
            "Evaluation: flagged — low-quality answer"
            + (f" (retried {retries}x)" if retries else "")
        )
    else:
        st.success("Evaluation: passed" + (f" (retried {retries}x)" if retries else ""))

    faith = ev.get("faithfulness", 0.0)
    ctx = ev.get("context_relevance", 0.0)
    grounded = ev.get("answer_grounded", False)

    col1, col2, col3 = st.columns(3)
    with col1:
        color = _score_color(faith)
        st.markdown(
            f"<div style='border-left:4px solid {color};padding-left:8px'>"
            f"<b>Faithfulness</b><br>{faith:.0%}</div>",
            unsafe_allow_html=True,
        )
    with col2:
        color = _score_color(ctx)
        st.markdown(
            f"<div style='border-left:4px solid {color};padding-left:8px'>"
            f"<b>Context relevance</b><br>{ctx:.0%}</div>",
            unsafe_allow_html=True,
        )
    with col3:
        color = _GREEN if grounded else _RED
        label = "Yes" if grounded else "No"
        st.markdown(
            f"<div style='border-left:4px solid {color};padding-left:8px'>"
            f"<b>Grounded</b><br>{label}</div>",
            unsafe_allow_html=True,
        )

    reasoning = ev.get("judge_reasoning")
    if reasoning:
        with st.expander("Judge reasoning"):
            st.caption(reasoning)

    tier_scores = ev.get("tier_scores") or {}
    if tier_scores:
        with st.expander("Tier scores"):
            for k, v in tier_scores.items():
                st.caption(f"{k}: {v:.3f}")


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Local RAG", page_icon="📚", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar: upload + document list + eval settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📚 Local RAG")

    st.subheader("Documents")
    uploaded = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])
    if uploaded is not None and st.button("Ingest", use_container_width=True):
        with st.spinner(f"Uploading {uploaded.name}…"):
            try:
                _upload(uploaded)
            except httpx.HTTPStatusError as exc:
                st.error(f"Upload failed: {exc.response.text}")
                st.stop()

        status_placeholder = st.empty()
        status_placeholder.info("Waiting for ingestion to start…")
        result = _poll_status(uploaded.name)
        status = result["status"]

        if status == "completed":
            chunks = result.get("chunks")
            msg = f"Ingested — {chunks} chunk(s) indexed." if chunks else "Ingestion complete."
            status_placeholder.success(msg)
        elif status == "failed":
            status_placeholder.error("Ingestion failed. Check worker logs.")
        else:
            status_placeholder.warning(f"Ingestion timed out (status: {status}).")

    st.divider()
    st.subheader("Indexed documents")
    if st.button("Refresh list", use_container_width=True):
        st.session_state.pop("doc_list", None)

    if "doc_list" not in st.session_state:
        try:
            st.session_state.doc_list = _list_documents()
        except Exception:
            st.session_state.doc_list = []

    docs = st.session_state.doc_list
    if docs:
        for doc in docs:
            st.caption(f"• {doc}")
    else:
        st.caption("No documents indexed yet.")

    st.divider()
    st.subheader("Evaluation")
    evaluate = st.toggle("Enable evaluation", value=False)
    max_retries = st.slider("Max retries", min_value=0, max_value=5, value=2) if evaluate else 2

# ---------------------------------------------------------------------------
# Process new question BEFORE rendering tabs so state is ready on re-render.
# st.chat_input() at module level (outside any container) renders as a sticky
# bar fixed to the bottom of the viewport — this is what makes it feel like a
# normal LLM chat UI.
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

question = st.chat_input("Ask a question about your documents…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Thinking…"):
        try:
            result = _query(question, evaluate=evaluate, max_retries=max_retries)
            answer = result["answer"]
            sources = result.get("sources", [])
            eval_result = result.get("eval_result")
        except httpx.HTTPStatusError as exc:
            answer = f"Query failed: {exc.response.text}"
            sources = []
            eval_result = None
        except httpx.TimeoutException:
            answer = "Request timed out. The model may still be loading — try again."
            sources = []
            eval_result = None

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "eval_result": eval_result}
    )

# ---------------------------------------------------------------------------
# Tabs: Chat | History
# ---------------------------------------------------------------------------

tab_chat, tab_history = st.tabs(["Chat", "History"])

with tab_chat:
    if not st.session_state.messages:
        st.markdown(
            "<div style='text-align:center;color:#888;margin-top:3rem'>"
            "Ask a question below to get started."
            "</div>",
            unsafe_allow_html=True,
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"Sources ({len(msg['sources'])})"):
                    for src in msg["sources"]:
                        st.write(f"• {src}")
            if msg.get("eval_result"):
                _render_eval_panel(msg["eval_result"])

with tab_history:
    col_left, col_right = st.columns([4, 1])
    with col_left:
        st.subheader("Query history")
    with col_right:
        history_limit = st.number_input(
            "Limit", min_value=1, max_value=500, value=50, label_visibility="collapsed"
        )
        if st.button("Refresh", use_container_width=True):
            st.session_state.pop("history_items", None)

    if "history_items" not in st.session_state:
        try:
            st.session_state.history_items = _get_history(limit=int(history_limit))
        except Exception as exc:
            st.session_state.history_items = []
            st.warning(f"Could not load history: {exc}")

    items = st.session_state.history_items
    if not items:
        st.info("No history yet. Ask a question with evaluation enabled.")
    else:
        for item in items:
            ts = item.get("timestamp", "")[:19].replace("T", " ")
            retries = item.get("retries", 0)
            ev = item.get("eval_result")
            flagged = ev.get("flagged", False) if ev else None

            badge = "🔴" if flagged is True else ("🟢" if flagged is False else "⚪")
            label = f"{badge} {ts} — {item['question'][:80]}"

            with st.expander(label):
                st.markdown(f"**Answer:** {item['answer']}")
                srcs = item.get("sources", [])
                if srcs:
                    st.caption("Sources: " + ", ".join(srcs))
                if retries:
                    st.caption(f"Retries: {retries}")
                if ev:
                    _render_eval_panel(ev)
