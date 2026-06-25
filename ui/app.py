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


def _query(question: str) -> dict:
    resp = httpx.post(
        f"{API_BASE}/query",
        json={"question": question, "evaluate": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Local RAG", page_icon="📚", layout="wide")
st.title("📚 Local RAG")

# ── Sidebar: upload + document list ─────────────────────────────────────────
with st.sidebar:
    st.header("Documents")

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
    if st.button("Refresh", use_container_width=True):
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

# ── Main: chat ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    st.write(f"• {src}")

# New question
question = st.chat_input("Ask a question about your documents…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = _query(question)
                answer = result["answer"]
                sources = result.get("sources", [])
            except httpx.HTTPStatusError as exc:
                answer = f"Query failed: {exc.response.text}"
                sources = []
            except httpx.TimeoutException:
                answer = "Request timed out. The model may still be loading — try again."
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander(f"Sources ({len(sources)})"):
                for src in sources:
                    st.write(f"• {src}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
