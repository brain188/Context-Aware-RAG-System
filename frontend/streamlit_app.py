"""Streamlit UI for the RAG API.

Talks to the API over HTTP only (no imports from app/). Run with:
    streamlit run frontend/streamlit_app.py
Set RAG_API_URL if the API is not on http://localhost:8000.
"""

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000").rstrip("/")
# Answers stream for a while; only connecting should fail fast.
_TIMEOUT = httpx.Timeout(connect=5.0, read=180.0, write=10.0, pool=5.0)

_STATUS_NOTES = {
    "low_confidence": ("warning", "Low confidence: check the citations."),
    "not_covered": ("info", "Not covered in your library."),
}


class ApiError(Exception):
    """The API returned an error envelope or could not be reached."""


def _error_text(payload: dict[str, Any]) -> str:
    return payload.get("detail") or payload.get("error") or "unknown error"


@st.cache_data(ttl=60, show_spinner=False)
def fetch_library() -> list[dict[str, Any]]:
    """Indexed books with their chapters."""
    response = httpx.get(f"{API_URL}/api/library", timeout=_TIMEOUT)
    if response.status_code != 200:
        raise ApiError(_error_text(response.json()))
    return response.json()["books"]


@st.cache_data(show_spinner=False)
def fetch_chunk(chunk_id: str) -> dict[str, Any] | None:
    """A cited chunk's stored text and location, or None if it is gone."""
    response = httpx.get(f"{API_URL}/api/chunks/{chunk_id}", timeout=_TIMEOUT)
    return response.json() if response.status_code == 200 else None


def stream_query(payload: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """POST the question and yield (event, data) pairs from the SSE stream."""
    with httpx.stream("POST", f"{API_URL}/api/query", json=payload, timeout=_TIMEOUT) as response:
        if response.status_code != 200:
            response.read()
            raise ApiError(_error_text(response.json()))
        event, data_lines = "message", []
        for line in response.iter_lines():
            if not line:  # a blank line ends one event
                if data_lines:
                    yield event, json.loads("\n".join(data_lines))
                event, data_lines = "message", []
            elif line.startswith(":"):
                continue  # keep-alive comment
            elif line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())


def render_citations(citations: list[dict[str, Any]]) -> None:
    """One expander per citation, showing the stored source text to check it against."""
    for citation in citations:
        with st.expander(f"{citation['marker']} {citation['label']}"):
            chunk = fetch_chunk(citation["chunk_id"])
            if chunk is None:
                st.caption("This chunk is no longer in the index (the book was re-ingested).")
                continue
            meta = chunk["metadata"]
            st.caption(f"Section: {meta['section']} · relevance {citation['score']:.2f}")
            st.markdown(chunk["text"])


def render_answer(message: dict[str, Any]) -> None:
    """Show an assistant message with its status note and citations."""
    st.markdown(message["content"])
    note = _STATUS_NOTES.get(message.get("status", ""))
    if note:
        getattr(st, note[0])(note[1])
    if message.get("cache_hit"):
        st.caption("Served from cache.")
    render_citations(message.get("citations", []))


def sidebar_filters(books: list[dict[str, Any]]) -> dict[str, Any]:
    """Book / chapter / page-range filters; returns the QueryRequest fields to send."""
    st.sidebar.header("Search scope")
    titles = [b["title"] for b in books]
    chosen = st.sidebar.multiselect("Books", titles, help="Leave empty to search every book.")
    filters: dict[str, Any] = {"book_filter": chosen or None}
    if len(chosen) == 1:
        chapters = next(b["chapters"] for b in books if b["title"] == chosen[0])
        chapter = st.sidebar.selectbox("Chapter", ["(any)", *chapters])
        filters["chapter_filter"] = None if chapter == "(any)" else chapter
    if st.sidebar.checkbox("Limit page range"):
        page_from = st.sidebar.number_input("From page", min_value=1, value=1, step=1)
        page_to = st.sidebar.number_input("To page", min_value=1, value=max(page_from, 50), step=1)
        filters["page_from"], filters["page_to"] = int(page_from), int(page_to)
    if st.sidebar.button("New conversation"):
        st.session_state.messages = []
        st.session_state.session_id = None
    return filters


def ask(question: str, filters: dict[str, Any]) -> None:
    """Stream one answer into the chat and store it in the session history."""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    payload = {"question": question, "session_id": st.session_state.session_id, **filters}
    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed = ""
        try:
            for event, data in stream_query(payload):
                if event == "token":
                    streamed += data["token"]
                    placeholder.markdown(streamed + "▌")
                elif event == "answer":
                    placeholder.empty()
                    st.session_state.session_id = data["session_id"]
                    message = {
                        "role": "assistant",
                        "content": data["text"],
                        "citations": data["citations"],
                        "status": data["status"],
                        "cache_hit": data["cache_hit"],
                    }
                    render_answer(message)
                    st.session_state.messages.append(message)
                elif event == "error":
                    placeholder.empty()
                    st.error(f"The query failed: {_error_text(data)}")
        except (ApiError, httpx.HTTPError) as exc:
            placeholder.empty()
            st.error(f"Could not get an answer: {exc}")


def main() -> None:
    st.set_page_config(page_title="Library RAG", page_icon="📚", layout="wide")
    st.title("Ask your library")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("session_id", None)

    try:
        books = fetch_library()
    except (ApiError, httpx.HTTPError) as exc:
        st.error(f"Cannot reach the API at {API_URL}: {exc}")
        st.stop()
    if not books:
        st.info("No books are indexed yet. Run the ingest pipeline first.")
    filters = sidebar_filters(books)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_answer(message)
            else:
                st.markdown(message["content"])

    if question := st.chat_input("Ask a question about your books"):
        ask(question, filters)


if __name__ == "__main__":  # Streamlit runs the script as __main__
    main()
