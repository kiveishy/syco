"""Simple chat UI to test the RAG.   Run:  streamlit run ui/app.py"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import chat, config, embeddings, retrieval, store  # noqa: E402
from rag.llm import get_llm  # noqa: E402

st.set_page_config(page_title="Syco – clinical knowledge assistant", page_icon="💬", layout="wide")

# Hebrew/English mixed text: let each paragraph pick its own direction
st.markdown("""
<style>
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li,
[data-testid="stExpander"] p, .src-text { unicode-bidi: plaintext; text-align: start; }
.src-text { white-space: pre-wrap; font-size: 0.9rem; border-inline-start: 3px solid #ccc; padding-inline-start: .6rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading embedding & reranker models (first time downloads ~4 GB)...")
def warm_up() -> bool:
    embeddings.embed(["warm up"])
    if config.USE_RERANKER:
        embeddings.rerank("warm up", ["warm up"])
    return True


@st.cache_resource
def llm():
    return get_llm()


def books() -> dict[str, str]:
    if not config.MANIFEST_PATH.exists():
        return {}
    m = json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
    return {v["book_id"]: v["title"] for v in m.values() if v.get("indexed")}


def show_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**[{s['n']}] {s['label']}** · relevance {s['score']:.2f}")
            for m in s["matched"]:
                st.markdown(f"<div class='src-text'>{html.escape(m)}</div>", unsafe_allow_html=True)
            with st.popover("Full section sent to the model"):
                st.markdown(f"<div class='src-text'>{html.escape(s['text'])}</div>", unsafe_allow_html=True)


# ---- sidebar ------------------------------------------------------------------
with st.sidebar:
    st.header("Syco RAG prototype")
    n_chunks = store.count()
    st.caption(f"LLM backend: **{config.LLM_BACKEND}** · model: `{config.LLM_MODEL}`")
    st.caption(f"Indexed chunks: **{n_chunks}**")
    all_books = books()
    selected = st.multiselect("Limit to books", options=list(all_books), format_func=lambda b: all_books[b])
    retrieval_only = st.toggle("Retrieval only (no LLM)", help="Show which passages the RAG finds, without calling Claude.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    if config.LLM_BACKEND == "cli":
        st.warning("CLI backend uses your personal claude.ai login — for personal prototyping only.", icon="⚠️")

if n_chunks == 0:
    st.info("The index is empty. Run `python scripts/ingest.py` first.")
    st.stop()

warm_up()
retrieval.reload()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_sources(msg.get("sources", []))

question = st.chat_input("Ask about the books… / שאלו שאלה על הספרים…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
    with st.chat_message("assistant"):
        if retrieval_only:
            with st.spinner("Searching…"):
                srcs = retrieval.retrieve(question, selected or None)
            answer = f"Found {len(srcs)} passages (retrieval-only mode)." if srcs else "No passages found."
            st.markdown(answer)
        else:
            with st.spinner("Searching the library and asking Claude…"):
                turn = chat.ask(question, history, llm(), selected or None)
                srcs = turn.sources
                try:
                    answer = st.write_stream(turn.answer_stream)
                except Exception as e:  # noqa: BLE001
                    answer = f"⚠️ LLM error: {e}"
                    st.error(answer)
        src_dicts = [{"n": s.n, "label": s.label, "score": s.score, "matched": s.matched, "text": s.text} for s in srcs]
        show_sources(src_dicts)
    st.session_state.messages.append({"role": "assistant", "content": answer if isinstance(answer, str) else str(answer),
                                      "sources": src_dicts})
