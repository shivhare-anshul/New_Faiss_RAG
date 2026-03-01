"""
Streamlit UI for the Philippine History RAG pipeline.

Run with:  streamlit run app.py

Features:
  - Select parser: PyMuPDF (fast ~2s) or Docling (thorough ~22s)
  - Type any question about Philippine History
  - Choose top-k chunks to retrieve
  - See Retrieved Chunks and LLM Response
  - Caches FAISS index per parser; first use of a parser triggers ingestion
"""
import os
import sys
import time
import logging
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)

import streamlit as st
import numpy as np

# -------------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Philippine History RAG",
    page_icon="📜",
    layout="wide",
)

st.title("📜 Philippine History — RAG Query Interface")
st.caption("Retrieval-Augmented Generation pipeline: PDF → FAISS → GPT-4o-mini")

# -------------------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    parser_choice = st.radio(
        "PDF Parser",
        options=["pymupdf", "docling"],
        format_func=lambda x: "PyMuPDF (fast, ~2s)" if x == "pymupdf" else "Docling (thorough, ~22s)",
        index=0,
        help="PyMuPDF is fast and reliable. Docling uses the same underlying text engine as the full Docling pipeline.",
    )

    top_k = st.slider("Top-K chunks to retrieve", min_value=1, max_value=10, value=5,
                       help="Number of most-relevant chunks to pass to the LLM as context.")

    st.markdown("---")
    st.markdown("**Ingestion cache**")
    for p in ["pymupdf", "docling"]:
        cache_dir = _PROJECT_ROOT / "data" / "faiss_index" / p
        status = "✅ cached" if (cache_dir / "index.faiss").exists() else "⬜ not ingested"
        st.markdown(f"`{p}`: {status}")

    if st.button("🔄 Rebuild selected index", help="Force re-ingestion for the selected parser"):
        st.session_state["rebuild"] = True
    else:
        st.session_state.setdefault("rebuild", False)

    st.markdown("---")
    st.markdown("**Sample queries**")
    sample_queries = [
        "When did the EDSA People Power Revolution happen?",
        "Who is José Rizal and why is he important?",
        "Tell me about the Spanish colonization of the Philippines.",
        "Who was Andres Bonifacio?",
        "What was the Treaty of Paris 1898?",
        "When did martial law begin under Ferdinand Marcos?",
    ]
    for q in sample_queries:
        if st.button(q, key=f"sample_{q[:20]}"):
            st.session_state["query_input"] = q

# -------------------------------------------------------------------------
# Helper: load or build index
# -------------------------------------------------------------------------

def get_or_build_index(parser: str, rebuild: bool = False):
    """
    Load FAISS index + chunks from cache, or build from scratch if needed.
    Returns (index, chunks, parse_time, embed_time, was_cached).
    """
    from rag.embed_faiss import load_faiss_index, save_faiss_index, embed_chunks, build_faiss_index
    from rag.parsers import load_and_chunk_pdf
    from run import find_pdf

    pdf_path = find_pdf(None)
    cache_dir = _PROJECT_ROOT / "data" / "faiss_index" / parser

    if not rebuild:
        cached = load_faiss_index(cache_dir, pdf_path)
        if cached is not None:
            index, chunks = cached
            return index, chunks, 0.0, 0.0, True

    # Full ingestion
    t0 = time.time()
    chunks = load_and_chunk_pdf(
        str(pdf_path),
        parser=parser,
        save_parsed_output_dir=str(_PROJECT_ROOT / "data" / "parsed_output"),
    )
    parse_time = time.time() - t0

    t1 = time.time()
    embeddings = embed_chunks(chunks)
    from rag.embed_faiss import build_faiss_index
    index = build_faiss_index(embeddings)
    embed_time = time.time() - t1

    save_faiss_index(index, chunks, cache_dir, pdf_path)
    return index, chunks, parse_time, embed_time, False


# -------------------------------------------------------------------------
# Query input
# -------------------------------------------------------------------------
st.subheader("Ask a question about Philippine History")

query = st.text_input(
    "Your question",
    value=st.session_state.get("query_input", "When did the EDSA People Power Revolution happen?"),
    placeholder="e.g. Who is José Rizal and why is he important?",
    key="query_input",
)

ask_col, rebuild_col = st.columns([4, 1])
with ask_col:
    ask_clicked = st.button("🔍 Ask", type="primary", use_container_width=True)

# -------------------------------------------------------------------------
# Run pipeline
# -------------------------------------------------------------------------
if ask_clicked and query.strip():
    rebuild = st.session_state.get("rebuild", False)
    if rebuild:
        st.session_state["rebuild"] = False

    with st.spinner(f"Loading index for **{parser_choice}** parser..."):
        try:
            index, chunks, parse_time, embed_time, was_cached = get_or_build_index(parser_choice, rebuild)
        except FileNotFoundError as e:
            st.error(f"**PDF not found.** {e}\n\nPlace `philippine_history.pdf` in the project root or `data/`.")
            st.stop()
        except Exception as e:
            st.error(f"**Ingestion failed:** {e}")
            st.stop()

    # Show ingestion info
    if was_cached:
        st.info(f"✅ Using cached index for **{parser_choice}** ({len(chunks):,} chunks)")
    else:
        st.success(
            f"✅ Ingestion done for **{parser_choice}**: {len(chunks):,} chunks | "
            f"parse={parse_time:.1f}s | embed={embed_time:.1f}s"
        )

    # Embed query + search
    from rag.embed_faiss import get_embedding_client, search_faiss
    client = get_embedding_client()

    with st.spinner("Embedding query and searching..."):
        t_q = time.time()
        qr = client.embeddings.create(input=[query], model="text-embedding-3-small")
        query_vec = np.array([qr.data[0].embedding], dtype=np.float32)
        _, indices = search_faiss(index, query_vec, min(top_k, len(chunks)))
        retrieved = [chunks[i] for i in indices if i < len(chunks)]
        query_time = time.time() - t_q

    # LLM
    from rag.llm import build_context, get_llm_response
    with st.spinner("Generating answer..."):
        t_llm = time.time()
        context = build_context(retrieved)
        response = get_llm_response(query, context)
        llm_time = time.time() - t_llm

    # -----------------------------------------------------------------------
    # Display results
    # -----------------------------------------------------------------------
    st.markdown("---")

    # Query
    st.markdown(f"### 🔎 User Query")
    st.info(query)

    # Timing
    col1, col2, col3 = st.columns(3)
    col1.metric("Parser", parser_choice)
    col2.metric("Chunks retrieved", len(retrieved))
    col3.metric("Query + LLM time", f"{query_time + llm_time:.1f}s")

    # Retrieved chunks
    st.markdown("### 📄 Retrieved Chunks")
    for i, chunk in enumerate(retrieved, 1):
        with st.expander(f"Chunk {i} — {len(chunk)} chars"):
            st.text(chunk)

    # LLM response
    st.markdown("### 💬 LLM Response")
    st.success(response)

    # Footer info
    st.caption(
        f"Parser: {parser_choice} | Top-K: {top_k} | Model: gpt-4o-mini | "
        f"Context: {len(context)} chars | Query embed: {query_time:.2f}s | LLM: {llm_time:.2f}s"
    )

elif ask_clicked and not query.strip():
    st.warning("Please enter a question.")
