"""
Unit tests for RAG pipeline components (no API key or PDF required).

- Chunking: chunk_text produces correct number of chunks and overlap.
- Context: build_context truncates at max_chars and joins with separator.
- FAISS: build_faiss_index + search_faiss return correct nearest-neighbor indices.
"""

import numpy as np
import pytest

from rag.parsers import chunk_text
from rag.llm import build_context
from rag.embed_faiss import build_faiss_index, search_faiss


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def test_chunk_text_empty_string_returns_empty_list():
    assert chunk_text("", chunk_size=800, chunk_overlap=150) == []


def test_chunk_text_short_text_one_chunk():
    text = "Short document."
    chunks = chunk_text(text, chunk_size=800, chunk_overlap=150)
    assert len(chunks) == 1
    assert chunks[0] == "Short document."


def test_chunk_text_respects_chunk_size_and_overlap():
    # 1000 chars, size=400, overlap=100 -> step=300
    # Chunk 1: 0:400, Chunk 2: 300:700, Chunk 3: 600:1000, Chunk 4: 900:1000 (1300 > 1000 so 900:1000)
    text = "a" * 1000
    chunks = chunk_text(text, chunk_size=400, chunk_overlap=100)
    assert len(chunks) >= 2
    assert all(len(c) <= 400 for c in chunks)
    # First chunk full size (minus strip: no spaces so unchanged)
    assert len(chunks[0]) == 400
    # Consecutive chunks overlap by 100: chunk2 starts at 300, so chars 300-399 are in both
    assert chunks[0][300:400] == chunks[1][0:100]


def test_chunk_text_whitespace_only_stripped_away():
    text = "  \n  content  \n  "
    chunks = chunk_text(text, chunk_size=800, chunk_overlap=150)
    assert len(chunks) == 1
    assert chunks[0] == "content"


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------

def test_build_context_empty_list_returns_empty_string():
    assert build_context([], max_chars=4000) == ""


def test_build_context_single_chunk():
    chunks = ["Only one chunk here."]
    out = build_context(chunks, max_chars=4000)
    assert out == "Only one chunk here."


def test_build_context_joins_with_separator():
    chunks = ["First", "Second", "Third"]
    out = build_context(chunks, max_chars=4000)
    assert out == "First\n\n---\n\nSecond\n\n---\n\nThird"


def test_build_context_truncates_at_max_chars():
    long_chunk = "x" * 500
    chunks = [long_chunk, long_chunk, long_chunk]  # 1500 chars total
    out = build_context(chunks, max_chars=600)
    # Should include first chunk (500) and stop before adding second (would exceed 600)
    assert out == long_chunk
    assert len(out) == 500


def test_build_context_strips_chunks():
    chunks = ["  a  ", "  b  "]
    out = build_context(chunks, max_chars=4000)
    assert out == "a\n\n---\n\nb"


# ---------------------------------------------------------------------------
# FAISS (with dummy vectors, no API)
# ---------------------------------------------------------------------------

def test_build_faiss_index_and_search_returns_nearest():
    dim = 4
    # Three vectors: v0=(1,0,0,0), v1=(0,1,0,0), v2=(0,0,1,0)
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)
    index = build_faiss_index(embeddings)
    assert index.ntotal == 3
    assert index.d == dim

    # Query identical to first vector -> nearest is index 0
    query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    distances, indices = search_faiss(index, query, top_k=2)
    assert indices[0] == 0
    assert distances[0] == pytest.approx(0.0, abs=1e-5)
    assert indices[1] in (1, 2)  # second nearest is one of the others


def test_search_faiss_accepts_1d_query():
    embeddings = np.random.randn(5, 8).astype(np.float32)
    index = build_faiss_index(embeddings)
    query_1d = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    distances, indices = search_faiss(index, query_1d, top_k=2)
    assert indices.shape == (2,)
    assert distances.shape == (2,)
