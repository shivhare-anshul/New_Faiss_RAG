"""
Embed text chunks with OpenAI and build a FAISS index for similarity search.

Data flow:
  chunks: list[str] -> embed_chunks() -> embeddings: np.ndarray (n, 1536)
  embeddings -> build_faiss_index() -> index: faiss.IndexFlatL2
  query: str -> embed -> query_vec (1, 1536) -> search_faiss(index, query_vec, top_k) -> (distances, indices)

Persistence: save_faiss_index / load_faiss_index to reuse index + chunks across runs.
Separate index directories are used per parser: data/faiss_index/pymupdf/ and data/faiss_index/docling/
"""

import json
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "text-embedding-3-small"  # OpenAI; output dimension: 1536


def get_embedding_client():
    """
    Return OpenAI client. Reads TEAMIFIED_OPENAI_API_KEY or OPENAI_API_KEY from environment.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("OpenAI package required: pip install openai")
    api_key = os.environ.get("TEAMIFIED_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set TEAMIFIED_OPENAI_API_KEY (or OPENAI_API_KEY) in .env or environment.")
    return OpenAI(api_key=api_key)


def embed_chunks(chunks: list[str], model: str = DEFAULT_EMBED_MODEL) -> np.ndarray:
    """
    Embed a list of text chunks with OpenAI. Returns one vector per chunk.

    Args:
        chunks: list[str]. E.g. ["On February 25, 1986...", "José Rizal was...", ...]
        model: OpenAI embedding model. Default "text-embedding-3-small" (dim=1536).

    Returns:
        np.ndarray shape (len(chunks), 1536), dtype float32.
        Row i is the embedding for chunks[i].
    """
    client = get_embedding_client()
    batch_size = 100
    all_embeddings = []
    n_batches = (len(chunks) + batch_size - 1) // batch_size
    logger.info("[embed_faiss] Embedding %d chunks in %d batch(es), model=%s", len(chunks), n_batches, model)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        batch_num = i // batch_size + 1
        logger.debug("[embed_faiss] Batch %d/%d (%d chunks)", batch_num, n_batches, len(batch))
        response = client.embeddings.create(input=batch, model=model)
        order = sorted(response.data, key=lambda x: x.index)
        all_embeddings.extend([d.embedding for d in order])
    logger.info("[embed_faiss] Embedding done: shape=(%d, %d)", len(all_embeddings), len(all_embeddings[0]) if all_embeddings else 0)
    return np.array(all_embeddings, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray):
    """
    Build a FAISS IndexFlatL2 for exact L2 nearest-neighbor search.

    Args:
        embeddings: np.ndarray shape (n, dim), dtype float32. E.g. (2419, 1536).

    Returns:
        faiss.IndexFlatL2 with index.ntotal == n.
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("FAISS required: pip install faiss-cpu")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    logger.info("[embed_faiss] FAISS index built: ntotal=%d, dim=%d", index.ntotal, dim)
    return index


def search_faiss(index, query_embedding: np.ndarray, top_k: int = 5):
    """
    Search FAISS index for the top-k nearest chunks.

    Args:
        index: faiss.IndexFlatL2 (from build_faiss_index).
        query_embedding: np.ndarray shape (1, 1536) or (1536,), dtype float32.
        top_k: Number of nearest neighbors.

    Returns:
        (distances, indices): both np.ndarray of shape (top_k,).
        distances: L2 distances (lower = more similar).
        indices: int64 positions in original chunks list. E.g. [142, 891, 12, 445, 203].
    """
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    return distances[0], indices[0]


def save_faiss_index(index, chunks: list[str], cache_dir: Path, pdf_path: Path) -> None:
    """
    Save FAISS index and chunks to cache_dir (index.faiss, chunks.json, manifest.json).
    Call after ingestion so later runs can skip parse + embed.
    """
    import faiss
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(cache_dir / "index.faiss"))
    with open(cache_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=0)
    mtime = pdf_path.stat().st_mtime if pdf_path.exists() else 0
    with open(cache_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "pdf_path": str(pdf_path.resolve()),
            "pdf_mtime": mtime,
            "n_chunks": len(chunks),
            "embed_model": DEFAULT_EMBED_MODEL,
        }, f, indent=2)
    logger.info("[embed_faiss] Saved index and %d chunks to %s", len(chunks), cache_dir)


def load_faiss_index(cache_dir: Path, pdf_path: Path):
    """
    Load FAISS index and chunks from cache_dir if valid (PDF path + mtime match).

    Returns:
        (index, chunks) if cache valid; None otherwise.
    """
    try:
        import faiss
    except ImportError:
        return None
    cache_dir = Path(cache_dir)
    index_path = cache_dir / "index.faiss"
    chunks_path = cache_dir / "chunks.json"
    manifest_path = cache_dir / "manifest.json"
    if not all(p.is_file() for p in [index_path, chunks_path, manifest_path]):
        return None
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    current_mtime = pdf_path.stat().st_mtime if pdf_path.exists() else 0
    if manifest.get("pdf_path") != str(pdf_path.resolve()) or manifest.get("pdf_mtime") != current_mtime:
        logger.info("[embed_faiss] Cache invalid (PDF changed), will rebuild")
        return None
    index = faiss.read_index(str(index_path))
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    if len(chunks) != index.ntotal:
        logger.warning("[embed_faiss] Cache inconsistent (chunks=%d != ntotal=%d), will rebuild", len(chunks), index.ntotal)
        return None
    logger.info("[embed_faiss] Loaded index and %d chunks from %s", len(chunks), cache_dir)
    return index, chunks
