#!/usr/bin/env python3
"""
AI Context Builder with RAG: parse PDF, chunk, embed with FAISS, retrieve chunks, answer via LLM.

Supports two parsers (--parser flag):
  pymupdf  - PyMuPDF: fast ~1.7s for 646 pages (default)
  docling  - docling_parse C++ engine: thorough ~22s, same Docling text layer

Ingestion cache is stored per parser in data/faiss_index/<parser>/ so both can coexist.
On the first run (or --rebuild), the pipeline runs fully and saves the cache.
On later runs with the same PDF, the cached index is loaded (skip parse + embed).

Usage:
    python run.py
    python run.py --parser docling
    python run.py --query "Who is José Rizal?"
    python run.py --pdf path/to/file.pdf
    python run.py --rebuild
"""
import argparse
import logging
import sys
from pathlib import Path
import os

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

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_PDF_NAME = "philippine_history.pdf"
DEFAULT_QUERY = "When did the EDSA People Power Revolution happen?"


def find_pdf(pdf_path: str | None) -> Path:
    """
    Resolve path to the PDF.
    If pdf_path is None, looks in:
      1. data/philippine_history.pdf
      2. ./philippine_history.pdf
      3. Single PDF in data/ (any filename)
    """
    if pdf_path:
        p = Path(pdf_path).resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"PDF not found: {p}")
    for candidate in [_PROJECT_ROOT / "data" / DEFAULT_PDF_NAME, _PROJECT_ROOT / DEFAULT_PDF_NAME]:
        if candidate.is_file():
            return candidate
    data_dir = _PROJECT_ROOT / "data"
    if data_dir.is_dir():
        pdfs = list(data_dir.glob("*.pdf"))
        if len(pdfs) == 1:
            return pdfs[0]
    raise FileNotFoundError(
        f"Could not find {DEFAULT_PDF_NAME}. Place it in the project root or in data/."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG pipeline: PDF → FAISS → LLM answer")
    parser.add_argument("--pdf", type=str, default=None, help="Path to PDF (default: auto-detect in data/)")
    parser.add_argument("--query", type=str, default=None, help=f"Question (default: '{DEFAULT_QUERY}')")
    parser.add_argument("--parser", type=str, default="pymupdf", choices=["pymupdf", "docling"],
                        help="PDF parser: pymupdf (fast, ~2s) or docling (thorough, ~22s). Default: pymupdf")
    parser.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve (default: 5)")
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size chars (default: 800)")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap chars (default: 150)")
    parser.add_argument("--rebuild", action="store_true", help="Ignore cache; re-ingest from scratch")
    args = parser.parse_args()

    # --- STEP 0: Resolve PDF ---
    logger.info("STEP 0: Resolving PDF path")
    pdf_path = find_pdf(args.pdf)
    logger.info("STEP 0 done: PDF = %s", pdf_path)

    # Cache dir is per-parser: data/faiss_index/pymupdf/ or data/faiss_index/docling/
    faiss_index_dir = _PROJECT_ROOT / "data" / "faiss_index" / args.parser

    import numpy as np
    from rag.embed_faiss import (
        build_faiss_index, embed_chunks, get_embedding_client,
        load_faiss_index, save_faiss_index, search_faiss,
    )

    # --- Try cache ---
    index = chunks = None
    if not args.rebuild:
        cached = load_faiss_index(faiss_index_dir, pdf_path)
        if cached is not None:
            index, chunks = cached
            logger.info("Using cached FAISS index (%s): %d chunks (skip STEP 1 & 2)", args.parser, len(chunks))

    # --- STEP 1: Parse + chunk ---
    if index is None or chunks is None:
        logger.info("STEP 1: Parse and chunk PDF (parser=%s)", args.parser)
        from rag.parsers import load_and_chunk_pdf
        parsed_output_dir = _PROJECT_ROOT / "data" / "parsed_output"
        chunks = load_and_chunk_pdf(
            str(pdf_path),
            parser=args.parser,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            save_parsed_output_dir=str(parsed_output_dir),
        )
        logger.info("STEP 1 done: %d chunks", len(chunks))

        # --- STEP 2: Embed + build FAISS + save cache ---
        logger.info("STEP 2: Embed %d chunks and build FAISS index", len(chunks))
        embeddings = embed_chunks(chunks)
        index = build_faiss_index(embeddings)
        logger.info("STEP 2 done: index ntotal=%d, dim=%d", index.ntotal, embeddings.shape[1])
        save_faiss_index(index, chunks, faiss_index_dir, pdf_path)

    client = get_embedding_client()

    # --- STEP 3: Embed query + search ---
    query = args.query or DEFAULT_QUERY
    logger.info("STEP 3: Query = %s", query[:80])
    print(f"\nUser Query: {query}")

    query_emb = client.embeddings.create(input=[query], model="text-embedding-3-small")
    query_vec = np.array([query_emb.data[0].embedding], dtype=np.float32)
    _, indices = search_faiss(index, query_vec, min(args.top_k, len(chunks)))
    retrieved = [chunks[i] for i in indices if i < len(chunks)]
    logger.info("STEP 3 done: retrieved %d chunks", len(retrieved))

    print("\nRetrieved Chunks:")
    for r in retrieved:
        snippet = (r[:200] + "...") if len(r) > 200 else r
        print(f'  - "{snippet}"')

    # --- STEP 4: Build context + LLM ---
    logger.info("STEP 4: Build context and call LLM")
    from rag.llm import build_context, get_llm_response
    context = build_context(retrieved)
    response = get_llm_response(query, context)
    logger.info("STEP 4 done: response length=%d chars", len(response))
    print("\nLLM Response:")
    print(response)
    print()


if __name__ == "__main__":
    main()
