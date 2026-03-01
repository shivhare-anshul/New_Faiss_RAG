#!/usr/bin/env python3
"""
Pre-ingestion script: parse the PDF, embed all chunks, and save FAISS index + chunks to disk.

Run this ONCE before sharing/zipping the repo. After ingestion, run.py and app.py can be
used without any PDF parsing or embedding — they load directly from the saved index.

Builds indexes for both parsers (or a specific one via --parser):
  data/faiss_index/pymupdf/  - index.faiss, chunks.json, manifest.json
  data/faiss_index/docling/  - index.faiss, chunks.json, manifest.json

Usage:
    python ingest.py                     # ingest with both parsers
    python ingest.py --parser pymupdf    # ingest only with PyMuPDF
    python ingest.py --parser docling    # ingest only with Docling
    python ingest.py --rebuild           # re-ingest even if cache exists
"""
import argparse
import logging
import sys
import time
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
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def ingest_one(pdf_path: Path, parser: str, rebuild: bool) -> dict:
    """
    Run ingestion for one parser. Returns a dict of metrics.
    """
    from rag.parsers import load_and_chunk_pdf
    from rag.embed_faiss import (
        embed_chunks, build_faiss_index, load_faiss_index, save_faiss_index
    )

    cache_dir = _PROJECT_ROOT / "data" / "faiss_index" / parser
    parsed_output_dir = _PROJECT_ROOT / "data" / "parsed_output"

    # Check if cache already exists and is valid
    if not rebuild:
        cached = load_faiss_index(cache_dir, pdf_path)
        if cached is not None:
            index, chunks = cached
            logger.info("[%s] Cache valid: %d chunks, %d vectors. Skipping ingestion.", parser, len(chunks), index.ntotal)
            return {"parser": parser, "status": "cache_loaded", "chunks": len(chunks), "parse_time": 0, "embed_time": 0}

    logger.info("[%s] Starting ingestion...", parser)
    t_parse = time.time()
    chunks = load_and_chunk_pdf(
        str(pdf_path),
        parser=parser,
        save_parsed_output_dir=str(parsed_output_dir),
    )
    parse_time = time.time() - t_parse
    logger.info("[%s] Parse done: %d chunks in %.2fs", parser, len(chunks), parse_time)

    t_embed = time.time()
    embeddings = embed_chunks(chunks)
    index = build_faiss_index(embeddings)
    embed_time = time.time() - t_embed
    logger.info("[%s] Embed done: %d vectors (dim=%d) in %.2fs", parser, index.ntotal, embeddings.shape[1], embed_time)

    save_faiss_index(index, chunks, cache_dir, pdf_path)
    logger.info("[%s] Ingestion complete. Cache saved to %s", parser, cache_dir)

    return {
        "parser": parser,
        "status": "ingested",
        "chunks": len(chunks),
        "parse_time": round(parse_time, 2),
        "embed_time": round(embed_time, 2),
        "total_time": round(parse_time + embed_time, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Pre-ingest PDF with FAISS index for RAG")
    parser.add_argument("--pdf", type=str, default=None, help="Path to PDF (default: auto-detect in data/)")
    parser.add_argument("--parser", type=str, default="both", choices=["both", "pymupdf", "docling"],
                        help="Which parser to use for ingestion (default: both)")
    parser.add_argument("--rebuild", action="store_true", help="Re-ingest even if cache exists")
    args = parser.parse_args()

    # Find PDF
    from run import find_pdf
    try:
        pdf_path = find_pdf(args.pdf)
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)
    logger.info("PDF: %s", pdf_path)

    parsers = ["pymupdf", "docling"] if args.parser == "both" else [args.parser]
    results = []

    for p in parsers:
        print(f"\n{'='*60}")
        print(f" Ingesting with parser: {p}")
        print(f"{'='*60}")
        result = ingest_one(pdf_path, p, args.rebuild)
        results.append(result)

    print("\n" + "="*60)
    print(" INGESTION SUMMARY")
    print("="*60)
    for r in results:
        if r["status"] == "cache_loaded":
            print(f"  [{r['parser']}] Already cached: {r['chunks']} chunks (no re-ingestion)")
        else:
            print(f"  [{r['parser']}] Done: {r['chunks']} chunks | parse={r['parse_time']}s | embed={r['embed_time']}s | total={r['total_time']}s")
    print()
    print("Indexes saved to:")
    for p in parsers:
        print(f"  data/faiss_index/{p}/  (index.faiss, chunks.json, manifest.json)")
    print()
    print("You can now:")
    print("  python run.py                      # query with PyMuPDF index (default)")
    print("  python run.py --parser docling     # query with Docling index")
    print("  streamlit run app.py               # open UI to select parser + query")


if __name__ == "__main__":
    main()
