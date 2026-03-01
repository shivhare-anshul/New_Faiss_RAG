#!/usr/bin/env python3
"""
Run RAG on 13 fixed queries and print Retrieved Chunks + LLM Response for each.
Used for evaluating RAG quality and comparing parsers.

Usage:
    python run_queries.py                     # PyMuPDF (default)
    python run_queries.py --parser docling    # Docling parser
    python run_queries.py --parser both       # run all queries for BOTH parsers, side-by-side
"""
import argparse
import logging
import os
import sys
import time
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

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ASSIGNMENT_QUERIES = [
    "When did the EDSA People Power Revolution happen?",
    "Who is José Rizal and why is he important?",
    "Tell me about the Spanish colonization of the Philippines.",
]

EXTRA_QUERIES = [
    "What was the First Philippine Republic?",
    "When did the Philippines gain independence from the United States?",
    "Who was Andres Bonifacio and what role did he play?",
    "What is the significance of the Cry of Pugad Lawin?",
    "When did the Philippine-American War start and end?",
    "Who was Emilio Aguinaldo?",
    "What was the Treaty of Paris 1898 and how did it affect the Philippines?",
    "Describe the Katipunan and its goals.",
    "When did martial law begin under Ferdinand Marcos?",
    "What is the EDSA Revolution and why is it important to Philippine history?",
]

ALL_QUERIES = ASSIGNMENT_QUERIES + EXTRA_QUERIES


def run_queries_for_parser(parser: str, pdf_path, rebuild: bool = False) -> None:
    """Run all 13 queries using the specified parser's FAISS index."""
    from rag.parsers import load_and_chunk_pdf
    from rag.embed_faiss import (
        embed_chunks, build_faiss_index, get_embedding_client,
        load_faiss_index, save_faiss_index, search_faiss,
    )
    from rag.llm import build_context, get_llm_response
    import numpy as np

    cache_dir = _PROJECT_ROOT / "data" / "faiss_index" / parser
    index = chunks = None
    if not rebuild:
        cached = load_faiss_index(cache_dir, pdf_path)
        if cached is not None:
            index, chunks = cached
            logger.info("[%s] Using cached index: %d chunks", parser, len(chunks))

    if index is None or chunks is None:
        logger.info("[%s] Ingesting PDF...", parser)
        t0 = time.time()
        chunks = load_and_chunk_pdf(
            str(pdf_path),
            parser=parser,
            save_parsed_output_dir=str(_PROJECT_ROOT / "data" / "parsed_output"),
        )
        embeddings = embed_chunks(chunks)
        index = build_faiss_index(embeddings)
        save_faiss_index(index, chunks, cache_dir, pdf_path)
        logger.info("[%s] Ingestion done: %d chunks, %.1fs", parser, len(chunks), time.time() - t0)

    client = get_embedding_client()

    print(f"\n{'='*72}")
    print(f" PARSER: {parser.upper()} | {len(chunks)} chunks | {len(ALL_QUERIES)} queries")
    print(f"{'='*72}")

    for i, query in enumerate(ALL_QUERIES, 1):
        print(f"\n{'='*72}")
        print(f"Query {i}/{len(ALL_QUERIES)}: {query}")
        print(f"{'='*72}")

        q_emb = client.embeddings.create(input=[query], model="text-embedding-3-small")
        q_vec = np.array([q_emb.data[0].embedding], dtype=np.float32)
        _, indices = search_faiss(index, q_vec, min(5, len(chunks)))
        retrieved = [chunks[j] for j in indices if j < len(chunks)]

        print("\nRetrieved Chunks:")
        for r in retrieved:
            snippet = (r[:180] + "...") if len(r) > 180 else r
            print(f'  - "{snippet}"')

        context = build_context(retrieved)
        response = get_llm_response(query, context)
        print(f"\nLLM Response:\n{response}\n")

    print(f"\nDone. {len(ALL_QUERIES)} queries answered with parser={parser}.")


def main():
    arg_parser = argparse.ArgumentParser(description="Run RAG on 13 queries for evaluation")
    arg_parser.add_argument("--pdf", type=str, default=None)
    arg_parser.add_argument("--parser", type=str, default="pymupdf",
                             choices=["pymupdf", "docling", "both"],
                             help="Parser to use: pymupdf, docling, or both (default: pymupdf)")
    arg_parser.add_argument("--rebuild", action="store_true", help="Force re-ingestion")
    args = arg_parser.parse_args()

    from run import find_pdf
    try:
        pdf_path = find_pdf(args.pdf)
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)

    parsers = ["pymupdf", "docling"] if args.parser == "both" else [args.parser]
    for p in parsers:
        run_queries_for_parser(p, pdf_path, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
