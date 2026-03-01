"""
PDF parsing for RAG: supports two parsers selectable via the `parser` parameter.

  - "pymupdf"  : PyMuPDF (fitz) — fast (~1.7s for 646 pages), no ML, no crashes.
  - "docling"  : docling_parse (Docling's C++ text engine) — thorough (~22s), no ML crash,
                 uses the same low-level engine as the full Docling pipeline.

Usage:
    chunks = load_and_chunk_pdf("data/philippine_history.pdf", parser="pymupdf")
    chunks = load_and_chunk_pdf("data/philippine_history.pdf", parser="docling")
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 800    # max characters per chunk
DEFAULT_CHUNK_OVERLAP = 150  # characters of overlap between consecutive chunks

SUPPORTED_PARSERS = ("pymupdf", "docling")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Sliding-window character-based chunking.

    Args:
        text: Full document text.
        chunk_size: Max characters per chunk (default 800).
        chunk_overlap: Overlap between consecutive chunks (default 150).
            Step = chunk_size - chunk_overlap = 650 chars per advance.

    Returns:
        list[str]. Non-empty stripped chunks.
        Example: ["Table of Contents\n...", "Rizal was born in...", ...]
    """
    chunks = []
    step = chunk_size - chunk_overlap
    start = 0
    while start < len(text):
        chunk = text[start: start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


# ---------------------------------------------------------------------------
# PyMuPDF parser
# ---------------------------------------------------------------------------

def _extract_text_pymupdf(document_path: str) -> tuple[str, int, float]:
    """
    Extract full text using PyMuPDF (fitz).

    Returns:
        (full_text, page_count, elapsed_seconds)
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF required: pip install pymupdf")

    t0 = time.time()
    doc = fitz.open(document_path)
    pages = len(doc)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    elapsed = time.time() - t0
    logger.info("[parsers:pymupdf] %d pages, %d chars in %.2fs", pages, len(text), elapsed)
    return text, pages, elapsed


# ---------------------------------------------------------------------------
# Docling parser (docling_parse C++ engine — no ML models)
# ---------------------------------------------------------------------------

def _extract_text_docling(document_path: str) -> tuple[str, int, float]:
    """
    Extract full text using docling_parse (Docling's C++ text extraction engine).
    Does NOT use the ML layout/OCR/table models — just the low-level PDF decoder.

    Returns:
        (full_text, page_count, elapsed_seconds)
    """
    try:
        from docling_parse.pdf_parsers import pdf_parser, DecodePageConfig
    except ImportError:
        raise ImportError("docling_parse required: pip install docling")

    t0 = time.time()
    parser = pdf_parser()
    parser.set_loglevel_with_label("error")  # suppress verbose C++ logs
    key = "rag_doc"
    parser.load_document(key, document_path)
    npages = parser.number_of_pages(key)
    cfg = DecodePageConfig()

    all_text = []
    for i in range(npages):
        decoder = parser.get_page_decoder(key, i, cfg)
        word_cells = decoder.get_word_cells()
        page_words = [c.text for c in word_cells if c.text and c.text.strip()]
        all_text.append(" ".join(page_words))
        parser.unload_document_page(key, i)

    full_text = "\n".join(all_text)
    elapsed = time.time() - t0
    logger.info("[parsers:docling] %d pages, %d chars in %.2fs", npages, len(full_text), elapsed)
    return full_text, npages, elapsed


# ---------------------------------------------------------------------------
# Save parsed output for inspection
# ---------------------------------------------------------------------------

def _save_parse_output(
    full_text: str,
    chunks: list[str],
    output_dir: Path,
    pdf_stem: str,
    pages: int,
    elapsed: float,
    parser: str,
) -> tuple[Path, Path]:
    """
    Save parse result to output_dir: JSON (metadata + chunks) and .txt (full text).

    Returns:
        (json_path, txt_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"{pdf_stem}_{parser}_{ts}.json"
    txt_path = output_dir / f"{pdf_stem}_{parser}_{ts}.txt"

    result = {
        "parser": parser,
        "timestamp": ts,
        "pages": pages,
        "text_length": len(full_text),
        "total_chunks": len(chunks),
        "parse_time_seconds": round(elapsed, 2),
        "chunks": chunks,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("[parsers] Saved parse JSON (%s): %s", parser, json_path)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    logger.info("[parsers] Saved parse TXT (%s): %s", parser, txt_path)

    return json_path, txt_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_and_chunk_pdf(
    document_path: str,
    parser: str = "pymupdf",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    save_parsed_output_dir: str | Path | None = None,
    # backward-compat alias
    save_docling_output_dir: str | Path | None = None,
) -> list[str]:
    """
    Parse a PDF and return text chunks for embedding.

    Args:
        document_path: Path to the PDF file.
        parser: "pymupdf" (default, fast ~2s) or "docling" (thorough ~22s).
        chunk_size: Max chars per chunk. Default 800.
        chunk_overlap: Overlap between chunks. Default 150.
        save_parsed_output_dir: If set, save parsed JSON + TXT here for inspection.
        save_docling_output_dir: Backward-compat alias for save_parsed_output_dir.

    Returns:
        list[str] of non-empty text chunks ready for embedding.

    Example:
        chunks = load_and_chunk_pdf("data/philippine_history.pdf", parser="pymupdf")
        # chunks[0] -> "Table of Contents\n..."
        # len(chunks) -> 2419
    """
    save_dir = save_parsed_output_dir or save_docling_output_dir
    path = Path(document_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {document_path}")

    if parser not in SUPPORTED_PARSERS:
        raise ValueError(f"Unknown parser '{parser}'. Choose from: {SUPPORTED_PARSERS}")

    logger.info("[parsers] Using parser=%s on %s", parser, document_path)

    if parser == "pymupdf":
        full_text, pages, elapsed = _extract_text_pymupdf(str(path))
    else:  # docling
        full_text, pages, elapsed = _extract_text_docling(str(path))

    chunks = chunk_text(full_text, chunk_size, chunk_overlap)
    logger.info(
        "[parsers] Chunking done: %d chunks (size=%d, overlap=%d)", len(chunks), chunk_size, chunk_overlap
    )

    if save_dir:
        _save_parse_output(full_text, chunks, Path(save_dir), path.stem, pages, elapsed, parser)

    if not chunks:
        raise RuntimeError("No chunks produced. Ensure the PDF path is correct and contains extractable text.")

    logger.info("[parsers] Total chunks for RAG: %d", len(chunks))
    return chunks
