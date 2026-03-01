# How to Run & Where Everything Is Stored

## Status: fully working end-to-end ✅

Full ingestion and all 13 queries have been verified. See **RUN_REPORT.md** for:
- Exact timing (first run ~29 s; cached runs ~4 s)
- Chunk count: **2,419 chunks** from 646 pages
- All 13 query results (retrieved chunks + LLM responses)
- 14 pytest tests (3 E2E + 11 unit)

---

## Parser: PyMuPDF

PDF text extraction uses **PyMuPDF** (`fitz`) by default. It extracts the 646-page Philippine History PDF in **~1.7 seconds** with no crashes. Docling is also supported via `--parser docling`. The chunking, embedding, FAISS, and LLM pipeline are the same for both parsers.

---

## Why save chunks / FAISS index?

The pipeline can run entirely in memory. We also save the FAISS index and chunks to `data/faiss_index/` after the first run so later runs skip parse + embed and go straight to retrieval.

---

## How to run

```bash
source venv/bin/activate
export KMP_DUPLICATE_LIB_OK=TRUE   # prevents OpenMP conflict on macOS

python run.py                                          # default query
python run.py --query "Who is José Rizal?"             # custom query
python run.py --pdf path/to/file.pdf                   # custom PDF
python run.py --rebuild                                # force full re-ingestion
python run_queries.py                                  # all 13 queries
python -m pytest tests/ -v                             # tests
```

**Before running:** Put your OpenAI API key in `.env` (see `.env.example`). Place the Philippine History PDF in the project root or in `data/`.

---

## Commands: what runs, defaults, and result

### If PDF path is **not** provided

The script looks in this order:
1. `data/philippine_history.pdf`
2. `./philippine_history.pdf` (project root)
3. If `data/` has exactly one `.pdf` file, that file is used (e.g. `data/PHILIPPINE-HISTORY-SOURCE-BOOK-FINAL-SEP022021.pdf`)

### If query is **not** provided

Default query: **"When did the EDSA People Power Revolution happen?"**

### Command-by-command

| Command | What runs | Result |
|--------|-----------|--------|
| `python run.py` | Resolve PDF. If **no cache**: parse (PyMuPDF, ~1.7s) → chunk (2419 chunks) → embed (OpenAI, ~24s, 25 batches) → build FAISS → **save** to `data/faiss_index/` → run default query → print. If **cache valid**: load index + chunks (~0.5s) → run default query → print. | Console: "User Query:", "Retrieved Chunks:", "LLM Response:". First run ~29s; cached ~4s. |
| `python run.py --query "..."` | Same as above but with your question. | Same console format with your question's answer. |
| `python run.py --pdf path/to/file.pdf` | Use explicit PDF path. Cache checks are still performed. | Same as `python run.py`. |
| `python run.py --rebuild` | Always run full pipeline even if cache exists. Overwrites `data/faiss_index/`. | Re-ingests from scratch; same output. |
| `python run_queries.py` | Load cache (or full ingest if no cache). Run **13 queries**. For each: retrieve chunks, call LLM, print. | One block per query, ~39s total. |
| `python -m pytest tests/ -v` | 14 tests: E2E (both parsers + no-PDF), plus unit tests for chunking, context, FAISS. | `14 passed` |

### Verify the two flows

1. **First run (full ingestion):** With no cache (or use `--rebuild`), run `python run.py`. Logs will show STEP 1 (parse), STEP 2 (embed + build FAISS), save to `data/faiss_index/<parser>/`, then STEP 3/4.
2. **Second run (cache):** Run `python run.py` again (same PDF, no `--rebuild`). Logs will show "Loaded index" and skip to STEP 3/4.

---

## Time: without vs with cache

| Scenario | Time |
|----------|------|
| First run (parse + embed + FAISS + query) | **~29 seconds** |
| Cached run (load index + query + LLM only) | **~4 seconds** |

---

## Where results and data are stored

### On disk (files)

| What | Where | When |
|------|--------|------|
| **Input PDF** | `data/` or project root | You place it there |
| **Parsed text (JSON/TXT)** | `data/parsed_output/{stem}_{parser}_{timestamp}.json` / `.txt` | Written on first run (no cache) |
| **FAISS index** | `data/faiss_index/pymupdf/` and `data/faiss_index/docling/` | `index.faiss`, `chunks.json`, `manifest.json` per parser |

### In memory during a run

| What | Notes |
|------|-------|
| **Chunks** | `list[str]`, 2419 strings; from parser or loaded from `chunks.json` |
| **Embeddings** | `np.ndarray (2419, 1536)` float32; built at embed time; inside FAISS index |
| **FAISS index** | `faiss.IndexFlatL2`; in-memory or loaded from `index.faiss` |
| **Query answer** | Printed to terminal only; not saved unless you redirect (`python run.py > out.txt`) |

---

## Summary

- **Parser:** PyMuPDF — 646 pages, 1.57M chars, 1.7 s, no crashes.
- **Chunks:** 2,419 (chunk size 800, overlap 150).
- **Embeddings:** OpenAI `text-embedding-3-small`, dim 1536.
- **FAISS:** `IndexFlatL2`, 2419 vectors.
- **LLM:** `gpt-4o-mini`; API key from `.env` (see README).
- **First run:** ~29 s. **Cached runs:** ~4 s.
- For full architecture details see **ARCHITECTURE.md**. For run results (all 13 queries) see **RUN_REPORT.md**.
