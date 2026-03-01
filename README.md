# AI Context Builder — Philippine History RAG

A Retrieval-Augmented Generation (RAG) pipeline that parses a Philippine History PDF, embeds chunks using OpenAI + FAISS, and answers natural language questions using GPT.

**Contents:**
- **README.md** — setup, run instructions, and LLM/config notes (below)
- **run.py** — main script: load/chunk PDF → FAISS → query → LLM → print answer
- **requirements.txt** — dependencies
- **.env.example** — copy to `.env` and add your OpenAI key (see Setup)

**Two parser options:**
| Parser | Speed | Description |
|--------|-------|-------------|
| `pymupdf` (default) | ~1.7s / 646 pages | PyMuPDF — fast, reliable, no ML models |
| `docling` | ~22s / 646 pages | docling_parse C++ engine — Docling's own text layer, no ML crash |

---

## 1. Prerequisites

- **Python 3.8+** (tested on 3.13)
- **OpenAI API key** — set as `TEAMIFIED_OPENAI_API_KEY`
- **PDF file** — place the Philippine History PDF in the project root or `data/` (the script looks for it there; any single .pdf in data/ is used if the default name is missing)

---

## 2. Setup

```bash
# 1. Clone and enter project
cd Document_Parsing

# 2. Create virtual environment
python3 -m venv venv

# macOS / Linux:
source venv/bin/activate
# Windows (PowerShell):
# venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
# Note: if faiss-cpu fails to build from source, use the binary wheel:
# pip install faiss-cpu --only-binary :all:

# 4. Set your API key (pick one method):
# Option A — .env file (recommended):
cp .env.example .env
# Edit .env and set:  TEAMIFIED_OPENAI_API_KEY=sk-your-key-here

# Option B — shell environment (no file needed):
export TEAMIFIED_OPENAI_API_KEY=sk-your-key-here

# 5. Place the PDF
# Put philippine_history.pdf in the project root or in data/
# (Any single .pdf file in data/ is auto-detected)
```

---

## 3. Run — CLI

### Single query (default)

```bash
python run.py
# Uses PyMuPDF parser, default query: "When did the EDSA People Power Revolution happen?"
```

### Single query with options

```bash
python run.py --query "Who is José Rizal and why is he important?"
python run.py --parser docling --query "Tell me about the Spanish colonization of the Philippines."
python run.py --pdf path/to/custom.pdf --parser pymupdf
python run.py --rebuild        # force re-ingestion even if cache exists
python run.py --top-k 3        # retrieve 3 chunks instead of 5
```

### Parser flag

```bash
python run.py --parser pymupdf   # fast (~29s first run, ~4s cached)
python run.py --parser docling   # thorough (~50s first run, ~4s cached)
```

**First run** (no cache): parses PDF → chunks → embeds → builds FAISS → saves to `data/faiss_index/<parser>/`  
**Later runs** (cache exists for same PDF): loads index in ~0.5s, runs query only

### All 13 queries for evaluation

```bash
python run_queries.py                     # PyMuPDF (default)
python run_queries.py --parser docling    # Docling
python run_queries.py --parser both       # both parsers side-by-side
```

---

## 4. Run — Web UI (Streamlit)

```bash
streamlit run app.py
```

Opens in browser at `http://localhost:8501`

Features:
- Select parser: PyMuPDF or Docling
- Type any question about Philippine History
- Adjust top-K retrieved chunks
- See Retrieved Chunks (expandable) + LLM Response
- Sidebar shows cache status and sample queries
- First use of a parser triggers automatic ingestion

---

## 5. Pre-ingestion (optional — for sharing/zipping)

Run this once to build both FAISS indexes so recipients can query immediately without waiting for ingestion:

```bash
python ingest.py              # build both pymupdf and docling indexes
python ingest.py --parser pymupdf    # only PyMuPDF
python ingest.py --parser docling    # only Docling
python ingest.py --rebuild           # force rebuild even if cached
```

Indexes saved to:
- `data/faiss_index/pymupdf/` — index.faiss (14 MB), chunks.json (1.9 MB), manifest.json
- `data/faiss_index/docling/` — same structure

After `ingest.py`, queries via `run.py` or `app.py` skip ingestion entirely.

---

## 6. Sample output (expected format)

The script prints **User Query**, **Retrieved Chunks**, and **LLM Response** to the console. Example:

```
$ python run.py

User Query: When did the EDSA People Power Revolution happen?

Retrieved Chunks:
  - "The EDSA People Power Revolution occurred in February 1986..."
  - "It led to the ousting of President Ferdinand Marcos..."

LLM Response:
"The EDSA People Power Revolution happened in February 1986 and marked the end of Marcos' dictatorship in the Philippines."
```

Actual content may vary slightly; the structure (User Query → Retrieved Chunks → LLM Response) is what the tests verify.

---

## 7. Tests

```bash
python -m pytest tests/ -v
```

**Unit tests** (no API key or PDF; 11 tests in `test_rag_units.py`):

| Test file | What it checks |
|-----------|----------------|
| `tests/test_rag_units.py` | Chunking (`chunk_text`), context building (`build_context`), FAISS index build + search with dummy vectors (11 tests) |

**End-to-end tests** (require `TEAMIFIED_OPENAI_API_KEY` and PDF in `data/` or project root):

| Test | What it checks |
|------|----------------|
| `test_e2e_run_py_produces_expected_output` | Full pipeline: `run.py` → stdout contains **"User Query:"**, **"Retrieved Chunks:"**, **"LLM Response:"** (parametrized for both `pymupdf` and `docling`; 900s timeout) |
| `test_e2e_run_py_fails_gracefully_without_pdf` | `run.py --pdf /nonexistent/...` exits non-zero with helpful message |

---

## 8. Notes on LLM choice

- **Model:** **`gpt-4o-mini`** by default; override with `OPENAI_MODEL` in `.env` if needed.
- **API key:** Set **`TEAMIFIED_OPENAI_API_KEY`** in `.env` or the environment (see Setup). The code also checks `OPENAI_API_KEY` if the first is unset.
- **Embeddings:** OpenAI **`text-embedding-3-small`** (1536 dimensions).
- **Temperature:** 0.2 to keep answers focused and factual.

---

## 9. Architecture

| Step | Detail |
|------|--------|
| Parse | PyMuPDF (~1.7s) or docling_parse (~22s) |
| Chunk | Sliding-window, size=800 chars, overlap=150 chars |
| Embed | OpenAI `text-embedding-3-small`, dim=1536, batch=100 |
| Index | FAISS `IndexFlatL2`, exact L2 nearest-neighbor |
| Retrieve | Top-5 chunks by L2 similarity to query vector |
| Generate | GPT-4o-mini, temperature=0.2, context max=4000 chars |

See `ARCHITECTURE.md` for full details and `RUN_REPORT.md` for benchmark results.

---

## 10. File structure

```
Document_Parsing/
├── run.py              # CLI: single query (--parser, --query, --rebuild, ...)
├── run_queries.py      # CLI: all 13 evaluation queries (--parser both)
├── ingest.py           # Pre-ingestion: build FAISS indexes for both parsers
├── app.py              # Streamlit UI
├── rag/
│   ├── parsers.py      # PDF parsing (pymupdf + docling)
│   ├── embed_faiss.py  # Embed chunks + build/save/load FAISS index
│   └── llm.py          # Build context + call OpenAI GPT
├── tests/
│   └── test_e2e_run.py # E2E tests
├── data/
│   ├── philippine_history.pdf   ← place PDF here (or project root)
│   ├── faiss_index/
│   │   ├── pymupdf/             # cached index for PyMuPDF ingestion
│   │   └── docling/             # cached index for Docling ingestion
│   └── parsed_output/           # saved parse JSON + TXT for inspection
├── requirements.txt
├── .env.example
├── README.md
├── ARCHITECTURE.md     # Detailed technical architecture
└── RUN_REPORT.md       # Benchmark: timing, chunk counts, all 13 query results
```
