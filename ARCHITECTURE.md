# RAG Pipeline Architecture (End-to-End)

This document describes the full architecture of the RAG workflow: parsing, chunking, embedding, retrieval, and generation.

---

## 1. High-Level Flow

```
PDF file → Parse (PyMuPDF or Docling) → Full text → Chunk (sliding window) → Chunks (list[str])
    → Embed chunks (OpenAI) → Vectors (n × 1536) → FAISS index (L2)
    → User query → Embed query (OpenAI) → Query vector (1 × 1536)
    → FAISS search (L2 nearest neighbors) → Top-k chunk indices → Retrieved chunks
    → Build context string → LLM (OpenAI GPT) → Final answer (printed to console)
```

Optional: after the first run, the FAISS index and chunks are saved per parser to `data/faiss_index/pymupdf/` and `data/faiss_index/docling/`. Later runs load from cache and skip parse + embed.

---

## 2. Parsing (PyMuPDF and Docling)

We support two parsers; the user selects one via `--parser pymupdf` or `--parser docling` (or the Streamlit UI).

**PyMuPDF (`parser=pymupdf`)**  
Uses the PyMuPDF (fitz) library: it opens the PDF and, for each page, calls the MuPDF engine’s native text extraction (`page.get_text()`). No ML or layout models — it reads the PDF’s embedded text layer and optional positioning. Very fast (~1.7 s for 646 pages), stable, and good for standard PDFs with selectable text. Accuracy is high when the PDF has a clean text layer; it can miss or reorder text when the PDF relies heavily on visual layout or images.

**Docling (`parser=docling`)**  
Uses Docling’s low-level C++ text engine via `docling_parse.pdf_parsers.pdf_parser`: it loads the document and, per page, decodes words/cells (fonts, positioning, normalization) and concatenates them. This is the same text layer used by the full Docling pipeline, but without the ML layout/OCR/table models, so it does not crash and is slower (~19 s for 646 pages). It can be slightly more consistent for complex or multi-column PDFs because of Docling’s internal normalization; for simple programmatic PDFs, PyMuPDF and Docling usually give similar text and similar RAG accuracy.

| Aspect | PyMuPDF | Docling |
|--------|---------|--------|
| **Library** | PyMuPDF (`fitz`) | `docling_parse` (C++ PDF decoder) |
| **How it works** | `fitz.open(path)` → each page `page.get_text()` → concatenate. | Load doc → per page decode word cells → join normalized words. |
| **Speed** | ~1.7 s (646 pages) | ~19 s (646 pages) |
| **Total chunks (Philippine History PDF)** | **2,419** | **2,387** |
| **Output** | Full text string → same chunking (size=800, overlap=150). | Full text string → same chunking. |
| **Typical use** | Default; fast ingestion and query. | When you want Docling’s text normalization; comparable RAG quality. |

---

## 3. Chunking Strategy

| Parameter | Value | Where defined |
|-----------|--------|----------------|
| **Strategy** | Sliding-window (character-based). |
| **Chunk size** | **800** characters (max length per chunk). Default in `run.py`: `--chunk-size 800`; in `rag/parsers.py`: `DEFAULT_CHUNK_SIZE = 800`. |
| **Chunk overlap** | **150** characters. Consecutive chunks overlap by 150 chars so context is not lost at boundaries. Default in `run.py`: `--chunk-overlap 150`; in `rag/parsers.py`: `DEFAULT_CHUNK_OVERLAP = 150`. |

**How it works:** Start at index 0; take `chunk_size` characters → one chunk. Advance by `chunk_size - chunk_overlap` (i.e. 650 characters). Repeat until the end of the text. So chunk 2 starts 650 chars in, overlapping chunk 1 by 150 chars.

**Number of chunks:** Depends only on document length and the above parameters.

- Formula (conceptually): `1 + ceil((text_length - chunk_size) / (chunk_size - chunk_overlap))`, then drop empty chunks.
- **For the Philippine History PDF** (646 pages, NCCA source book):  
  - **PyMuPDF parser:** **2,419** total chunks (saved in `data/faiss_index/pymupdf/chunks.json`, `manifest.json` has `n_chunks: 2419`).  
  - **Docling parser:** **2,387** total chunks (saved in `data/faiss_index/docling/chunks.json`, `manifest.json` has `n_chunks: 2387`).  
  The small difference is due to slightly different extracted text length/format between the two parsers. Exact count is also in logs: `[parsers] Total chunks for RAG: N`.

---

## 4. Embedding (Document Chunks)

| Parameter | Value | Where defined |
|-----------|--------|----------------|
| **Model** | **OpenAI `text-embedding-3-small`** | `rag/embed_faiss.py`: `DEFAULT_EMBED_MODEL = "text-embedding-3-small"`. |
| **Dimension** | **1536** | Fixed for this model. Each chunk → one vector of 1536 floats. |
| **Batch size** | 100 chunks per API call | `rag/embed_faiss.py`: `batch_size = 100`. |
| **Output** | `np.ndarray` of shape `(n_chunks, 1536)`, dtype `float32`. Row `i` is the embedding of `chunks[i]`. |

All document chunks are embedded once (at ingestion). These vectors are not stored as a separate file; they are added to the FAISS index (the index “holds” the vectors).

---

## 5. FAISS Index and Why L2 (Not Cosine)

**What is L2?**  
L2 (Euclidean) distance between two vectors is the square root of the sum of squared differences: \( \|a - b\|_2 \). Lower L2 distance means the vectors are closer in space.

**Cosine** measures direction (angle between vectors): it’s the dot product of unit vectors. It ignores length and only cares whether vectors point the same way. So cosine is “direction,” L2 is “distance in space.”

**Do we use L2, and does FAISS use it by default?**  
We **explicitly** build a **L2 index**: `faiss.IndexFlatL2(dim)`. FAISS does not “default” to L2 — it offers several index types (e.g. `IndexFlatIP` for inner product, which is equivalent to cosine when vectors are normalized). We chose **IndexFlatL2** because it is the standard exact nearest-neighbor index and fits our embedding dimension; no extra normalization step is required in our code.

**Why are results similar to what cosine would give?**  
OpenAI’s `text-embedding-3-small` outputs **unit-norm** vectors (length 1). For unit-norm vectors, minimizing L2 distance is mathematically equivalent to maximizing cosine similarity: the closer two vectors are in L2, the more they point in the same direction. So the **ranking** of nearest chunks is the same whether we use L2 or cosine; we use L2 because the index type is simple and FAISS returns the k smallest L2 distances, which correspond to the k most similar chunks by direction.

**Numerical example (L2 vs cosine with unit-norm vectors)**  
Use small 2D vectors for illustration (real embeddings are 1536-D, same idea). Suppose all vectors are **unit length** (length = 1):

- **Query:** \( q = (1, 0) \)
- **Chunk A:** \( a = (1, 0) \) (same direction as query)
- **Chunk B:** \( b = (0.707, 0.707) \approx (1/\sqrt{2}, 1/\sqrt{2}) \) (45° from query)
- **Chunk C:** \( c = (0, 1) \) (90° from query)
- **Chunk D:** \( d = (-1, 0) \) (opposite direction)

**L2 distance** \( \|u - v\|_2 = \sqrt{\sum_i (u_i - v_i)^2} \):

| Pair | Difference vector | L2 distance |
|------|-------------------|-------------|
| \( (q, a) \) | \( (0, 0) \) | \( 0 \) |
| \( (q, b) \) | \( (1 - 0.707,\; 0 - 0.707) = (0.293, -0.707) \) | \( \sqrt{0.293^2 + 0.707^2} \approx 0.77 \) |
| \( (q, c) \) | \( (1, -1) \) | \( \sqrt{2} \approx 1.41 \) |
| \( (q, d) \) | \( (2, 0) \) | \( 2 \) |

**Cosine similarity** (for unit vectors, \( \cos\theta = q \cdot v \), dot product):

| Pair | Dot product (cosine) |
|------|----------------------|
| \( (q, a) \) | \( 1\cdot 1 + 0\cdot 0 = 1 \) |
| \( (q, b) \) | \( 1\cdot 0.707 + 0\cdot 0.707 = 0.707 \) |
| \( (q, c) \) | \( 1\cdot 0 + 0\cdot 1 = 0 \) |
| \( (q, d) \) | \( 1\cdot(-1) + 0\cdot 0 = -1 \) |

**Ranking (most similar first):**

- **By L2 (smallest distance first):** A (0) → B (0.77) → C (1.41) → D (2).
- **By cosine (largest similarity first):** A (1) → B (0.707) → C (0) → D (-1).

The order is identical. So for unit-norm vectors, “nearest by L2” and “most similar by cosine” give the same ranking. That is why using L2 in FAISS is fine here: we still retrieve the same top-k chunks as we would with cosine. The link is \( \|q - v\|^2 = 2(1 - q\cdot v) \) when both are unit, so smaller L2 ⇔ larger cosine.

**Why L2 is good here:** We use one index type (IndexFlatL2), no extra normalization in code, and retrieval order matches “most similar by direction” because embeddings are unit-norm. So L2 is both simple and correct for our pipeline.

| Parameter | Value | Where defined |
|-----------|--------|----------------|
| **Index type** | **`faiss.IndexFlatL2`** | We build this in `rag/embed_faiss.py`; FAISS uses L2 because we choose this index type. |
| **Dimension** | **1536** (must match embedding dimension). |
| **Vectors in index** | `n_chunks` (same as number of chunks). |
| **Similarity / distance** | **L2 (Euclidean).** Lower distance = more similar. Ranking matches cosine for unit-norm OpenAI embeddings. |

The index is saved per parser to `data/faiss_index/<parser>/index.faiss` so later runs can load it and skip re-embedding.

---

## 6. User Query Path (Retrieval)

| Parameter | Value | Where defined |
|-----------|--------|----------------|
| **Query input** | Single string (e.g. "When did the EDSA People Power Revolution happen?"). |
| **Query embedding model** | Same as document: **OpenAI `text-embedding-3-small`**. |
| **Query vector dimension** | **1536** (same as chunk vectors). |
| **Query embedding shape** | `(1, 1536)` (one row) before calling FAISS. |
| **Similarity used** | **L2 distance** (same as index). FAISS returns the top-k chunk indices with smallest L2 distance to the query vector. |
| **Top-k** | **5** by default. `run.py`: `--top-k 5`. So we retrieve the **5** most similar chunks (nearest neighbors in L2 sense). |

Flow: user query string → one API call to `embeddings.create(input=[query], model="text-embedding-3-small")` → query vector → `index.search(query_vec, k=top_k)` → distances and **indices** (positions in the original chunks list). We use the indices to take `retrieved = [chunks[i] for i in indices]`.

---

## 7. Context Building

| Parameter | Value | Where defined |
|-----------|--------|----------------|
| **Input** | List of retrieved chunk strings (length = top_k, e.g. 5). |
| **Max context length** | **4000** characters | `rag/llm.py`: `build_context(..., max_chars=4000)`. |
| **Separator** | `"\n\n---\n\n"` between chunks. |
| **Output** | Single string: chunks concatenated with the separator, truncated so that the total length does not exceed 4000 characters (we stop adding chunks when the next would exceed the limit). |

This string is the “context” passed to the LLM together with the user question.

---

## 8. LLM (Final Answer)

| Parameter | Value | Where defined |
|-----------|--------|----------------|
| **Model** | **OpenAI `gpt-4o-mini`** by default. Configurable via env **`OPENAI_MODEL`**. | `rag/llm.py`: `os.environ.get("OPENAI_MODEL", "gpt-4o-mini")`. |
| **Temperature** | **0.2** (low, for more deterministic answers). |
| **Input** | System prompt + user message (context + question). See **exact prompt** below. |
| **Output** | Single string (the model’s answer). Printed to the console; not saved to a file unless the user redirects. |

**Exact prompt used after context is built** (from `rag/llm.py`):

**System message:**
```
You are a helpful assistant that answers questions based only on the provided context
from a document about Philippine History. If the context does not contain enough
information to answer, say so. Keep answers concise and factual.
```

**User message (context + question):**
```
Use the following context to answer the question.

Context:
<context string — retrieved chunks joined with "\n\n---\n\n", up to 4000 chars>

Question: <user query>

Answer:
```

The LLM is called with these two messages; the model’s reply is the final answer we print.

---

## 9. End-to-End Summary Table

| Stage | What runs | Key parameters | Output |
|-------|-----------|----------------|--------|
| **Parse** | PyMuPDF or Docling on PDF | parser=pymupdf | docling | Full text string. |
| **Chunk** | Sliding window on full text | chunk_size=800, overlap=150 | `chunks`: list of strings (length N). |
| **Embed chunks** | OpenAI Embeddings API | model=text-embedding-3-small, dim=1536 | `embeddings`: (N, 1536) float32. |
| **Build index** | FAISS | IndexFlatL2, dim=1536 | In-memory index (saved to `data/faiss_index/<parser>/`). |
| **Query** | User string | — | e.g. "When did the EDSA People Power Revolution happen?" |
| **Embed query** | Same OpenAI model | dim=1536 | Query vector (1, 1536). |
| **Retrieve** | FAISS search | L2 (IndexFlatL2), top_k=5 | 5 chunk indices → 5 chunk strings. |
| **Context** | Concatenate chunks | max_chars=4000, separator `\n\n---\n\n` | One context string. |
| **LLM** | OpenAI Chat | system + user prompt above, temperature=0.2 | Final answer string (printed). |

---

## 10. File / Code Mapping

- **Parsing (both PyMuPDF and Docling):** `rag/parsers.py` (`_extract_text_pymupdf`, `_extract_text_docling`, `chunk_text`, `load_and_chunk_pdf`).
- **Embedding + FAISS:** `rag/embed_faiss.py` (`embed_chunks`, `build_faiss_index`, `search_faiss`, `save_faiss_index`, `load_faiss_index`).
- **Context + LLM:** `rag/llm.py` (`build_context`, `get_llm_response`).
- **Orchestration:** `run.py`, `ingest.py`, `app.py` (Streamlit).

---

## 11. How to Evaluate This RAG Pipeline

- **Correctness:** Run the same query with both parsers (`run.py --parser pymupdf` and `run.py --parser docling`) and compare answers to the source PDF. Check that dates, names, and events match the document.
- **Coverage:** Use the fixed set of 13 queries (`python run_queries.py --parser both`). For each query, inspect “Retrieved Chunks” — they should be topically relevant. If chunks are off-topic, try increasing `--top-k` or adjusting chunk size/overlap.
- **Faithfulness:** The LLM is instructed to answer only from context and to say when context is insufficient. If the model hallucinates or ignores the context, the prompt in §8 can be tightened (e.g. “Do not use outside knowledge”).
- **Latency:** First run (no cache) = parse + embed + query; cached run = load index + embed query + LLM. Measure with `time python run.py` and `time python run.py --parser docling`. Compare PyMuPDF vs Docling ingestion time and query-time behavior.
- **Regression:** Run `python -m pytest tests/ -v`. The E2E tests run the full pipeline for both parsers and check that output contains “User Query:”, “Retrieved Chunks:”, and “LLM Response:”.
- **Structured comparison:** See `RUN_REPORT.md` for timing, chunk counts, and side-by-side LLM responses for all 13 queries with both parsers.
