# Run Report: Both Parsers — Ingestion + All 13 Queries

Recorded: **2026-03-01**, macOS Apple Silicon ARM64, Python 3.13.11

---

## 1. Parser Comparison

| Feature | PyMuPDF (`pymupdf`) | Docling (`docling`) |
|---------|---------------------|---------------------|
| **Library** | `fitz` (PyMuPDF) | `docling_parse.pdf_parsers.pdf_parser` |
| **Parse time** | **1.73s** | **19.08s** |
| **Total ingestion time** | **22.03s** (parse + embed) | **36.79s** (parse + embed) |
| **Pages** | 646 | 646 |
| **Text extracted** | 1,572,034 chars | 1,551,264 chars |
| **Chunks** | **2,419** | **2,387** |
| **Crashes** | None | None (uses docling_parse, not full Docling ML pipeline) |
| **Text quality** | Good — standard text layer extraction | Good — Docling's own normalization |
| **Use case** | Fast querying, default | When you want Docling's text processing |

**Ingestion is cached per-parser.** After first run, both indexes load in < 1s.

---

## 2. Commands run

### Pre-ingestion (both parsers)

```bash
source venv/bin/activate
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
python ingest.py   # builds both data/faiss_index/pymupdf/ and data/faiss_index/docling/
```

**Output:**
```
[pymupdf] Done: 2419 chunks | parse=1.73s | embed=20.3s  | total=22.03s
[docling] Done: 2387 chunks | parse=19.08s | embed=17.71s | total=36.79s
```

### All 13 queries — both parsers

```bash
python run_queries.py --parser both    # ~70s (cache loaded; only query + LLM time)
```

### Tests

```bash
python -m pytest tests/ -v
```

**Result: 3/3 PASSED** (pymupdf E2E + docling E2E + no-PDF graceful error)

---

## 3. Ingestion metrics

### PyMuPDF

| Metric | Value |
|--------|-------|
| Parse time | 1.73 s |
| Embed time | 20.30 s |
| Total time | 22.03 s |
| Pages | 646 |
| Text extracted | 1,572,034 chars |
| Chunks | 2,419 |
| Embedding model | `text-embedding-3-small` (dim=1536) |
| Batches | 25 (100 chunks each) |
| FAISS type | `IndexFlatL2` |
| Cached at | `data/faiss_index/pymupdf/` |

### Docling

| Metric | Value |
|--------|-------|
| Parse time | 19.08 s |
| Embed time | 17.71 s |
| Total time | 36.79 s |
| Pages | 646 |
| Text extracted | 1,551,264 chars |
| Chunks | 2,387 |
| Embedding model | `text-embedding-3-small` (dim=1536) |
| Batches | 24 (100 chunks each) |
| FAISS type | `IndexFlatL2` |
| Cached at | `data/faiss_index/docling/` |

---

## 4. All 13 queries — results (both parsers)

---

### Q1: When did the EDSA People Power Revolution happen?

| Parser | LLM Response |
|--------|-------------|
| PyMuPDF | The EDSA People Power Revolution happened on February 25, 1986. |
| Docling | The EDSA People Power Revolution happened on February 25, 1986. |

---

### Q2: Who is José Rizal and why is he important?

| Parser | LLM Response |
|--------|-------------|
| PyMuPDF | José Rizal is a national hero of the Philippines, recognized for his contributions to the country's fight for freedom and social reform during the Spanish colonial period. He is important because he understood the social problems of his time and used his writings, particularly his novels "Noli Me Tangere" and "El Filibusterismo," to awaken national consciousness and inspire patriotism among Filipinos. His life and works serve as a source of inspiration for Filipinos to live for their country and address contemporary challenges with constructive solutions. |
| Docling | José Rizal is a national hero of the Philippines, recognized for his significant contributions to the country's fight for freedom and social reform during the Spanish colonial period. He is important because he understood the social problems of his time and used his writings, particularly his novels "Noli Me Tangere" and "El Filibusterismo," to awaken national consciousness and inspire patriotism among Filipinos. His approach to nationalism was characterized by realism and constructive solutions, making him a key figure in promoting positive nationalism and encouraging future generations to live for their country. |

---

### Q3: Tell me about the Spanish colonization of the Philippines.

| Parser | LLM Response |
|--------|-------------|
| PyMuPDF | The Spanish colonization of the Philippines began in the 16th century, marked by the arrival of Spanish missionaries and explorers. The Jesuit Fr. Pedro Chirino (arrived 1590) documented early interactions with the Tagalog people. The Spanish established colonial administration, focusing on spiritual and temporal conquests. The galleon trade became a crucial economic activity, lasting over two centuries until its abolition in 1815. The period also saw various insurrections by Filipinos against Spanish rule. The 19th century brought discussions on the Philippines' future, including debates on emancipation from Spanish control. |
| Docling | The Spanish colonization of the Philippines began in the 16th century, involving the establishment of Spanish dominion over the islands, leading to significant cultural, religious, and social changes among the Filipino population. Spanish missionaries played a crucial role in converting Filipinos to Christianity and documenting their daily lives and cultures. The period saw various insurrections by Filipinos against Spanish rule. The galleon trade became the economic backbone of the colony, facilitating trade between the Philippines and Spain. The colonization period laid the groundwork for developments in the 19th century, culminating in a growing sense of Filipino identity and nationalism. |

---

### Q4: What was the First Philippine Republic?

| Parser | LLM Response |
|--------|-------------|
| PyMuPDF | The First Philippine Republic, established on June 12, 1898, was self-proclaimed by the Filipino people. It was the first authentic Filipino republic and the first Asian Republic, led by General Emilio Aguinaldo, following a nationwide revolution against Spanish rule. It was characterized by the adoption of a Constitution in Malolos, Bulacan. |
| Docling | The First Philippine Republic, established on June 12, 1898, was self-proclaimed by the Filipino people and marked the first authentic Filipino republic and the first Asian Republic. It was led by General Emilio Aguinaldo, following a nationwide revolution against Spanish rule, and was characterized by the adoption of a Constitution in Malolos, Bulacan. |

---

### Q5: When did the Philippines gain independence from the United States?

| Parser | LLM Response |
|--------|-------------|
| PyMuPDF | The Philippines gained independence from the United States on July 4, 1946. |
| Docling | The Philippines gained independence from the United States on July 4, 1946. |

---

### Q6: Who was Andres Bonifacio and what role did he play?

| Parser | LLM Response |
|--------|-------------|
| PyMuPDF | Andres Bonifacio was a key figure in the Philippine Revolution against Spanish colonial rule. He was a leader of the revolutionary movement, often referred to as the "Father of the Philippine Revolution." Bonifacio played a significant role in organizing the Katipunan, a secret society aimed at gaining independence from Spain. He also sought to spread the revolution beyond Manila, particularly to provinces like Nueva Ecija and Bulacan. His leadership and vision were crucial during the early stages of the revolution, although he faced internal conflicts, such as the controversial Tejeros convention. |
| Docling | Andres Bonifacio was a key figure in the Philippine Revolution against Spanish colonial rule. He was a leader of the revolutionary movement and is often referred to as the "Father of the Philippine Revolution." Bonifacio played a significant role in organizing the Katipunan, a secret society aimed at gaining independence from Spain. He also sought to spread the revolution beyond Manila, particularly to provinces like Nueva Ecija and Bulacan. His leadership and vision were crucial during the early stages of the revolution, although he faced internal conflicts, such as the controversial Tejeros convention. |

---

### Q7: What is the significance of the Cry of Pugad Lawin?

| Parser | LLM Response |
|--------|-------------|
| PyMuPDF | The provided context does not contain information about the Cry of Pugad Lawin, so I cannot answer the question. |
| Docling | The provided context does not contain information regarding the significance of the Cry of Pugad Lawin. |

*(Both parsers correctly flagged insufficient context — honest RAG behavior)*

---

### Q8: When did the Philippine-American War start and end?

| Parser | LLM Response |
|--------|-------------|
| PyMuPDF | The Philippine-American War started on February 4, 1899, and officially ended in July 1902. |
| Docling | The Philippine-American War started on February 4, 1899, and officially ended in July 1902. |

---

### Q9: Who was Emilio Aguinaldo?

| Parser | LLM Response |
|--------|-------------|
| PyMuPDF | Emilio Aguinaldo was the President of the Philippines and the leader of the Philippine revolutionary government. He declared war on the United States on February 4, 1899. He established his government in Malolos, Bulacan, and surrendered to the United States on April 19, 1901, after being captured by American forces. |
| Docling | Emilio Aguinaldo was the President of the Philippines and the leader of the Philippine revolutionary government. He declared war on the United States on February 4, 1899, following conflicts between Philippine and American forces. He established his government in Malolos, Bulacan, and later surrendered to the United States on April 19, 1901, after being captured by American forces. |

---

### Q10: What was the Treaty of Paris 1898 and how did it affect the Philippines?

| Parser | LLM Response |
|--------|-------------|
| PyMuPDF | The Treaty of Paris 1898 was a peace agreement between Spain and the United States that ended the Spanish-American War. Under the treaty, the United States gained possession of the Philippine Islands, along with Puerto Rico and Guam, while Cuba gained independence from Spain. The treaty was concluded without Filipino representation or consultation, leading to resentment among Filipinos, which contributed to the outbreak of the Filipino-American War, beginning on February 4, 1899. |
| Docling | The Treaty of Paris 1898 was a peace agreement between Spain and the United States that ended the Spanish-American War. Under the treaty, the United States gained possession of the Philippine Islands, along with Puerto Rico and Guam, while Cuba gained independence from Spain. The treaty was concluded without Filipino representation or consultation, leading to resentment among Filipinos, who felt that the provisions favored imperialist interests rather than their own. This discontent contributed to the outbreak of hostilities between Filipinos and Americans, known as the Filipino-American War, which began on February 4, 1899. |

---

### Q11: Describe the Katipunan and its goals.

| Parser | LLM Response |
|--------|-------------|
| PyMuPDF | The Katipunan (Kataastaasan Kagalanggalangang Katipunan ng mga Anak ng Bayan) was a revolutionary organization established on July 7, 1892, under Andres Bonifacio. Its primary goal was to gain independence from Spanish colonial rule. It aimed to liberate the nation from oppression and promoted values such as defending the oppressed, upholding good morals, observing equality, and respecting women. The organization also emphasized fraternity, patriotism, and the need for freedom among its members. |
| Docling | The Katipunan (Kataastaasan Kagalanggalangang Katipunan ng mga Anak ng Bayan) was a revolutionary organization established on July 7, 1892, under the guidance of Andres Bonifacio. Its primary goal was to gain independence for the Philippines from Spanish colonial rule. The Katipunan aimed to liberate the nation from oppression and promoted values such as defending the oppressed, upholding good morals, observing equality, and respecting women. The organization also emphasized fraternity, patriotism, and the need for freedom among its members. |

---

### Q12: When did martial law begin under Ferdinand Marcos?

| Parser | LLM Response |
|--------|-------------|
| PyMuPDF | Martial law began under Ferdinand Marcos on September 21, 1972. |
| Docling | Martial law began under Ferdinand Marcos on September 21, 1972. |

---

### Q13: What is the EDSA Revolution and why is it important to Philippine history?

| Parser | LLM Response |
|--------|-------------|
| PyMuPDF | The EDSA Revolution (People Power Revolution) occurred from February 22–25, 1986. It was a peaceful uprising leading to the ousting of President Ferdinand Marcos. It is important to Philippine history because it marked the restoration of democracy after years of authoritarian rule, symbolizing the power of the Filipino people to effect change through nonviolent means. |
| Docling | The EDSA Revolution (People Power Revolution) occurred from February 22–25, 1986, and was a peaceful uprising that led to the ousting of President Ferdinand Marcos. It is important to Philippine history because it marked the restoration of democracy after years of authoritarian rule, symbolizing the power of the Filipino people to effect change through nonviolent means. The revolution is celebrated for its role in reclaiming civil liberties and democratic institutions in the Philippines. |

---

## 5. Tests — 3/3 PASSED

```
tests/test_e2e_run.py::test_e2e_run_py_produces_expected_output[pymupdf]  PASSED
tests/test_e2e_run.py::test_e2e_run_py_produces_expected_output[docling]  PASSED
tests/test_e2e_run.py::test_e2e_run_py_fails_gracefully_without_pdf       PASSED
3 passed in 4.37s
```

