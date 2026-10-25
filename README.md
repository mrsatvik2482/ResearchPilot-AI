# 🔬 ResearchPilot AI — Multi-Agent Research Assistant Using RAG

A **local-first, completely free** college-project prototype. Upload research
paper PDFs, ask a research question, and get a **cited literature review**
written by three cooperating AI agents — with **no API keys, no cloud
services, and no paid models**. Everything (PDF processing, embeddings,
vector search and LLM inference) runs on your own machine.

| Layer | Technology |
|---|---|
| UI | Streamlit (dark dashboard) |
| PDF extraction | PyMuPDF (page-by-page text) |
| Embeddings | Sentence Transformers — `all-MiniLM-L6-v2` (384-dim, ~90 MB) |
| Vector search | FAISS (CPU, cosine similarity) |
| Agent orchestration | LangGraph (3-node stateful workflow) |
| LLM | Ollama — `llama3.2:3b` (~2 GB, fits 16 GB RAM machines) |

---

## Features

- **PDF ingestion** — upload multiple papers; text is extracted page by page
  with filename, detected title and page numbers preserved. Invalid, empty,
  password-protected and scanned PDFs are handled with clear error messages.
- **RAG** — pages are split into overlapping 900-character chunks, embedded
  locally, and indexed in FAISS. Retrieved passages are shown with their
  sources and similarity scores.
- **Three real agents (LangGraph)** with explicit shared state:
  1. **Paper Analysis Agent** — per-paper research question, methodology,
     findings, limitations (grounded in retrieved passages).
  2. **Comparison Agent** — cross-paper comparison of methods, datasets,
     findings and limitations; missing information is marked *unknown*.
  3. **Citation Verification Agent** — re-checks each generated claim against
     retrieved passages and flags `SUPPORTED` / `UNSUPPORTED` /
     `INSUFFICIENT_EVIDENCE`.
- **Literature review** — introduction, paper summaries, comparison table,
  tentative research gaps, conclusion and references — all citing **real
  uploaded filenames and real PDF page numbers**. Nothing is invented; gaps
  are labelled tentative; weak evidence is called out explicitly.
- **Tests** — 40 automated tests (chunking, metadata preservation, retrieval,
  full agent workflow with a mocked LLM).

---

## Setup (Windows · PowerShell)

> Requirements: Windows 10/11, ~6 GB free disk, 16 GB RAM recommended,
> Python **3.10–3.12** (3.11 tested).

```powershell
# 0. Open the project folder
cd ResearchPilot-AI

# 1. Create and activate a virtual environment (one time)
#    (keeps this project's packages fully isolated from your other projects)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked, run this ONCE and retry:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 2. Install Python dependencies (one time, includes torch — a few minutes)
pip install -r requirements.txt

# 3. Install Ollama (one time) — the local LLM runtime
#    https://ollama.com/download
winget install -e --id Ollama.Ollama

# 4. Download the local LLM (one time, ~2 GB)
ollama pull llama3.2:3b

# 5. Launch the app
streamlit run app.py
```

The browser opens at **http://localhost:8501**. The embedding model
(~90 MB) downloads automatically the first time you index PDFs — after that
the app runs **fully offline**.

**Weaker machine?** Use a smaller model: `ollama pull llama3.2:1b` and pick
it in the sidebar. Faster machine? `qwen2.5:7b` gives better writing.

---

## Running the tests

```powershell
python -m pytest -v
```

40 tests run **offline in ~1 second** — the LLM is replaced by a scripted
mock and embeddings by a deterministic fake, so no model download is needed:

- `tests/test_chunking.py` — overlapping chunks, no words lost, chunks never
  cross page boundaries
- `tests/test_metadata.py` — filename/title/page preservation, invalid /
  empty / scanned / password-protected PDF handling
- `tests/test_retrieval.py` — FAISS search returns the right passages with
  correct `filename, p. N` sources
- `tests/test_agents.py` — the full LangGraph workflow: agent call order,
  real data flow between agents, failure handling, unparseable replies
  become `UNKNOWN` (never invented)
- `tests/test_review.py` — report structure, references are only real files
  and pages, gaps labelled tentative, LLM/fallback behaviour

---

## Command-line demo (no UI)

```powershell
python scripts/run_e2e.py sample_pdfs/sample_paper_1_transformers.pdf sample_pdfs/sample_paper_2_leaf_disease.pdf "How are neural networks applied in classification tasks?"
```

Prints the full review and saves it to `output/literature_review.md`.

Two **synthetic sample papers** (clearly labelled as synthetic) are included
in `sample_pdfs/` — perfect for a quick demo without hunting for real PDFs.
Regenerate them anytime with `python scripts/make_sample_pdfs.py`.

---

## Project structure

```
ResearchPilot-AI/
├── app.py                      # Streamlit dashboard (dark theme)
├── requirements.txt
├── conftest.py                 # shared test fixtures (mock LLM, fake embedder, PDF factory)
├── pytest.ini
├── researchpilot/
│   ├── config.py               # all settings in one place
│   ├── pdf_ingest.py           # PyMuPDF extraction + overlapping chunking
│   ├── embeddings.py           # Sentence Transformers wrapper (unit vectors)
│   ├── vector_store.py         # FAISS index (cosine similarity)
│   ├── retriever.py            # global + per-paper search, cited passages
│   ├── llm.py                  # Ollama REST client + grounding system prompt
│   ├── review.py               # deterministic report assembly + references
│   └── agents/
│       ├── state.py            # LangGraph shared state (the data contract)
│       ├── analysis.py         # Agent 1 — Paper Analysis
│       ├── comparison.py       # Agent 2 — Comparison
│       ├── citation.py         # Agent 3 — Citation Verification
│       └── graph.py            # graph wiring: START → 1 → 2 → 3 → END
├── tests/                      # 40 tests, all offline
├── scripts/
│   ├── make_sample_pdfs.py     # generates the synthetic demo papers
│   └── run_e2e.py              # full pipeline from the command line
└── sample_pdfs/                # two synthetic demo papers
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Ollama is not running` (red box in sidebar) | Start the Ollama desktop app, or run `ollama serve` in a terminal |
| `Model 'llama3.2:3b' is not installed` | `ollama pull llama3.2:3b` (~2 GB, one time) |
| Very slow first run | First PDF upload downloads the ~90 MB embedding model — only once |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry |
| PDF rejected as "scanned" | The PDF has no text layer (images only). This prototype has no OCR — use a text-based PDF |
| `faiss-cpu` / `torch` install errors on Python 3.13 | Use Python 3.11 or 3.12: `py -3.11 -m venv .venv` |
| Torch DLL errors on Windows | This repo pins `torch<2.8` and `pymupdf<2` in `requirements.txt` — run `pip install -r requirements.txt` again inside the venv |

---

## How it works (60-second version)

1. **Ingest** — PyMuPDF extracts text page by page; every page keeps its
   filename, detected title and page number.
2. **Chunk** — each page becomes overlapping 900-char chunks (chunks never
   cross pages, so every citation has exactly one page number).
3. **Embed** — `all-MiniLM-L6-v2` turns chunks into 384-dim unit vectors.
4. **Index** — FAISS `IndexFlatIP` stores them; inner product on unit
   vectors = cosine similarity.
5. **Agents** — a LangGraph workflow runs three agents that share one state
   object: analyses → comparison → verification.
6. **Review** — a deterministic assembler writes the final report, listing
   only the sources that were actually retrieved.

See `PRESENTATION_GUIDE.md` for the full architecture explanation, a demo
script and viva Q&A.

---

## Honest limitations

- **No OCR** — scanned PDFs are rejected with a clear message.
- **Small model, simple parsing** — a 3B model can misformat replies; the
  parsers default to `UNKNOWN`/`INSUFFICIENT_EVIDENCE` rather than guessing.
- **Chunk-level retrieval** — retrieval sees 900-char chunks, not whole
  papers; very long papers may need more retrieved passages.
- **Citation verification is LLM-based** — it catches most drift but is not
  a mathematical guarantee; the retrieved-passage sources shown in the UI
  are the ground truth.
