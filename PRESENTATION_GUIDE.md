# 🎓 ResearchPilot AI — Presentation & Viva Guide

This document explains **what you built, why each piece works the way it
does, and how to defend it in a viva**. Read it once before the demo and
once before the viva.

---

## 1. The one-sentence pitch

> "ResearchPilot AI is a fully local multi-agent research assistant: it
> ingests research-paper PDFs, indexes them with semantic search (RAG), and
> three cooperating LangGraph agents — analysis, comparison, and citation
> verification — produce a literature review that only cites pages that
> actually exist in the uploaded PDFs."

Why it matters: no API keys, no cloud, no cost, works offline — ideal for
research assistance on private papers.

---

## 2. Architecture

```
┌────────────┐   ┌──────────────┐   ┌──────────────────────────────┐
│  PDF files │──►│ PyMuPDF      │──►│ Chunks (900 chars, 150 ov.)  │
└────────────┘   │ page-by-page │   │ each keeps: filename, title, │
                 │ extraction   │   │ page number                  │
                 └──────────────┘   └──────────────┬───────────────┘
                                                    │ SentenceTransformers
                                                    │ all-MiniLM-L6-v2 (384-d)
                                                    ▼
┌──────────────────────────────────────────────────────────────────┐
│ FAISS IndexFlatIP — cosine similarity (vectors are unit-length)  │
│ one global index + one index per uploaded paper                  │
└───────────────────────────────────┬──────────────────────────────┘
                                    │ retrieved passages + sources
                                    ▼
┌─────────────────────── LangGraph workflow ───────────────────────┐
│                                                                  │
│  START ─► [Paper Analysis Agent] ─► [Comparison Agent]           │
│                        │                   │                     │
│                        ▼                   ▼                     │
│              shared AgentState     ─► [Citation Verification     │
│              (question, analyses,       Agent] ─► END            │
│               comparison, errors)                                 │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
                 ┌───────────────────────────┐
                 │ Report assembler (review.py)│
                 │ Introduction / Summaries /  │
                 │ Comparison / Gaps (tentative)│
                 │ Verification / Conclusion /  │
                 │ References (real pages only)│
                 └───────────────────────────┘
```

**Key design decision:** the agents are *not* the same prompt three times.
Each agent has a different prompt, different parsing, and reads/writes
different fields of the shared state. The Comparison Agent literally
consumes the structured output of the Paper Analysis Agent (there is a unit
test proving it: `test_comparison_agent_consumes_analysis_output`).

---

## 3. RAG explained (for the viva)

**RAG = Retrieval-Augmented Generation.** Instead of asking an LLM to answer
from its (unreliable, possibly hallucinated) memory, we first *retrieve* the
most relevant passages from the user's own PDFs and force the LLM to answer
only from those passages.

Steps in this project:

1. **Chunking** — pages are split into 900-character chunks with 150
   characters of overlap. Overlap prevents a sentence cut in half from
   losing context. Chunks never cross page boundaries → every chunk has
   exactly one page number → citations are unambiguous.
2. **Embedding** — `all-MiniLM-L6-v2` maps each chunk to a 384-dim vector.
   Texts with similar *meaning* land close together in vector space (not
   just similar words — "tiny transformers" is close to "small attention
   models"). Vectors are normalised to unit length.
3. **Vector search** — FAISS `IndexFlatIP` computes the inner product of
   the query vector with every chunk vector. On unit vectors, inner product
   = **cosine similarity** (1 = identical meaning, 0 = unrelated). Flat =
   exact brute-force search; for hundreds of chunks this is instant and
   100% accurate.
4. **Generation** — the retrieved passages are pasted into the agent prompts
   with their sources (`file.pdf, p. 3`), and a strict system prompt says:
   *answer only from the passages, else say UNKNOWN*.

---

## 4. The three agents (LangGraph)

**LangGraph** is a library for building LLM workflows as a graph: nodes
(functions) pass through a shared **state** object, edges define the
execution order. Our graph is linear — `START → analysis → comparison →
verification → END` — but the *state* is what makes it a real multi-agent
system:

| Agent | Reads from state | Writes to state | LLM calls |
|---|---|---|---|
| Paper Analysis | question, filenames, retriever, llm | `paper_analyses` (4 structured fields per paper + evidence) | 1 per paper |
| Comparison | `paper_analyses` (from agent 1!), retriever | `comparison` (narrative, datasets, gaps) | 1 |
| Citation Verification | `paper_analyses` (claims), retriever | `verification` (verdict per claim) | 1 per claim (≤12) |

State fields live in `researchpilot/agents/state.py` — point at it in the
viva as "the explicit data contract between agents".

**Agent 1 — Paper Analysis.** For each paper: retrieve the 6 passages most
relevant to (the user's question + section-steering words), ask the LLM for
four labelled sections (research question / methodology / findings /
limitations). A tolerant parser reads the sections; anything missing or
unparseable becomes `UNKNOWN` — never invented.

**Agent 2 — Comparison.** Takes agent 1's structured summaries for ALL
papers, retrieves fresh cross-paper evidence, and produces a comparison
narrative + per-paper dataset lines + 2–4 research gaps. Every gap is
labelled *tentative* in the UI and the report.

**Agent 3 — Citation Verification.** Takes the methodology and findings
claims produced earlier, retrieves the passages most similar to *each
claim*, and asks the LLM: is this claim directly supported by these
passages? Verdicts: `SUPPORTED`, `UNSUPPORTED`, `INSUFFICIENT_EVIDENCE`.
Unparseable replies default to `INSUFFICIENT_EVIDENCE` — the system never
*assumes* support.

---

## 5. Citation integrity (the anti-hallucination story)

This is the strongest part to present:

1. **Grounding system prompt** on every LLM call: answer only from the
   passages; never invent papers, authors, numbers, quotes or page numbers;
   say UNKNOWN otherwise.
2. **UNKNOWN, not guesses** — unparseable/missing fields become `UNKNOWN`
   in analyses and comparisons.
3. **Page numbers come from retrieval, not the LLM** — a citation's page
   number is attached to a chunk by PyMuPDF during ingestion. The LLM never
   types a page number; it can only echo a source string we gave it.
4. **Verification agent** re-checks claims against retrieved passages and
   flags unsupported ones visibly (red/amber/green badges).
5. **References = union of actually-retrieved evidence** — the reference
   list is computed from the `evidence` lists attached to each analysis,
   i.e. only real uploaded files and real page numbers.
6. **Weak-match warning** — if the top similarity score is below 0.30, the
   UI warns that the papers may not relate to the question.

---

## 6. Demo script (≈4 minutes)

1. **Setup line** — "Everything runs locally: Streamlit UI, FAISS search,
   a 90 MB embedding model, and a 3-billion-parameter LLM via Ollama."
2. **Sidebar** — show Ollama status green (model `llama3.2:3b`).
3. **Upload** — drag in the two PDFs from `sample_pdfs/` (or your own).
   Show the document table: detected titles, pages, chunks. Mention: a
   scanned PDF would be rejected with a clear message.
4. **Ask** — type *"How are neural networks applied in classification
   tasks?"* and press **Generate literature review**. Narrate the progress:
   the three agents finishing one by one.
5. **Review tab** — scroll: introduction, summaries with page citations,
   comparison, tentative gaps, verification table, references.
6. **Comparison tab** — the structured table + datasets.
7. **Citation Verification tab** — point at the SUPPORTED badges: "every
   claim is checked against the actual PDF text".
8. **Retrieved Evidence tab** — "here are the exact passages and pages the
   agents used — nothing hidden."
9. **Honesty beat** — upload an unrelated PDF and ask a question about a
   different topic: the app warns about weak matches and says information
   is unknown instead of inventing it.

If Ollama is missing on the demo machine, the app shows an exact fix
message — you can even show that as an error-handling feature.

---

## 7. Likely viva questions & answers

**Q1. Why local-only? What's the advantage over ChatGPT/Claude?**
Privacy (papers never leave the machine), zero cost, reproducibility for
the project, offline use, and no API-key dependency. Trade-off: smaller
model, weaker writing — which is exactly why the citation-verification
agent matters.

**Q2. What is an embedding?**
A fixed-size numeric vector (384 numbers here) that represents a text's
*meaning*. Similar meanings → nearby vectors, so similarity becomes
arithmetic (cosine similarity). `all-MiniLM-L6-v6` is a 22M-parameter
sentence transformer distilled for speed.

**Q3. Why cosine similarity and not Euclidean distance?**
Cosine measures the angle between vectors — it ignores vector length, so a
long chunk doesn't look "far" just for being long. We L2-normalise every
vector, after which cosine = inner product, which is what FAISS's
`IndexFlatIP` computes.

**Q4. Why FAISS? Why a flat index?**
FAISS is Facebook AI's open-source vector search library, the industry
standard. `IndexFlatIP` does exact brute-force search — for a few hundred
chunks it's instant and cannot miss. For millions of vectors you'd switch
to an approximate index (IVF/HNSW) trading a little accuracy for speed.

**Q5. Why 900-character chunks with 150 overlap?**
Big enough to carry a complete argument (~200 tokens), small enough that a
retrieved passage stays focused and fits several times into the LLM's 4K
context. Overlap prevents losing a sentence that straddles a boundary.
Chunks stay within one page so citations have exactly one page number.

**Q6. What is LangGraph and why use it instead of plain functions?**
LangGraph models the workflow as a graph of nodes over a shared typed
state. Benefits: explicit data flow (you can *see* agent 2 consuming agent
1's output), free orchestration (streaming, retries, future parallelism),
and a visual structure that maps to the report. Plain functions would hide
the data flow.

**Q7. How do you prevent hallucination?**
Five layers: grounding system prompt; UNKNOWN instead of guessing; page
numbers come from retrieval metadata, never from the LLM; an entire
verification agent that re-checks claims against passages; references built
only from actually-retrieved evidence. (Detail in §5 above.)

**Q8. What happens if the LLM output is garbage?**
Tolerant parsers look for section headers; anything missing becomes
UNKNOWN. Verification replies that can't be parsed become
INSUFFICIENT_EVIDENCE. There are unit tests for exactly this
(`test_unparseable_llm_reply_yields_unknown_not_invented`).

**Q9. What if Ollama isn't running?**
A health check (`/api/version`, `/api/tags`) runs before generation and the
sidebar shows an exact fix message ("start ollama serve / run ollama pull
llama3.2:3b"). The app degrades gracefully — uploads and indexing still
work.

**Q10. Why does the scanned-PDF case fail?**
Scanned PDFs are images; text extraction returns nothing. We detect it
(total extracted characters < 20) and show a clear message. OCR (e.g.
Tesseract) is listed as future work — it would slot in right before
chunking.

**Q11. How do the agents share data — concretely?**
A `TypedDict` state (`AgentState`). Node 1 returns `{"paper_analyses": …,
"errors": …}`; LangGraph merges it into the state; node 2 reads
`state["paper_analyses"]` and the test suite asserts agent 1's text is
literally inside agent 2's prompt.

**Q12. Why does the report say gaps are "tentative"?**
Research gaps are *inferences* from limitations, not statements the papers
make. Labelling them tentative is the honest thing to do — and it shows the
examiner you understand the difference between retrieval and inference.

**Q13. What model sizes/limits? Why llama3.2:3b?**
~2 GB download, runs comfortably on 16 GB RAM CPU-only, good instruction
following. `num_ctx=4096` bounds memory; `temperature=0.2` keeps output
factual; `num_predict` caps response length per agent.

**Q14. How is this tested without an LLM?**
The Ollama client is replaced by a `ScriptedLLM` mock that returns
canned, realistic agent replies; embeddings are replaced by a deterministic
hash-based fake. So the whole pipeline (including LangGraph data flow) is
tested offline in under a second — 40 tests.

**Q15. What are the biggest limitations?**
No OCR; chunk-level (not document-level) retrieval; a 3B model's writing
quality; verification is LLM-based (strong heuristic, not a proof); no
author/date parsing from PDFs (citations use filename + page).

**Q16. What would you add next?**
OCR for scanned PDFs; author/year metadata parsing for real bibliographic
references; streaming LLM output; an approximate FAISS index (HNSW) for
large libraries; exporting the review to DOCX/PDF.

---

## 8. Numbers worth memorising

- Embedding model: `all-MiniLM-L6-v2`, **384 dimensions**, ~90 MB.
- Chunks: **900 chars, 150 overlap**, never cross pages.
- Retrieval: **top-6** passages per paper for analysis, top-4 per claim for
  verification, weak-match warning below **0.30** cosine.
- LLM: `llama3.2:3b`, temperature **0.2**, context **4096**.
- Tests: **40**, all offline, < 1 second.
- Verified claims per run: up to **12** (methodology + findings per paper).

---

## 9. File-to-feature map (for "show me the code" moments)

| Feature | File |
|---|---|
| PDF extraction & error handling | `researchpilot/pdf_ingest.py` |
| Overlapping chunking | `researchpilot/pdf_ingest.py` → `chunk_document` |
| Embeddings | `researchpilot/embeddings.py` |
| FAISS index | `researchpilot/vector_store.py` |
| Retrieval + citations | `researchpilot/retriever.py` |
| Ollama client + grounding prompt | `researchpilot/llm.py` |
| Agent state (data contract) | `researchpilot/agents/state.py` |
| Agent 1 / 2 / 3 | `researchpilot/agents/analysis.py`, `comparison.py`, `citation.py` |
| Graph wiring | `researchpilot/agents/graph.py` |
| Report assembly & references | `researchpilot/review.py` |
| Dashboard | `app.py` |
| Mocked-LLM tests | `conftest.py` + `tests/` |
