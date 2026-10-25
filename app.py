"""ResearchPilot AI — Multi-Agent Research Assistant Using RAG (Streamlit UI).

Run with:  streamlit run app.py
Everything runs locally: PyMuPDF ingestion, Sentence Transformers
embeddings, FAISS retrieval, a LangGraph agent workflow and Ollama for
local LLM inference. No API keys, no cloud calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make `import researchpilot` work no matter where streamlit is launched from.
sys.path.insert(0, str(Path(__file__).parent))

from researchpilot.agents.graph import (  # noqa: E402
    AGENT_LABELS,
    PAPER_ANALYSIS_NODE,
    build_graph,
    initial_state,
)
from researchpilot.config import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_OLLAMA_MODEL,
    EMBEDDING_MODEL_NAME,
    OLLAMA_BASE_URL,
    WEAK_MATCH_SCORE,
)
from researchpilot.embeddings import Embedder  # noqa: E402
from researchpilot.llm import OllamaClient  # noqa: E402
from researchpilot.pdf_ingest import (  # noqa: E402
    PdfIngestError,
    chunk_document,
    extract_document,
)
from researchpilot.review import build_review  # noqa: E402
from researchpilot.retriever import Retriever  # noqa: E402

st.set_page_config(page_title="ResearchPilot AI", page_icon="🔬", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.8rem; max-width: 1250px;}
      h1, h2, h3 {letter-spacing: 0.2px;}
      .rp-badge {padding: 0.12rem 0.6rem; border-radius: 999px; font-size: 0.78rem;
                  font-weight: 700; color: #0b0f14;}
      .rp-ok   {background: #2ea043;}
      .rp-bad  {background: #f85149;}
      .rp-warn {background: #d29922;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_embedder() -> Embedder:
    """Load the embedding model once (downloads on first ever run, then cached)."""
    return Embedder(EMBEDDING_MODEL_NAME)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🔬 ResearchPilot AI")
    st.caption("Local-first multi-agent research assistant — RAG + LangGraph + Ollama")
    st.divider()

    st.subheader("Local LLM (Ollama)")
    probe = OllamaClient()
    installed_models = probe.list_models()
    model_options = sorted(installed_models) or [DEFAULT_OLLAMA_MODEL]
    default_index = (
        model_options.index(DEFAULT_OLLAMA_MODEL)
        if DEFAULT_OLLAMA_MODEL in model_options
        else 0
    )
    selected_model = st.selectbox(
        "Model",
        model_options,
        index=default_index,
        help="Models installed in your local Ollama. Recommended: llama3.2:3b",
    )
    client = OllamaClient(OLLAMA_BASE_URL, model=selected_model)
    llm_ok, llm_message = client.health()
    if llm_ok:
        st.success(llm_message)
    else:
        st.error(llm_message)
        st.caption(
            "Install Ollama once (https://ollama.com/download), keep it running, "
            f"then run: `ollama pull {selected_model}`"
        )

    st.divider()
    st.subheader("Embedding model")
    st.caption(
        f"`{EMBEDDING_MODEL_NAME}` — downloads once (~90 MB) on first use, "
        "then runs fully offline."
    )

    st.divider()
    st.subheader("Pipeline")
    st.markdown(
        "1. Upload PDFs (PyMuPDF)\n"
        "2. Chunk + embed (Sentence Transformers)\n"
        "3. Vector search (FAISS)\n"
        "4. 3 agents (LangGraph + Ollama)\n"
        "5. Cited literature review"
    )
    st.caption(
        f"Chunks: {CHUNK_SIZE} chars with {CHUNK_OVERLAP} overlap · "
        "All processing stays on this machine."
    )

# ------------------------------------------------------------------ header
st.title("ResearchPilot AI — Multi-Agent Research Assistant")
st.caption(
    "Upload research papers, ask a question, and get a cited literature review "
    "generated entirely on your machine."
)

# ------------------------------------------------------------------ 1. upload
st.header("1 · Upload research papers")
uploaded_files = st.file_uploader(
    "Research paper PDFs", type="pdf", accept_multiple_files=True,
    label_visibility="collapsed",
)

signature = tuple((f.name, f.size) for f in uploaded_files) if uploaded_files else ()


def process_uploads(uploads):
    """Extract text and chunk every uploaded PDF, collecting errors/warnings."""
    docs, all_chunks, errors, warnings = [], [], [], []
    for upload in uploads:
        try:
            doc = extract_document(upload.getvalue(), upload.name)
        except PdfIngestError as exc:
            errors.append(str(exc))
            continue
        docs.append(doc)
        warnings.extend(doc.warnings)
        all_chunks.extend(chunk_document(doc))
    return docs, all_chunks, errors, warnings


if uploaded_files and st.session_state.get("upload_signature") != signature:
    bar = st.progress(0.05, "Extracting PDF text with PyMuPDF…")
    docs, chunks, ingest_errors, ingest_warnings = process_uploads(uploaded_files)

    retriever = None
    if chunks:
        bar.progress(
            0.45, f"Embedding {len(chunks)} chunks with {EMBEDDING_MODEL_NAME}…"
        )
        try:
            embedder = load_embedder()
            retriever = Retriever(embedder)
            retriever.build(chunks)
            bar.progress(1.0, f"Search index ready — {len(chunks)} chunks in FAISS.")
        except Exception as exc:  # network issue on first download, disk, etc.
            ingest_errors.append(f"Could not build the search index: {exc}")
    else:
        bar.progress(1.0, "No usable text found.")

    for key, value in [
        ("docs", docs),
        ("chunks", chunks),
        ("ingest_errors", ingest_errors),
        ("ingest_warnings", ingest_warnings),
        ("retriever", retriever),
        ("upload_signature", signature),
    ]:
        st.session_state[key] = value

docs = st.session_state.get("docs", [])
chunks = st.session_state.get("chunks", [])
retriever = st.session_state.get("retriever")

if docs:
    table = pd.DataFrame(
        [
            {
                "File": d.filename,
                "Detected title": d.title,
                "Pages": d.n_pages,
                "Characters": d.n_chars,
                "Chunks": sum(1 for c in chunks if c.filename == d.filename),
            }
            for d in docs
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
else:
    st.info(
        "Upload one or more research paper PDFs to begin. Two synthetic sample "
        "papers are included in the `sample_pdfs/` folder for a quick demo."
    )

for error in st.session_state.get("ingest_errors", []):
    st.error(error)
for warning in st.session_state.get("ingest_warnings", []):
    st.warning(warning)

# ------------------------------------------------------------------ 2. question
st.header("2 · Ask a research question")
question = st.text_input(
    "Research question or topic",
    value=st.session_state.get("question", ""),
    placeholder="e.g. How are neural networks used for text classification?",
)
st.session_state["question"] = question
generate_clicked = st.button(
    "🚀 Generate literature review",
    type="primary",
    disabled=not (docs and retriever is not None),
)


def run_pipeline(question_text: str, llm_client: OllamaClient) -> None:
    """Run the 3-agent LangGraph workflow, then assemble the review."""
    filenames = [d.filename for d in docs]
    state = initial_state(question_text, filenames, retriever, llm_client)
    graph = build_graph()

    final = dict(state)
    bar = st.progress(0.05, "Starting multi-agent workflow…")
    with st.status("🔬 Multi-agent workflow", expanded=True) as status:
        st.markdown(f"**Question:** {question_text}")
        try:
            done = 0
            for update in graph.stream(state, stream_mode="updates"):
                for node_name, node_output in update.items():
                    node_output = node_output or {}
                    final.update(node_output)
                    done += 1
                    bar.progress(
                        min(done / 3, 1.0), f"{AGENT_LABELS.get(node_name, node_name)} finished"
                    )
                    st.markdown(f"✅ **{AGENT_LABELS.get(node_name, node_name)}** finished")
            status.update(label="Workflow complete", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Workflow crashed", state="error", expanded=True)
            st.error(f"Unexpected error while running the agents: {exc}")
            return

    bar.progress(0.95, "Writing the literature review…")
    try:
        review_markdown = build_review(final, llm=llm_client)
    except Exception:
        review_markdown = build_review(final, llm=None)
    bar.progress(1.0, "Done.")

    st.session_state["results"] = {
        "question": question_text,
        "state": final,
        "review": review_markdown,
    }


if generate_clicked:
    if not question.strip():
        st.warning("Please enter a research question first.")
    else:
        ok, message = client.health()
        if not ok:
            st.error(message)
            st.info(
                "**How to fix:**\n"
                "1. Install Ollama: https://ollama.com/download (one time)\n"
                "2. Make sure it is running (the desktop app, or `ollama serve`)\n"
                f"3. Download the model: `ollama pull {client.model}`\n"
                "4. Press **Generate** again."
            )
        else:
            run_pipeline(question.strip(), client)

# ------------------------------------------------------------------ 3. results
results = st.session_state.get("results")
if results:
    st.header("3 · Results")
    state = results["state"]
    analyses = state.get("paper_analyses", {})
    comparison = state.get("comparison")
    verification = state.get("verification", [])
    errors = state.get("errors", [])
    warnings = state.get("warnings", [])

    for error in errors:
        st.error(error)
    for warning in warnings:
        st.warning(warning)

    # Weak-match warning: be honest when the papers barely relate to the question.
    top_scores = [a.evidence[0].score for a in analyses.values() if a.evidence]
    if top_scores and max(top_scores) < WEAK_MATCH_SCORE:
        st.warning(
            "Semantic matches are weak — the uploaded papers may not relate "
            "closely to this question. Interpret the review with caution."
        )

    tab_review, tab_papers, tab_compare, tab_verify, tab_evidence = st.tabs(
        [
            "📄 Literature Review",
            "🧪 Paper Analyses",
            "⚖️ Comparison",
            "✅ Citation Verification",
            "🔎 Retrieved Evidence",
        ]
    )

    with tab_review:
        st.markdown(results["review"])
        st.download_button(
            "⬇️ Download review (Markdown)",
            results["review"],
            file_name="literature_review.md",
            mime="text/markdown",
        )

    with tab_papers:
        if not analyses:
            st.info("No analyses were produced in this run.")
        for filename, analysis in analyses.items():
            with st.expander(f"📄 {filename} — {analysis.title}"):
                for label, value in [
                    ("Research question", analysis.research_question),
                    ("Methodology", analysis.methodology),
                    ("Findings", analysis.findings),
                    ("Limitations", analysis.limitations),
                ]:
                    st.markdown(f"**{label}:** {value}")
                if analysis.evidence:
                    st.markdown("**Evidence passages used**")
                    for passage in analysis.evidence:
                        st.markdown(
                            f"- `{passage.source}` · similarity {passage.score:.2f}"
                        )

    with tab_compare:
        if comparison is None:
            st.info("No comparison was produced in this run.")
        else:
            if len(analyses) >= 2:
                rows = [
                    {
                        "Paper": filename,
                        "Research question": a.research_question,
                        "Methodology": a.methodology,
                        "Findings": a.findings,
                        "Limitations": a.limitations,
                    }
                    for filename, a in analyses.items()
                ]
                st.markdown("**Structured comparison** (from the Paper Analysis Agent)")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.markdown("**Narrative** (from the Comparison Agent)")
            st.markdown(comparison.narrative or "_Not available._")
            if comparison.datasets:
                st.markdown("**Datasets**")
                st.table(
                    pd.DataFrame(
                        comparison.datasets.items(), columns=["Paper", "Dataset(s)"]
                    )
                )
            if comparison.gaps:
                st.markdown("**Research gaps — tentative (inferred, not stated)**")
                for gap in comparison.gaps:
                    st.markdown(f"- ⚠️ {gap} _(tentative)_")
            else:
                st.markdown(
                    "**Research gaps:** none could be identified from the "
                    "retrieved evidence."
                )

    with tab_verify:
        if not verification:
            st.info(
                "No claims were verified in this run (no LLM available, or nothing "
                "to verify)."
            )
        colors = {
            "SUPPORTED": "rp-ok",
            "UNSUPPORTED": "rp-bad",
            "INSUFFICIENT_EVIDENCE": "rp-warn",
        }
        for check in verification:
            badge_class = colors.get(check.verdict, "rp-warn")
            st.markdown(
                f'<span class="rp-badge {badge_class}">{check.verdict}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Claim:** {check.claim}")
            st.caption(f"Why: {check.reason} · Source: {check.source}")

    with tab_evidence:
        seen: dict[tuple[str, str], object] = {}
        for analysis in analyses.values():
            for passage in analysis.evidence:
                seen[(passage.source, passage.text)] = passage
        if comparison:
            for passage in comparison.evidence:
                seen.setdefault((passage.source, passage.text), passage)
        if not seen:
            st.info("No passages were retrieved in this run.")
        for passage in seen.values():
            with st.expander(f"{passage.source} · similarity {passage.score:.2f}"):
                st.markdown(passage.text)
