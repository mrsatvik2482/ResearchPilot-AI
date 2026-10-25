"""Run the full pipeline from the command line (no UI) — also serves as an
end-to-end integration check.

Usage:
    python scripts/run_e2e.py sample_pdfs/sample_paper_1_transformers.pdf \
        sample_pdfs/sample_paper_2_leaf_disease.pdf \
        "How are neural networks applied in classification tasks?"

Requires Ollama to be running with a pulled model (see README). The
literature review is printed and saved to output/literature_review.md.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from researchpilot.agents.graph import AGENT_LABELS, build_graph, initial_state  # noqa: E402
from researchpilot.embeddings import Embedder  # noqa: E402
from researchpilot.llm import OllamaClient  # noqa: E402
from researchpilot.pdf_ingest import ingest_pdf  # noqa: E402
from researchpilot.review import build_review  # noqa: E402
from researchpilot.retriever import Retriever  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2

    pdf_paths = [Path(p) for p in argv[:-1]]
    question = argv[-1]

    client = OllamaClient()
    ok, message = client.health()
    print(message)
    if not ok:
        return 1

    chunks, filenames = [], []
    for path in pdf_paths:
        if not path.exists():
            print(f"ERROR: {path} not found")
            return 1
        doc, doc_chunks = ingest_pdf(path.read_bytes(), path.name)
        print(f"ingested {doc.filename}: {doc.n_pages} pages, {len(doc_chunks)} chunks")
        chunks.extend(doc_chunks)
        filenames.append(doc.filename)

    print("building embedding index (first run downloads the model)…")
    retriever = Retriever(Embedder())
    retriever.build(chunks)

    graph = build_graph()
    state = initial_state(question, filenames, retriever, client)
    final = dict(state)
    for update in graph.stream(state, stream_mode="updates"):
        for node_name, node_output in update.items():
            final.update(node_output or {})
            print(f"[done] {AGENT_LABELS.get(node_name, node_name)}")

    review = build_review(final, llm=client)

    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "literature_review.md"
    out_path.write_text(review, encoding="utf-8")

    print("\n" + "=" * 70)
    print(review)
    print("=" * 70)
    print(f"\nsaved to {out_path}")
    for error in final.get("errors", []):
        print(f"ERROR: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
