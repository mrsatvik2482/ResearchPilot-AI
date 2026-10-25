"""Tests for FAISS retrieval using the deterministic FakeEmbedder (offline)."""
import pytest

from conftest import FakeEmbedder, make_pdf_bytes
from researchpilot.pdf_ingest import chunk_document, extract_document
from researchpilot.retriever import Retriever, dedupe
from researchpilot.vector_store import VectorStore


def _build_retriever():
    pdf1 = make_pdf_bytes(
        [
            "This page introduces the attention mechanism and transformer architecture "
            "for language tasks.",
            "The transformer architecture uses self attention layers stacked deeply "
            "for better representations.",
        ],
        title="Attention Paper",
    )
    pdf2 = make_pdf_bytes(
        [
            "This page describes a convolutional neural network for chest x-ray "
            "image classification.",
        ],
        title="CNN Paper",
    )
    chunks = []
    for name, data in (("attention.pdf", pdf1), ("cnn.pdf", pdf2)):
        chunks.extend(chunk_document(extract_document(data, name)))
    retriever = Retriever(FakeEmbedder())
    retriever.build(chunks)
    return retriever, chunks


def test_search_returns_relevant_chunk_with_source():
    retriever, _ = _build_retriever()
    hits = retriever.search("attention mechanism transformer", k=2)
    assert hits, "expected at least one hit"
    top = hits[0]
    assert "attention" in top.text.lower()
    assert top.filename == "attention.pdf"
    assert top.source == "attention.pdf, p. 1"
    assert 0.0 < top.score <= 1.0001


def test_search_empty_index_returns_nothing():
    retriever = Retriever(FakeEmbedder())
    retriever.build([])
    assert retriever.search("anything") == []
    assert retriever.n_chunks == 0


def test_search_for_paper_filters_by_file():
    retriever, _ = _build_retriever()
    hits = retriever.search_for_paper("cnn.pdf", "convolutional neural network", k=3)
    assert hits
    assert all(h.filename == "cnn.pdf" for h in hits)


def test_search_per_paper_covers_every_file():
    retriever, _ = _build_retriever()
    per_paper = retriever.search_per_paper("neural network", k=2)
    assert set(per_paper) == {"attention.pdf", "cnn.pdf"}
    assert all(passages for passages in per_paper.values())


def test_scores_sorted_descending():
    retriever, _ = _build_retriever()
    hits = retriever.search("neural network", k=4)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_dedupe_removes_repeated_passages():
    retriever, _ = _build_retriever()
    hits = retriever.search("attention", k=3)
    assert len(dedupe(hits + hits)) == len(dedupe(hits))


def test_vector_store_rejects_dimension_mismatch():
    import numpy as np

    from researchpilot.pdf_ingest import Chunk

    store = VectorStore(dim=8)
    chunks = [Chunk(text="hello", filename="a.pdf", title="A", page=1)]
    with pytest.raises(ValueError):
        store.add(chunks, np.ones((1, 16), dtype="float32"))
