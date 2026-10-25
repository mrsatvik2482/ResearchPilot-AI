"""Tests for overlapping chunking (requirement: split text into overlapping chunks)."""
from conftest import make_pdf_bytes
from researchpilot.pdf_ingest import chunk_document, extract_document


def _long_page() -> str:
    sentences = [
        f"Sentence number {i} talks about transformers, attention, and datasets."
        for i in range(80)
    ]
    return " ".join(sentences)  # ~6k characters


def test_short_page_becomes_single_chunk():
    data = make_pdf_bytes(["A short page about cats."])
    doc = extract_document(data, "short.pdf")
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "A short page about cats."
    assert chunks[0].page == 1


def test_long_page_split_into_overlapping_chunks():
    data = make_pdf_bytes([_long_page()])
    doc = extract_document(data, "long.pdf")
    chunks = chunk_document(doc, chunk_size=500, overlap=100)
    assert len(chunks) > 1
    for chunk in chunks:
        assert 0 < len(chunk.text) <= 500


def test_consecutive_chunks_overlap():
    data = make_pdf_bytes([_long_page()])
    doc = extract_document(data, "overlap.pdf")
    chunks = chunk_document(doc, chunk_size=500, overlap=100)
    # the tail of chunk 1 must reappear near the head of chunk 2
    assert chunks[0].text[-60:] in chunks[1].text


def test_coverage_no_words_lost():
    text = _long_page()
    data = make_pdf_bytes([text])
    doc = extract_document(data, "coverage.pdf")
    chunks = chunk_document(doc, chunk_size=500, overlap=100)
    joined = " ".join(c.text for c in chunks)
    for word in {w.strip(".,") for w in text.split()}:
        if len(word) > 3:
            assert word in joined, f"word '{word}' lost during chunking"


def test_chunks_never_cross_page_boundaries():
    data = make_pdf_bytes(
        [
            "Page one has plenty of text about neural networks and attention mechanisms.",
            "Page two discusses convolutional networks for image classification tasks.",
        ]
    )
    doc = extract_document(data, "two_pages.pdf")
    chunks = chunk_document(doc, chunk_size=60, overlap=15)
    assert len(chunks) > 2  # both pages were actually split
    for chunk in chunks:
        # each chunk must be a substring of exactly its own page
        assert chunk.text in doc.pages[chunk.page - 1].text


def test_empty_pages_produce_no_chunks():
    data = make_pdf_bytes(["Real text here about methods.", ""])
    doc = extract_document(data, "mixed.pdf")
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert all(c.page == 1 for c in chunks)


def test_overlap_larger_than_chunk_size_is_safe():
    data = make_pdf_bytes([_long_page()])
    doc = extract_document(data, "guard.pdf")
    chunks = chunk_document(doc, chunk_size=200, overlap=300)  # step clamps to 1
    assert len(chunks) > 1
