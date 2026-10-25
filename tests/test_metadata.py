"""Tests for metadata preservation: filename, title and page numbers
must survive ingestion and chunking (requirement: never lose provenance)."""
import pytest

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

from conftest import make_pdf_bytes
from researchpilot.pdf_ingest import PdfIngestError, chunk_document, extract_document


def test_pages_numbered_correctly():
    data = make_pdf_bytes(["alpha page", "beta page", "gamma page"])
    doc = extract_document(data, "multi.pdf")
    assert doc.n_pages == 3
    assert [p.page for p in doc.pages] == [1, 2, 3]
    assert "alpha" in doc.pages[0].text
    assert "beta" in doc.pages[1].text
    assert "gamma" in doc.pages[2].text


def test_filename_preserved_on_every_page_and_chunk():
    data = make_pdf_bytes(["first page text", "second page text"])
    doc = extract_document(data, "keep_me.pdf")
    assert all(p.filename == "keep_me.pdf" for p in doc.pages)
    for chunk in chunk_document(doc):
        assert chunk.filename == "keep_me.pdf"
        assert chunk.source.startswith("keep_me.pdf, p. ")


def test_title_taken_from_pdf_metadata():
    data = make_pdf_bytes(
        ["Some body text about methods and datasets used in the study."],
        title="A Metadata Title",
    )
    doc = extract_document(data, "t.pdf")
    assert doc.title == "A Metadata Title"


def test_title_falls_back_to_first_line():
    data = make_pdf_bytes(
        ["Deep Learning for Astronomical Surveys", "", "Body text goes here."]
    )
    doc = extract_document(data, "t2.pdf")
    assert doc.title == "Deep Learning for Astronomical Surveys"


def test_title_falls_back_to_filename_when_no_plausible_line():
    data = make_pdf_bytes(["ok", "", "Body text goes here and more words for length."])
    doc = extract_document(data, "mystery.pdf")
    assert doc.title == "mystery.pdf"


def test_invalid_bytes_raise_clear_error():
    with pytest.raises(PdfIngestError, match="not a valid PDF"):
        extract_document(b"this is definitely not a pdf", "broken.pdf")


def test_empty_bytes_raise_clear_error():
    with pytest.raises(PdfIngestError):
        extract_document(b"", "empty.pdf")


def test_scanned_pdf_without_text_layer_raises():
    blank = make_pdf_bytes(["", "", ""])  # pages with no text at all
    with pytest.raises(PdfIngestError, match="scanned"):
        extract_document(blank, "scan.pdf")


def test_partially_scanned_pdf_warns_but_ingests():
    data = make_pdf_bytes(["Real methods text on page one.", ""])
    doc = extract_document(data, "mixed.pdf")
    assert doc.n_pages == 2
    assert any("page 2" in w for w in doc.warnings)
    assert all(c.page == 1 for c in chunk_document(doc))


def test_password_protected_pdf_raises():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 96), "secret text about methods")
    data = None
    try:
        data = doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    except TypeError:
        try:
            data = doc.tobytes(
                encryption=fitz.PDF_ENCRYPT_AES_256, user_password="pw", owner_password="pw"
            )
        except TypeError:
            pytest.skip("this pymupdf build cannot create encrypted test PDFs")
    finally:
        doc.close()
    with pytest.raises(PdfIngestError, match="password"):
        extract_document(data, "locked.pdf")
