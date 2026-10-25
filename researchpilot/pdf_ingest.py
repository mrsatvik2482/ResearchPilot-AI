"""PDF ingestion.

Extracts text page by page with PyMuPDF, keeps provenance (filename, title,
page number) for every chunk, and splits pages into overlapping chunks so
context is not lost at chunk boundaries.

Scanned PDFs (images without a text layer) cannot be processed — OCR is out
of scope for this prototype, so they are reported as clear errors instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

try:  # PyMuPDF >= 1.24 recommends the new import name
    import pymupdf as fitz
except ImportError:  # pragma: no cover - older releases only expose "fitz"
    import fitz

from researchpilot.config import CHUNK_OVERLAP, CHUNK_SIZE


class PdfIngestError(Exception):
    """Raised when a PDF cannot be read or contains no usable text."""


@dataclass
class PageText:
    """Text of a single PDF page with its provenance."""

    filename: str
    page: int  # 1-based page number, as used in citations
    text: str


@dataclass
class IngestedDoc:
    """All extracted pages of one uploaded PDF."""

    filename: str
    title: str
    pages: list[PageText]
    warnings: list[str] = field(default_factory=list)

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    @property
    def n_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)


@dataclass
class Chunk:
    """A retrievable piece of text that always knows where it came from."""

    text: str
    filename: str
    title: str
    page: int

    @property
    def source(self) -> str:
        """Human-readable citation, e.g. 'attention.pdf, p. 3'."""
        return f"{self.filename}, p. {self.page}"


_MIN_USABLE_CHARS = 20  # fewer characters than this => treat as scanned


def _clean(text: str) -> str:
    """Collapse runs of spaces and drop empty lines."""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _detect_title(first_page_text: str) -> str | None:
    """Pick the first plausible title line from page 1.

    Heuristic: the first line that is long enough to be a title but short
    enough not to be a sentence, and does not end in a period.
    """
    for line in first_page_text.splitlines():
        line = line.strip()
        if 8 <= len(line) <= 150 and not line.endswith("."):
            return line
    return None


def extract_document(file_bytes: bytes, filename: str) -> IngestedDoc:
    """Extract per-page text from one PDF, preserving provenance."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:  # pymupdf raises several exception types
        raise PdfIngestError(f"'{filename}' is not a valid PDF: {exc}") from exc

    with doc:
        if doc.needs_pass:
            raise PdfIngestError(
                f"'{filename}' is password-protected and cannot be read."
            )
        if doc.page_count == 0:
            raise PdfIngestError(f"'{filename}' contains no pages.")

        pages: list[PageText] = []
        warnings: list[str] = []
        for number, page in enumerate(doc, start=1):
            text = _clean(page.get_text("text"))
            pages.append(PageText(filename=filename, page=number, text=text))
            if not text:
                warnings.append(
                    f"{filename}: page {number} has no extractable text "
                    "(possibly a scanned image page)."
                )

        if sum(len(p.text) for p in pages) < _MIN_USABLE_CHARS:
            raise PdfIngestError(
                f"'{filename}' looks like a scanned PDF: no extractable text was "
                "found. This prototype does not include OCR."
            )

        title = (doc.metadata or {}).get("title", "").strip()
        if not title or title.lower().startswith("untitled"):
            title = _detect_title(pages[0].text) or filename

    return IngestedDoc(filename=filename, title=title, pages=pages, warnings=warnings)


def chunk_document(
    doc: IngestedDoc,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split every page into overlapping chunks that never cross pages.

    Keeping chunks inside a single page means every chunk has exactly one
    page number, which makes citations unambiguous.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    step = max(1, chunk_size - overlap)  # guard against overlap >= chunk_size

    chunks: list[Chunk] = []
    for page in doc.pages:
        text = page.text
        if not text:
            continue
        if len(text) <= chunk_size:
            chunks.append(
                Chunk(text=text, filename=doc.filename, title=doc.title, page=page.page)
            )
            continue
        start = 0
        while start < len(text):
            piece = text[start : start + chunk_size].strip()
            if piece:
                chunks.append(
                    Chunk(text=piece, filename=doc.filename, title=doc.title, page=page.page)
                )
            if start + chunk_size >= len(text):
                break
            start += step
    return chunks


def ingest_pdf(
    file_bytes: bytes,
    filename: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> tuple[IngestedDoc, list[Chunk]]:
    """Convenience wrapper: extract + chunk one PDF."""
    doc = extract_document(file_bytes, filename)
    return doc, chunk_document(doc, chunk_size, overlap)
