"""Semantic retrieval: embed a query, search FAISS, return cited passages."""
from __future__ import annotations

from dataclasses import dataclass

from researchpilot.config import PER_PAPER_K, TOP_K
from researchpilot.pdf_ingest import Chunk
from researchpilot.vector_store import VectorStore


@dataclass
class RetrievedPassage:
    """A chunk that was retrieved for a query, with its score and source."""

    text: str
    filename: str
    title: str
    page: int
    score: float  # cosine similarity, 0..1

    @property
    def source(self) -> str:
        """Human-readable citation, e.g. 'paper1.pdf, p. 3'."""
        return f"{self.filename}, p. {self.page}"

    @classmethod
    def from_chunk(cls, chunk: Chunk, score: float) -> "RetrievedPassage":
        return cls(
            text=chunk.text,
            filename=chunk.filename,
            title=chunk.title,
            page=chunk.page,
            score=score,
        )


def dedupe(passages: list[RetrievedPassage]) -> list[RetrievedPassage]:
    """Drop duplicate passages (same text from the same source)."""
    seen: set[tuple[str, str]] = set()
    out: list[RetrievedPassage] = []
    for passage in passages:
        key = (passage.source, passage.text)
        if key not in seen:
            seen.add(key)
            out.append(passage)
    return out


class Retriever:
    """Owns one global FAISS index plus one index per uploaded file.

    Per-file indexes let each agent retrieve evidence from a specific paper
    without mixing in passages from the other papers.
    """

    def __init__(self, embedder):
        self.embedder = embedder
        self._global: VectorStore | None = None
        self._per_file: dict[str, VectorStore] = {}

    # -- build ------------------------------------------------------------
    def build(self, chunks: list[Chunk]) -> None:
        """Embed all chunks once and build the global and per-file indexes."""
        self._per_file = {}
        if not chunks:
            self._global = None
            return
        vectors = self.embedder.encode([c.text for c in chunks])
        dim = int(vectors.shape[1])
        self._global = VectorStore(dim)
        self._global.add(chunks, vectors)
        for i, chunk in enumerate(chunks):
            store = self._per_file.setdefault(chunk.filename, VectorStore(dim))
            store.add([chunk], vectors[i : i + 1])

    # -- queries ------------------------------------------------------------
    @property
    def filenames(self) -> list[str]:
        return list(self._per_file)

    @property
    def n_chunks(self) -> int:
        return len(self._global) if self._global is not None else 0

    def search(self, query: str, k: int = TOP_K) -> list[RetrievedPassage]:
        """Search across every uploaded paper."""
        if self._global is None or not self._global.chunks:
            return []
        query_vector = self.embedder.encode([query])[0]
        return [
            RetrievedPassage.from_chunk(chunk, score)
            for chunk, score in self._global.search(query_vector, k)
        ]

    def search_for_paper(
        self, filename: str, query: str, k: int = PER_PAPER_K
    ) -> list[RetrievedPassage]:
        """Search only within one paper (used by the analysis and verification agents)."""
        store = self._per_file.get(filename)
        if store is None:
            return []
        query_vector = self.embedder.encode([query])[0]
        return [
            RetrievedPassage.from_chunk(chunk, score)
            for chunk, score in store.search(query_vector, k)
        ]

    def search_per_paper(self, query: str, k: int = 3) -> dict[str, list[RetrievedPassage]]:
        """Search every paper separately; returns {filename: passages}."""
        return {
            filename: self.search_for_paper(filename, query, k)
            for filename in self._per_file
        }
