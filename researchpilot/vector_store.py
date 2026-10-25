"""FAISS vector index over text chunks (exact cosine-similarity search).

IndexFlatIP performs a brute-force inner-product search. Because the
embeddings are unit-length, inner product == cosine similarity, and an
exact flat scan is perfectly accurate — the right choice for the small
document sets (hundreds of chunks) this prototype handles.
"""
from __future__ import annotations

import faiss
import numpy as np

from researchpilot.pdf_ingest import Chunk


class VectorStore:
    """A flat FAISS index that stores chunks next to their vectors."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product == cosine on unit vectors
        self.chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise ValueError("embeddings must be a 2-D array with one row per chunk")
        if chunks and embeddings.shape[1] != self.dim:
            raise ValueError(
                f"embedding dimension {embeddings.shape[1]} does not match "
                f"index dimension {self.dim}"
            )
        if chunks:
            self.index.add(embeddings)
            self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[tuple[Chunk, float]]:
        """Return the k most similar chunks with their cosine-similarity scores."""
        if not self.chunks:
            return []
        k = min(k, len(self.chunks))
        query = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        scores, ids = self.index.search(query, k)
        return [
            (self.chunks[i], float(score))
            for score, i in zip(scores[0], ids[0])
            if i != -1
        ]
