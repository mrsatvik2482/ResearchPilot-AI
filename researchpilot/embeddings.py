"""Local text embeddings via Sentence Transformers.

The model (all-MiniLM-L6-v2, ~90 MB) is downloaded once from Hugging Face
and cached locally; after that no internet is needed. Vectors are
L2-normalised so that an inner product in FAISS equals cosine similarity.
"""
from __future__ import annotations

import numpy as np

from researchpilot.config import EMBEDDING_MODEL_NAME


class Embedder:
    """Wraps a local Sentence Transformers model.

    Any object with the same ``encode(texts) -> float32 ndarray`` interface
    (e.g. the deterministic FakeEmbedder used in tests) can be used in its
    place — the rest of the code only relies on that contract.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        # Imported lazily so that tests (which use a fake embedder) never
        # pay the cost of importing torch.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def dim(self) -> int:
        try:  # sentence-transformers >= 6 renamed this method
            return int(self._model.get_embedding_dimension())
        except AttributeError:  # pragma: no cover - older versions
            return int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of unit-length vectors."""
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        vectors = self._model.encode(
            list(texts), convert_to_numpy=True, show_progress_bar=False
        )
        vectors = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vectors / norms
