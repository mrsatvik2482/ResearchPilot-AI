"""Shared pytest fixtures and helpers.

Everything here is offline:
* ``make_pdf_bytes`` builds real (tiny) PDFs with PyMuPDF,
* ``FakeEmbedder`` is a deterministic bag-of-words embedder so retrieval
  tests never download a model,
* ``ScriptedLLM`` replaces the Ollama client with canned agent replies.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

from researchpilot.pdf_ingest import chunk_document, extract_document
from researchpilot.retriever import Retriever


# ------------------------------------------------------------------ PDF factory
def _wrap(text: str, width: int = 90) -> str:
    """Wrap text so insert_text keeps it inside the page."""
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return "\n".join(lines)


def make_pdf_bytes(pages: list[str], title: str | None = None) -> bytes:
    """Build a small real PDF; each entry in ``pages`` becomes one page."""
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 96), _wrap(text), fontsize=11)
    if title:
        doc.set_metadata({"title": title})
    data = doc.tobytes()
    doc.close()
    return data


# ------------------------------------------------------------------ fake embedder
class FakeEmbedder:
    """Deterministic bag-of-words embedder for offline tests.

    Each word is hashed into one of ``dim`` buckets. No model download is
    needed, and the same text always produces the same unit vector.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype="float32")
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        return vec

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        vectors = np.stack([self._vector(t) for t in texts])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (vectors / norms).astype("float32")


# ------------------------------------------------------------------ scripted LLM
ANALYSIS_REPLY = """RESEARCH QUESTION: How attention mechanisms improve language modelling.
METHODOLOGY: A transformer encoder with self-attention was trained on the IMDB dataset.
FINDINGS: The attention model reaches 92 percent accuracy on the benchmark.
LIMITATIONS: The dataset is small and only English text is covered."""

COMPARISON_REPLY = """COMPARISON: The first paper applies attention to text, while the second uses convolutional networks on medical images. The datasets differ between the papers, and neither paper reports results on a shared benchmark.
DATASETS:
paper_one.pdf: IMDB movie reviews
paper_two.pdf: ChestX-ray14
RESEARCH GAPS:
- No shared evaluation benchmark across the papers (tentative).
- Both papers evaluate on small datasets (tentative)."""

INTRO_REPLY = (
    "INTRODUCTION: These papers study machine learning approaches.\n"
    "CONCLUSION: More work is needed on shared benchmarks."
)


def _verification_reply(prompt: str) -> str:
    match = re.search(r'analysis of "([^"]+)"', prompt)
    filename = match.group(1) if match else "paper_one.pdf"
    return (
        "VERDICT: SUPPORTED\n"
        "REASON: Passage 1 states the model reaches 92 percent accuracy.\n"
        f"SOURCE: {filename}, p. 1"
    )


class ScriptedLLM:
    """Stand-in for OllamaClient that returns deterministic replies per agent.

    Routing works the same way the real prompts do: each agent's prompt asks
    for a distinctive header, so we can tell them apart.
    """

    def __init__(self):
        self.calls: list[str] = []

    def generate(self, prompt: str, system=None, num_predict: int = 512) -> str:
        self.calls.append(prompt)
        if "VERDICT:" in prompt:
            return _verification_reply(prompt)
        if "RESEARCH GAPS:" in prompt:
            return COMPARISON_REPLY
        if "RESEARCH QUESTION:" in prompt:
            return ANALYSIS_REPLY
        if "INTRODUCTION:" in prompt:
            return INTRO_REPLY
        raise AssertionError(f"Unexpected prompt:\n{prompt[:400]}")


# ------------------------------------------------------------------ workflow helper
def run_workflow(llm, question: str = "How do neural networks help with machine learning?"):
    """Run the full 3-agent LangGraph graph over two tiny synthetic papers."""
    from researchpilot.agents.graph import build_graph, initial_state

    papers = [
        (
            "paper_one.pdf",
            make_pdf_bytes(
                [
                    "Attention mechanisms study. This paper studies attention mechanisms "
                    "and self attention transformers for language processing.",
                    "We train a transformer encoder on the IMDB movie review dataset "
                    "and measure accuracy on the benchmark.",
                ],
                title="Attention for Small Language Models",
            ),
        ),
        (
            "paper_two.pdf",
            make_pdf_bytes(
                [
                    "Convolutional networks for chest x-ray classification with the "
                    "ChestXray14 dataset of medical images.",
                    "Our convolutional model detects pneumonia accurately. Limitations "
                    "include the small dataset size.",
                ],
                title="CNNs for Medical Imaging",
            ),
        ),
    ]

    chunks, filenames = [], []
    for name, data in papers:
        doc = extract_document(data, name)
        filenames.append(name)
        chunks.extend(chunk_document(doc))

    retriever = Retriever(FakeEmbedder())
    retriever.build(chunks)
    graph = build_graph()
    return graph.invoke(initial_state(question, filenames, retriever, llm))


# ------------------------------------------------------------------ fixtures
@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def scripted_llm():
    return ScriptedLLM()
