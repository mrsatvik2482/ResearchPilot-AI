"""ResearchPilot AI — a local-first multi-agent research assistant built on RAG.

Pipeline: PDF ingestion -> local embeddings -> FAISS retrieval ->
LangGraph multi-agent workflow -> cited literature review.

Everything runs on the user's machine: embeddings come from a local
Sentence Transformers model and text generation from a local Ollama model.
No API keys, no cloud calls.
"""

__version__ = "1.0.0"
