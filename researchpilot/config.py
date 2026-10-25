"""Central configuration for ResearchPilot AI.

Everything runs locally: embeddings come from a local Sentence Transformers
model and text generation from a local Ollama model.
"""

# --- Embeddings -----------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # ~90 MB, 384-dim

# --- Chunking -------------------------------------------------------------
CHUNK_SIZE = 900       # characters per chunk (~200 tokens)
CHUNK_OVERLAP = 150    # characters shared between consecutive chunks

# --- Retrieval ------------------------------------------------------------
TOP_K = 6              # passages retrieved for a global query
PER_PAPER_K = 6        # passages retrieved per paper for analysis
WEAK_MATCH_SCORE = 0.30  # cosine similarity below this is flagged as a weak match

# --- Ollama (local LLM) ---------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"  # ~2 GB download, runs well on 16 GB RAM
LLM_TIMEOUT_SECONDS = 240
LLM_OPTIONS = {"temperature": 0.2, "num_ctx": 4096}

# --- Agents ---------------------------------------------------------------
MAX_VERIFIED_CLAIMS = 12   # claims the Citation Verification Agent checks per run
MAX_RESEARCH_GAPS = 5
