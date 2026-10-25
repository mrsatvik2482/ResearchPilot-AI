"""Minimal client for Ollama's local REST API.

No API keys, no cloud calls — Ollama must be running on this machine
(default: http://localhost:11434). Tests replace this client with a
scripted stand-in, so the rest of the pipeline never talks to the network
during testing.
"""
from __future__ import annotations

import requests

from researchpilot.config import (
    DEFAULT_OLLAMA_MODEL,
    LLM_OPTIONS,
    LLM_TIMEOUT_SECONDS,
    OLLAMA_BASE_URL,
)


class LLMError(Exception):
    """Raised when the local LLM cannot be reached or returns an error."""


SYSTEM_PROMPT = (
    "You are ResearchPilot AI, a rigorous academic research assistant.\n"
    "You must follow these rules:\n"
    "1. Answer ONLY from the provided passages. Never invent papers, authors, "
    "numbers, statistics, quotations, or page numbers.\n"
    "2. If the passages do not contain the information, reply exactly: UNKNOWN.\n"
    "3. Be concise and factual. Do not add preamble."
)


class OllamaClient:
    """Talks to a local Ollama server over HTTP."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = DEFAULT_OLLAMA_MODEL,
        timeout: int = LLM_TIMEOUT_SECONDS,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    # -- health checks --------------------------------------------------
    def list_models(self) -> list[str]:
        """Return installed model names, or [] if the server is unreachable."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            payload = response.json()
            return [m.get("name", "") for m in payload.get("models", [])]
        except requests.RequestException:
            return []

    def health(self) -> tuple[bool, str]:
        """Check that the server is up and the selected model is installed.

        Returns (ok, message); the message tells the user exactly how to fix
        the problem when something is missing.
        """
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=3)
        except requests.RequestException:
            return False, (
                "Ollama is not running at "
                f"{self.base_url}. Start it with 'ollama serve' (or launch the "
                "Ollama desktop app), then try again."
            )
        if response.status_code != 200:
            return False, f"Ollama responded with HTTP {response.status_code}."

        models = self.list_models()
        if not models:
            return False, (
                "Ollama is running but no models are installed. "
                f"Run this in a terminal:  ollama pull {self.model}"
            )
        if self.model not in models:
            return False, (
                f"Model '{self.model}' is not installed (found: {', '.join(models)}). "
                f"Run:  ollama pull {self.model}  — or pick an installed model "
                "in the sidebar."
            )
        return True, f"Ollama is ready — using model '{self.model}'."

    # -- generation ------------------------------------------------------
    def generate(self, prompt: str, system: str = SYSTEM_PROMPT, num_predict: int = 512) -> str:
        """Run one non-streaming completion on the local model."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {**LLM_OPTIONS, "num_predict": num_predict},
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/generate", json=payload, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise LLMError(
                f"Could not reach Ollama at {self.base_url}: {exc}. Is 'ollama serve' running?"
            ) from exc
        if response.status_code != 200:
            raise LLMError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:300]}"
            )
        data = response.json()
        if "error" in data:
            raise LLMError(f"Ollama error: {data['error']}")
        return (data.get("response") or "").strip()
