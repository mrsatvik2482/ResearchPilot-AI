"""Agent 3 — Citation Verification Agent.

Takes the claims produced by the Paper Analysis Agent, retrieves the
passages most similar to each claim, and asks the LLM whether the passage
text actually supports the claim. Unsupported claims are flagged rather
than silently kept in the review — this is the anti-hallucination layer.
"""
from __future__ import annotations

import re

from researchpilot.agents.state import (
    UNKNOWN_VALUES,
    AgentState,
    CitationCheck,
    merge_diagnostics,
)
from researchpilot.config import MAX_VERIFIED_CLAIMS
from researchpilot.llm import LLMError

VERDICTS = {"SUPPORTED", "UNSUPPORTED", "INSUFFICIENT_EVIDENCE"}

_VERDICT_LINE = re.compile(
    r"VERDICT\s*[:\-]\s*(SUPPORTED|UNSUPPORTED|INSUFFICIENT[ _]EVIDENCE)",
    re.IGNORECASE,
)
_REASON_LINE = re.compile(r"REASON\s*[:\-]\s*(.+)", re.IGNORECASE)
_SOURCE_LINE = re.compile(r"SOURCE\s*[:\-]\s*(.+)", re.IGNORECASE)


def _claims_from_analyses(analyses) -> list[tuple[str, str]]:
    """Collect (filename, claim) pairs to verify — methodology + findings per paper.

    UNKNOWN / unavailable fields are skipped: there is nothing to verify.
    """
    claims: list[tuple[str, str]] = []
    for filename, analysis in analyses.items():
        for field_name in ("methodology", "findings"):
            text = getattr(analysis, field_name).strip()
            if text and text.upper() not in UNKNOWN_VALUES:
                claims.append((filename, text))
    return claims


def _verification_prompt(filename: str, claim: str, passages) -> str:
    context = "\n\n".join(f"[source: {p.source}]\n{p.text}" for p in passages)
    return (
        f'Claim to verify (taken from the generated analysis of "{filename}"):\n'
        f'"{claim}"\n\n'
        f"Passages retrieved from the uploaded papers:\n\n{context}\n\n"
        "Question: is the claim directly supported by these passages?\n"
        "Reply with EXACTLY these three lines:\n"
        "VERDICT: SUPPORTED or UNSUPPORTED or INSUFFICIENT_EVIDENCE\n"
        "REASON: <one sentence>\n"
        f'SOURCE: <the supporting passage source, e.g. "{filename}, p. 2", or NONE>'
    )


def parse_verdict(raw: str, default_source: str) -> tuple[str, str, str]:
    """Extract (verdict, reason, source) from a verification reply.

    If the reply cannot be parsed we default to INSUFFICIENT_EVIDENCE — we
    never assume a claim is supported.
    """
    verdict_match = _VERDICT_LINE.search(raw)
    verdict = (
        verdict_match.group(1).upper().replace(" ", "_")
        if verdict_match
        else "INSUFFICIENT_EVIDENCE"
    )
    if verdict not in VERDICTS:
        verdict = "INSUFFICIENT_EVIDENCE"

    reason_match = _REASON_LINE.search(raw)
    reason = (
        reason_match.group(1).strip()
        if reason_match
        else "The verification reply could not be parsed."
    )
    source_match = _SOURCE_LINE.search(raw)
    source = source_match.group(1).strip() if source_match else default_source
    return verdict, reason, source


def citation_verification_node(state: AgentState) -> dict:
    """LangGraph node: check every generated claim against retrieved passages."""
    analyses = state.get("paper_analyses", {})
    retriever = state["retriever"]
    llm = state["llm"]
    errors: list[str] = []

    claims = _claims_from_analyses(analyses)[:MAX_VERIFIED_CLAIMS]
    if not claims:
        return {"verification": [], **merge_diagnostics(state, errors, [])}

    checks: list[CitationCheck] = []
    for filename, claim in claims:
        # Retrieve passages that are most similar to the claim itself.
        passages = retriever.search_for_paper(filename, query=claim, k=4)
        if not passages:
            checks.append(
                CitationCheck(
                    claim=claim,
                    verdict="INSUFFICIENT_EVIDENCE",
                    reason="No passages could be retrieved to verify this claim.",
                    source="NONE",
                )
            )
            continue

        prompt = _verification_prompt(filename, claim[:600], passages)
        try:
            raw = llm.generate(prompt, num_predict=120)
        except LLMError as exc:
            errors.append(
                f"Citation Verification Agent failed for a claim in '{filename}': {exc}"
            )
            checks.append(
                CitationCheck(
                    claim=claim,
                    verdict="INSUFFICIENT_EVIDENCE",
                    reason=f"Verification failed: {exc}",
                    source="NONE",
                )
            )
            continue

        verdict, reason, source = parse_verdict(raw, default_source=passages[0].source)
        checks.append(
            CitationCheck(claim=claim, verdict=verdict, reason=reason, source=source)
        )

    return {"verification": checks, **merge_diagnostics(state, errors, [])}
