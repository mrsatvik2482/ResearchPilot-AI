"""Assemble the final literature review from the agents' outputs.

The section structure is deterministic. The introduction and conclusion
paragraphs are written by the local LLM from the analyses — with a plain
template fallback when the LLM is unavailable. The References section lists
ONLY sources that were actually retrieved as evidence: real uploaded
filenames with real page numbers.
"""
from __future__ import annotations

import re

from researchpilot.agents.state import (
    UNKNOWN_VALUES,
    AgentState,
    ComparisonResult,
    PaperAnalysis,
)
from researchpilot.retriever import RetrievedPassage


# ------------------------------------------------------------------ prompts
def _intro_conclusion_prompt(question: str, analyses: dict[str, PaperAnalysis]) -> str:
    notes = "\n".join(
        f"- {a.filename}: findings: {a.findings}; limitations: {a.limitations}"
        for a in analyses.values()
    )
    return (
        f'Research question: "{question}"\n\n'
        "Notes from the paper analyses (nothing else is known — do not invent "
        f"anything):\n{notes}\n\n"
        "Write the INTRODUCTION and CONCLUSION of a short literature review of "
        "these papers. Each section must be 3-4 sentences. Do not invent "
        "citations or statistics.\n"
        "Reply with EXACTLY these two headers:\n"
        "INTRODUCTION: <text>\n"
        "CONCLUSION: <text>"
    )


def _parse_intro_conclusion(raw: str) -> tuple[str, str]:
    intro_match = re.search(
        r"INTRODUCTION\s*[:\-]\s*(.*?)(?=CONCLUSION\s*[:\-]|$)",
        raw,
        re.IGNORECASE | re.DOTALL,
    )
    conclusion_match = re.search(
        r"CONCLUSION\s*[:\-]\s*(.*)$", raw, re.IGNORECASE | re.DOTALL
    )
    intro = intro_match.group(1).strip() if intro_match else ""
    conclusion = conclusion_match.group(1).strip() if conclusion_match else ""
    return intro, conclusion


def _fallback_intro(question: str, n_papers: int) -> str:
    return (
        f"This review examines {n_papers} uploaded paper(s) in relation to the "
        f"research question '{question}'. All analysis was produced by a local "
        "pipeline: passages were retrieved with semantic search over the uploaded "
        "PDFs and summarised by local AI agents. Every statement in this review "
        "is intended to be traceable to a specific PDF page."
    )


def _fallback_conclusion(question: str, n_papers: int) -> str:
    return (
        f"Across the {n_papers} paper(s) analysed, the evidence retrieved from "
        f"the uploaded PDFs offers a partial answer to '{question}'. Where the "
        "papers did not state information, this review marks it as unknown "
        "rather than guessing. Uploading more papers on the topic would broaden "
        "the evidence base."
    )


# ------------------------------------------------------------------ helpers
def _references(
    analyses: dict[str, PaperAnalysis], comparison: ComparisonResult | None
) -> list[str]:
    """Unique evidence sources actually used, sorted by filename then page."""
    sources: set[tuple[str, int, str]] = set()
    for analysis in analyses.values():
        for passage in analysis.evidence:
            sources.add((passage.filename, passage.page, passage.title))
    if comparison:
        for passage in comparison.evidence:
            sources.add((passage.filename, passage.page, passage.title))
    return [
        f'{filename} — "{title}", p. {page}.'
        for filename, page, title in sorted(sources)
    ]


def _sufficiency_note(analyses: dict[str, PaperAnalysis]) -> str | None:
    """Warn when the retrieved evidence was too thin to fill most fields."""
    fields = ["research_question", "methodology", "findings", "limitations"]
    total = missing = 0
    for analysis in analyses.values():
        for field_name in fields:
            total += 1
            if getattr(analysis, field_name).upper() in UNKNOWN_VALUES:
                missing += 1
    if total and missing / total >= 0.5:
        return (
            "**Note on evidence sufficiency:** the retrieved passages did not "
            "contain enough information to fill most analysis fields. Treat the "
            "generated content with caution and consider uploading papers with "
            "full text."
        )
    return None


# ------------------------------------------------------------------ builder
def build_review(state: AgentState, llm=None) -> str:
    """Build the full literature review as Markdown.

    ``llm`` is optional: when given, it writes the introduction/conclusion;
    otherwise template text is used.
    """
    question = state["question"]
    analyses: dict[str, PaperAnalysis] = state.get("paper_analyses", {})
    comparison: ComparisonResult | None = state.get("comparison")
    verification = state.get("verification", [])
    warnings: list[str] = state.get("warnings", []) or []

    intro, conclusion = "", ""
    if llm is not None and analyses:
        try:
            raw = llm.generate(_intro_conclusion_prompt(question, analyses), num_predict=420)
            intro, conclusion = _parse_intro_conclusion(raw)
        except Exception:
            intro, conclusion = "", ""
    if not intro:
        intro = _fallback_intro(question, len(analyses))
    if not conclusion:
        conclusion = _fallback_conclusion(question, len(analyses))

    parts: list[str] = []
    parts.append(f"# Literature Review — {question}")
    parts.append("_Generated locally by ResearchPilot AI from the uploaded PDFs only._")

    # 1. Introduction
    parts.append("## 1. Introduction\n\n" + intro)

    # 2. Paper summaries
    parts.append("## 2. Paper Summaries")
    note = _sufficiency_note(analyses)
    if note:
        parts.append(note)
    for analysis in analyses.values():
        parts.append(f"### {analysis.title}\n_({analysis.filename})_")
        parts.append(f"- **Research question:** {analysis.research_question}")
        parts.append(f"- **Methodology:** {analysis.methodology}")
        parts.append(f"- **Findings:** {analysis.findings}")
        parts.append(f"- **Limitations:** {analysis.limitations}")
        if analysis.evidence:
            sources = "; ".join(p.source for p in analysis.evidence[:6])
            parts.append(f"- **Evidence pages:** {sources}")

    # 3. Comparison
    parts.append("## 3. Cross-Paper Comparison")
    if comparison and comparison.narrative:
        parts.append(comparison.narrative)
        if comparison.datasets:
            parts.append("**Datasets used**\n")
            for filename, dataset in comparison.datasets.items():
                parts.append(f"- {filename}: {dataset}")
    else:
        parts.append("A cross-paper comparison was not available for this run.")

    # 4. Research gaps
    parts.append("## 4. Research Gaps (tentative)")
    if comparison and comparison.gaps:
        parts.append(
            "_These gaps are **inferred** from the analysed papers and are "
            "tentative; confirm them by reading the full papers._"
        )
        for gap in comparison.gaps:
            parts.append(f"- {gap}")
    else:
        parts.append(
            "No research gaps could be identified from the retrieved evidence."
        )

    # 5. Citation verification
    parts.append("## 5. Citation Verification")
    if verification:
        parts.append("| # | Claim | Verdict | Reason | Source |")
        parts.append("|---|-------|---------|--------|--------|")
        for i, check in enumerate(verification, start=1):
            claim = check.claim.replace("|", "\\|")[:200]
            reason = check.reason.replace("|", "\\|")
            source = check.source.replace("|", "\\|")
            parts.append(
                f"| {i} | {claim} | {check.verdict} | {reason} | {source} |"
            )
    else:
        parts.append(
            "No claims were verified in this run (no LLM available, or nothing "
            "to verify)."
        )

    # 6. Conclusion
    parts.append("## 6. Conclusion\n\n" + conclusion)

    # 7. References — only real retrieved sources
    parts.append("## 7. References")
    parts.append("_Only uploaded files and pages that were actually retrieved as "
                "evidence are listed. Nothing is invented._")
    references = _references(analyses, comparison)
    if references:
        for i, reference in enumerate(references, start=1):
            parts.append(f"[{i}] {reference}")
    else:
        parts.append("No evidence passages were retrieved in this run.")

    if warnings:
        parts.append("---\n**Processing warnings**")
        for warning in warnings:
            parts.append(f"- {warning}")

    return "\n\n".join(parts)
