"""Agent 1 — Paper Analysis Agent.

For every uploaded paper it retrieves passages relevant to the user's
research question and asks the local LLM for a structured summary:
research question, methodology, findings, limitations. Any field the
retrieved passages do not cover is marked UNKNOWN instead of guessed.
"""
from __future__ import annotations

import re

from researchpilot.agents.state import (
    ANALYSIS_SECTIONS,
    AgentState,
    PaperAnalysis,
    merge_diagnostics,
)
from researchpilot.config import PER_PAPER_K
from researchpilot.llm import LLMError

# Retrieval query = user's question + words that steer towards the
# sections we need evidence for.
_RETRIEVAL_QUERY_SUFFIX = (
    "abstract introduction motivation methodology approach dataset "
    "experiments results findings limitations conclusion"
)

# Header variants the parser accepts (models sometimes shorten headers).
_HEADER_ALIASES = {
    "RESEARCH QUESTION": "RESEARCH QUESTION",
    "RESEARCHQUESTION": "RESEARCH QUESTION",
    "PROBLEM": "RESEARCH QUESTION",
    "METHOD": "METHODOLOGY",
    "METHODS": "METHODOLOGY",
    "METHODOLOGY": "METHODOLOGY",
    "APPROACH": "METHODOLOGY",
    "FINDING": "FINDINGS",
    "FINDINGS": "FINDINGS",
    "RESULT": "FINDINGS",
    "RESULTS": "FINDINGS",
    "LIMITATION": "LIMITATIONS",
    "LIMITATIONS": "LIMITATIONS",
}

# Matches '## **RESEARCH QUESTION:** some text' and friends.
_HEADER_LINE = re.compile(
    r"^\s*(?:#+|\*+|-+)*\s*(?P<header>[A-Za-z ]+?)\s*(?:\*+|#+)*\s*[:\-–]\s*(?:\*+|#+)*\s*(?P<rest>.*)$"
)


def parse_sections(raw: str) -> dict[str, str]:
    """Parse 'HEADER: text' sections out of an LLM reply.

    Tolerates markdown decoration (##, **) and minor header variants such
    as 'METHOD:' instead of 'METHODOLOGY:'. Sections the reply does not
    contain stay UNKNOWN — never invented.
    """
    parsed: dict[str, str] = {}
    current: str | None = None
    for line in raw.splitlines():
        match = _HEADER_LINE.match(line)
        if match:
            header = _HEADER_ALIASES.get(match.group("header").strip().upper())
            if header:
                current = header
                parsed[header] = match.group("rest").strip()
                continue
        if current and line.strip():
            parsed[current] = (parsed[current] + " " + line.strip()).strip()
    return {
        section: (parsed.get(section, "") or "UNKNOWN").strip()
        for section in ANALYSIS_SECTIONS
    }


def _analysis_prompt(question: str, filename: str, title: str, passages) -> str:
    context = "\n\n".join(
        f"[Passage {i} | source: {passage.source}]\n{passage.text}"
        for i, passage in enumerate(passages, start=1)
    )
    return (
        f'The student\'s research question is: "{question}"\n\n'
        f'Below are passages extracted from the paper "{filename}" '
        f'(detected title: "{title}").\n\n'
        f"{context}\n\n"
        "Using ONLY these passages, analyse this paper.\n"
        "Reply with EXACTLY these four section headers, each on its own line, "
        "in this order:\n\n"
        "RESEARCH QUESTION: <the problem the paper addresses — 1-2 sentences>\n"
        "METHODOLOGY: <approach, methods, models and dataset used — 2-3 sentences>\n"
        "FINDINGS: <the main results — 2-3 sentences>\n"
        "LIMITATIONS: <weaknesses or limitations mentioned — 1-2 sentences>\n\n"
        "If the passages do not cover a section, write UNKNOWN after that header. "
        "Do not invent anything."
    )


def paper_analysis_node(state: AgentState) -> dict:
    """LangGraph node: summarise every uploaded paper from retrieved evidence."""
    retriever = state["retriever"]
    llm = state["llm"]
    question = state["question"]

    analyses: dict[str, PaperAnalysis] = {}
    errors: list[str] = []
    warnings: list[str] = []

    for filename in state["filenames"]:
        passages = retriever.search_for_paper(
            filename, query=f"{question} {_RETRIEVAL_QUERY_SUFFIX}", k=PER_PAPER_K
        )
        title = passages[0].title if passages else filename

        if not passages:
            analyses[filename] = PaperAnalysis(
                filename=filename,
                title=title,
                research_question="UNKNOWN",
                methodology="UNKNOWN",
                findings="UNKNOWN",
                limitations="UNKNOWN",
            )
            warnings.append(f"No passages could be retrieved for '{filename}'.")
            continue

        try:
            raw = llm.generate(
                _analysis_prompt(question, filename, title, passages), num_predict=420
            )
        except LLMError as exc:
            errors.append(f"Paper Analysis Agent failed for '{filename}': {exc}")
            analyses[filename] = PaperAnalysis(
                filename=filename,
                title=title,
                research_question="ANALYSIS UNAVAILABLE",
                methodology="ANALYSIS UNAVAILABLE",
                findings="ANALYSIS UNAVAILABLE",
                limitations="ANALYSIS UNAVAILABLE",
                evidence=passages,
            )
            continue

        sections = parse_sections(raw)
        analyses[filename] = PaperAnalysis(
            filename=filename,
            title=title,
            research_question=sections["RESEARCH QUESTION"],
            methodology=sections["METHODOLOGY"],
            findings=sections["FINDINGS"],
            limitations=sections["LIMITATIONS"],
            evidence=passages,
        )

    return {"paper_analyses": analyses, **merge_diagnostics(state, errors, warnings)}
