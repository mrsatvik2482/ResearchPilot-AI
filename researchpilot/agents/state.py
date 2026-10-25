"""Shared LangGraph state: the explicit data contract between the agents.

Each agent node reads what it needs from this state and writes its own
output back into it — that is how data flows through the graph:

    Paper Analysis Agent  -> paper_analyses -> Comparison Agent
    Paper Analysis Agent  -> paper_analyses -> Citation Verification Agent
    Comparison Agent      -> comparison     -> (report + UI)

LangGraph state channels are replace-on-write, so nodes that accumulate
diagnostics use ``merge_diagnostics`` to return the whole updated list.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from researchpilot.retriever import RetrievedPassage

# Sections the Paper Analysis Agent fills in for every paper.
ANALYSIS_SECTIONS = ["RESEARCH QUESTION", "METHODOLOGY", "FINDINGS", "LIMITATIONS"]

# Values that mean "the passages did not contain this information".
UNKNOWN_VALUES = {"UNKNOWN", "ANALYSIS UNAVAILABLE"}


@dataclass
class PaperAnalysis:
    """Structured summary of one paper, produced by the Paper Analysis Agent."""

    filename: str
    title: str
    research_question: str
    methodology: str
    findings: str
    limitations: str
    evidence: list[RetrievedPassage] = field(default_factory=list)  # passages used


@dataclass
class ComparisonResult:
    """Cross-paper comparison, produced by the Comparison Agent."""

    narrative: str = ""                                  # markdown comparison text
    datasets: dict[str, str] = field(default_factory=dict)  # filename -> dataset(s)
    gaps: list[str] = field(default_factory=list)        # tentative research gaps
    evidence: list[RetrievedPassage] = field(default_factory=list)


@dataclass
class CitationCheck:
    """Verdict of the Citation Verification Agent for one generated claim."""

    claim: str
    verdict: str   # SUPPORTED | UNSUPPORTED | INSUFFICIENT_EVIDENCE
    reason: str
    source: str    # e.g. 'paper1.pdf, p. 3', or 'NONE'


class AgentState(TypedDict, total=False):
    """LangGraph state shared by all three agents.

    ``retriever`` and ``llm`` are injected inputs (passed through unchanged);
    every other key is produced by an agent node.
    """

    # inputs
    question: str
    filenames: list[str]
    retriever: Any  # researchpilot.retriever.Retriever
    llm: Any        # researchpilot.llm.OllamaClient

    # agent outputs
    paper_analyses: dict[str, PaperAnalysis]  # filename -> analysis
    comparison: Any                           # ComparisonResult | None
    verification: list[CitationCheck]

    # diagnostics accumulated by all agents
    errors: list[str]
    warnings: list[str]


def merge_diagnostics(
    state: AgentState, new_errors: list[str], new_warnings: list[str]
) -> dict:
    """Return the accumulated error/warning lists for a node update."""
    return {
        "errors": list(state.get("errors", [])) + new_errors,
        "warnings": list(state.get("warnings", [])) + new_warnings,
    }
