"""Agent 2 — Comparison Agent.

Reads the Paper Analysis Agent's structured summaries from the graph state
(explicit data flow between agents), retrieves additional cross-paper
evidence, and produces:

* a comparison narrative (methods, datasets, findings, limitations),
* a per-paper dataset mapping,
* tentative research gaps.

Missing information is reported as unknown, never invented.
"""
from __future__ import annotations

import re

from researchpilot.agents.state import (
    AgentState,
    ComparisonResult,
    merge_diagnostics,
)
from researchpilot.config import MAX_RESEARCH_GAPS
from researchpilot.llm import LLMError
from researchpilot.retriever import dedupe

_COMPARISON_QUERY = (
    "comparison methods methodology dataset evaluation results findings limitations"
)


def _papers_block(analyses) -> str:
    """Render agent 1's structured output — this is the agent-to-agent input."""
    blocks = []
    for filename, analysis in analyses.items():
        blocks.append(
            f"Paper: {filename} (title: {analysis.title})\n"
            f"- Research question: {analysis.research_question}\n"
            f"- Methodology: {analysis.methodology}\n"
            f"- Findings: {analysis.findings}\n"
            f"- Limitations: {analysis.limitations}"
        )
    return "\n\n".join(blocks)


def _passages_block(passages) -> str:
    return "\n\n".join(f"[source: {p.source}]\n{p.text}" for p in passages)


def parse_comparison(raw: str) -> tuple[str, dict[str, str], list[str]]:
    """Split the reply into narrative, per-paper datasets and research gaps.

    Returns (narrative, datasets, gaps). Anything that cannot be parsed is
    simply missing — never fabricated.
    """
    narrative, datasets, gaps = raw, {}, []

    # 1) Cut the RESEARCH GAPS section off the tail.
    parts = re.split(
        r"^\s*RESEARCH GAPS\s*:\s*",
        raw,
        maxsplit=1,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if len(parts) == 2:
        narrative, gap_text = parts
        gaps = [
            re.sub(r"^[\s\-*\d.)]+", "", line).strip()
            for line in gap_text.splitlines()
            if line.strip() and line.strip().upper() != "UNKNOWN"
        ][:MAX_RESEARCH_GAPS]

    # 2) Cut the DATASETS section out of what remains.
    parts = re.split(
        r"^\s*DATASETS\s*:\s*",
        narrative,
        maxsplit=1,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if len(parts) == 2:
        narrative, dataset_text = parts
        for line in dataset_text.splitlines():
            match = re.match(r"^\s*[-*]?\s*([^:]{1,120}?)\s*:\s*(.+)$", line)
            if match:
                datasets[match.group(1).strip()] = match.group(2).strip()

    # 3) Drop the leading COMPARISON header, if present.
    narrative = re.sub(r"^\s*COMPARISON\s*:\s*", "", narrative, flags=re.IGNORECASE).strip()
    return narrative, datasets, gaps


def comparison_node(state: AgentState) -> dict:
    """LangGraph node: compare the papers using agent 1's analyses."""
    analyses = state.get("paper_analyses", {})
    retriever = state["retriever"]
    llm = state["llm"]
    question = state["question"]
    errors: list[str] = []

    if len(analyses) < 2:
        result = ComparisonResult(
            narrative=(
                "Only one document was analysed, so a cross-paper comparison is not "
                "possible. Upload at least two papers to enable the Comparison Agent."
            )
        )
        return {"comparison": result, **merge_diagnostics(state, errors, [])}

    # Cross-paper evidence: top passages from every paper for a comparison query.
    passages = []
    for filename in analyses:
        passages.extend(
            retriever.search_for_paper(
                filename, f"{question} {_COMPARISON_QUERY}", k=3
            )
        )
    passages = dedupe(passages)[:10]

    prompt = (
        f'The student\'s research question is: "{question}"\n\n'
        "Structured analyses produced by the Paper Analysis Agent:\n\n"
        f"{_papers_block(analyses)}\n\n"
        "Passages retrieved from the uploaded papers:\n\n"
        f"{_passages_block(passages)}\n\n"
        "Compare these papers. Reply with EXACTLY these three section headers:\n\n"
        "COMPARISON:\n<3-6 sentences comparing the papers' methods, datasets, "
        "findings and limitations. Where a paper's information is unknown, say so "
        "instead of guessing.>\n\n"
        "DATASETS:\n<one line per paper in the form 'filename: dataset(s) used' — "
        "or 'filename: UNKNOWN' if the passages do not say>\n\n"
        "RESEARCH GAPS:\n<2-4 lines each starting with '- ', describing gaps that "
        "follow from the stated limitations or differences. End each line with "
        "(tentative).>"
    )

    try:
        raw = llm.generate(prompt, num_predict=520)
    except LLMError as exc:
        errors.append(f"Comparison Agent failed: {exc}")
        result = ComparisonResult(
            narrative="Comparison unavailable: the local LLM could not be reached."
        )
        return {"comparison": result, **merge_diagnostics(state, errors, [])}

    narrative, datasets, gaps = parse_comparison(raw)
    result = ComparisonResult(
        narrative=narrative, datasets=datasets, gaps=gaps, evidence=passages
    )
    return {"comparison": result, **merge_diagnostics(state, errors, [])}
