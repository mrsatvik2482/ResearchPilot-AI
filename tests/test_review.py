"""Tests for literature-review assembly: structure and citation integrity."""
import re

from conftest import ScriptedLLM, run_workflow
from researchpilot.review import build_review

SECTIONS = [
    "## 1. Introduction",
    "## 2. Paper Summaries",
    "## 3. Cross-Paper Comparison",
    "## 4. Research Gaps (tentative)",
    "## 5. Citation Verification",
    "## 6. Conclusion",
    "## 7. References",
]


def test_review_contains_all_sections():
    result = run_workflow(ScriptedLLM())
    review = build_review(result, llm=None)
    for section in SECTIONS:
        assert section in review, f"missing section: {section}"


def test_references_are_only_real_files_and_pages():
    """References must be real uploaded filenames with real page numbers."""
    result = run_workflow(ScriptedLLM())
    review = build_review(result, llm=None)
    references_block = review.split("## 7. References")[1]
    reference_lines = [ln for ln in references_block.splitlines() if ln.startswith("[")]
    assert reference_lines, "references section must not be empty"
    for line in reference_lines:
        assert "paper_one.pdf" in line or "paper_two.pdf" in line
        assert re.search(r"p\. \d+\.", line), f"no page number in: {line}"


def test_research_gaps_are_labelled_tentative():
    result = run_workflow(ScriptedLLM())
    review = build_review(result, llm=None)
    assert "tentative" in review.lower()
    assert "confirm them by reading" in review


def test_llm_written_introduction_is_used_when_available():
    result = run_workflow(ScriptedLLM())
    review = build_review(result, llm=ScriptedLLM())
    assert "These papers study machine learning approaches." in review
    assert "More work is needed on shared benchmarks." in review


def test_fallback_introduction_without_llm():
    result = run_workflow(ScriptedLLM())
    review = build_review(result, llm=None)
    assert "This review examines" in review


def test_verification_table_lists_every_check():
    result = run_workflow(ScriptedLLM())
    review = build_review(result, llm=None)
    table_rows = [ln for ln in review.splitlines() if ln.startswith("| ")]
    # 1 header row + 4 claims (methodology + findings per paper); the |---|
    # separator row does not start with "| " and is not counted.
    assert len(table_rows) == 1 + 4
    assert all("SUPPORTED" in row for row in table_rows[1:])


def test_insufficient_evidence_is_stated_when_analysis_empty():
    empty_state = {
        "question": "What is known about X?",
        "paper_analyses": {},
        "comparison": None,
        "verification": [],
        "errors": [],
        "warnings": [],
    }
    review = build_review(empty_state, llm=None)
    assert "No evidence passages were retrieved" in review
    assert "not available" in review.lower()
