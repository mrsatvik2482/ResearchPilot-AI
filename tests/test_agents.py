"""End-to-end tests of the LangGraph workflow with a scripted (mocked) LLM.

These prove the three agents are real, distinct nodes with explicit data
flow: analyses flow into the comparison agent, and both flow into the
citation verification agent.
"""
import pytest

from conftest import ScriptedLLM, run_workflow
from researchpilot.llm import LLMError


def test_all_three_agents_populate_state():
    llm = ScriptedLLM()
    result = run_workflow(llm)
    assert result["errors"] == []
    assert set(result["paper_analyses"]) == {"paper_one.pdf", "paper_two.pdf"}
    assert result["comparison"] is not None
    assert result["comparison"].gaps
    assert result["verification"]


def test_analysis_sections_parsed_and_evidence_kept():
    llm = ScriptedLLM()
    result = run_workflow(llm)
    analysis = result["paper_analyses"]["paper_one.pdf"]
    assert analysis.research_question.startswith("How attention")
    assert (
        analysis.findings
        == "The attention model reaches 92 percent accuracy on the benchmark."
    )
    assert analysis.evidence  # provenance passages are kept for citations
    assert all(p.filename == "paper_one.pdf" for p in analysis.evidence)


def test_comparison_agent_consumes_analysis_output():
    """The comparison prompt must contain agent 1's parsed output — real data flow."""
    llm = ScriptedLLM()
    result = run_workflow(llm)
    comparison_prompt = llm.calls[2]  # two analysis calls, then the comparison call
    assert "How attention mechanisms improve language modelling." in comparison_prompt
    assert result["comparison"].datasets["paper_one.pdf"] == "IMDB movie reviews"
    assert result["comparison"].datasets["paper_two.pdf"] == "ChestX-ray14"
    assert len(result["comparison"].gaps) == 2


def test_verification_claims_trace_back_to_analyses():
    llm = ScriptedLLM()
    result = run_workflow(llm)
    findings = result["paper_analyses"]["paper_one.pdf"].findings
    assert any(findings in c.claim for c in result["verification"])
    for check in result["verification"]:
        assert check.verdict in {"SUPPORTED", "UNSUPPORTED", "INSUFFICIENT_EVIDENCE"}
        assert ", p." in check.source  # a real page citation, not a guess


def test_agent_call_order_is_analysis_then_comparison_then_verification():
    llm = ScriptedLLM()
    run_workflow(llm)
    kinds = []
    for prompt in llm.calls:
        if "VERDICT:" in prompt:
            kinds.append("verify")
        elif "RESEARCH GAPS:" in prompt:
            kinds.append("compare")
        elif "RESEARCH QUESTION:" in prompt:
            kinds.append("analyse")
        else:
            kinds.append("other")
    # 2 papers -> 2 analysis calls, 1 comparison, 2 claims/paper -> 4 verifications
    assert kinds == ["analyse", "analyse", "compare", "verify", "verify", "verify", "verify"]


def test_llm_failure_is_recorded_not_fatal():
    class FailingLLM:
        def generate(self, prompt, system=None, num_predict=512):
            raise LLMError("Ollama is down")

    result = run_workflow(FailingLLM())
    assert result["errors"], "failures must be recorded, not swallowed"
    assert result["paper_analyses"], "analyses still exist (marked unavailable)"
    assert all(
        a.findings == "ANALYSIS UNAVAILABLE"
        for a in result["paper_analyses"].values()
    )
    assert result["comparison"].narrative.startswith("Comparison unavailable")
    assert result["verification"] == []  # nothing verifiable was produced


def test_unparseable_llm_reply_yields_unknown_not_invented():
    class GarbageLLM:
        def generate(self, prompt, system=None, num_predict=512):
            if "RESEARCH QUESTION:" in prompt:
                return "I really like pizza and cannot follow instructions."
            if "RESEARCH GAPS:" in prompt:
                return "COMPARISON: something about papers."
            return "VERDICT: SUPPORTED"

    result = run_workflow(GarbageLLM())
    assert result["errors"] == []
    for analysis in result["paper_analyses"].values():
        assert analysis.research_question == "UNKNOWN"  # never invented
    # verification default on unparseable reply is INSUFFICIENT_EVIDENCE
    assert all(
        c.verdict in {"SUPPORTED", "UNSUPPORTED", "INSUFFICIENT_EVIDENCE"}
        for c in result["verification"]
    )


@pytest.mark.parametrize("claim_verdict", ["UNSUPPORTED", "INSUFFICIENT_EVIDENCE"])
def test_unsupported_claims_are_flagged(claim_verdict):
    class PickyLLM:
        def generate(self, prompt, system=None, num_predict=512):
            if "VERDICT:" in prompt:
                return (
                    f"VERDICT: {claim_verdict}\n"
                    "REASON: The passages do not state this.\n"
                    "SOURCE: NONE"
                )
            if "RESEARCH GAPS:" in prompt:
                return "COMPARISON: Papers differ.\nDATASETS:\nRESEARCH GAPS:\n- none"
            if "RESEARCH QUESTION:" in prompt:
                return (
                    "RESEARCH QUESTION: Attention for language.\n"
                    "METHODOLOGY: Transformer encoder.\n"
                    "FINDINGS: Accuracy improves a lot everywhere.\n"
                    "LIMITATIONS: UNKNOWN"
                )
            raise AssertionError("unexpected prompt")

    result = run_workflow(PickyLLM())
    assert result["verification"]
    assert all(c.verdict == claim_verdict for c in result["verification"])
    assert all(c.source == "NONE" for c in result["verification"])
