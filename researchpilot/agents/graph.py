"""The LangGraph workflow that wires the three agents together.

    START
      │
      ▼
    Paper Analysis Agent ──► Comparison Agent ──► Citation Verification Agent ──► END

Each agent reads and writes the shared AgentState (see state.py), which is
the explicit data flow of the application:

* the Comparison Agent consumes ``paper_analyses`` from the Paper Analysis
  Agent,
* the Citation Verification Agent checks the claims inside those analyses
  against freshly retrieved passages.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from researchpilot.agents.analysis import paper_analysis_node
from researchpilot.agents.citation import citation_verification_node
from researchpilot.agents.comparison import comparison_node
from researchpilot.agents.state import AgentState

PAPER_ANALYSIS_NODE = "paper_analysis_agent"
COMPARISON_NODE = "comparison_agent"
CITATION_VERIFICATION_NODE = "citation_verification_agent"

AGENT_LABELS = {
    PAPER_ANALYSIS_NODE: "Paper Analysis Agent",
    COMPARISON_NODE: "Comparison Agent",
    CITATION_VERIFICATION_NODE: "Citation Verification Agent",
}


def build_graph():
    """Compile the three-agent workflow."""
    builder = StateGraph(AgentState)
    builder.add_node(PAPER_ANALYSIS_NODE, paper_analysis_node)
    builder.add_node(COMPARISON_NODE, comparison_node)
    builder.add_node(CITATION_VERIFICATION_NODE, citation_verification_node)

    builder.add_edge(START, PAPER_ANALYSIS_NODE)
    builder.add_edge(PAPER_ANALYSIS_NODE, COMPARISON_NODE)
    builder.add_edge(COMPARISON_NODE, CITATION_VERIFICATION_NODE)
    builder.add_edge(CITATION_VERIFICATION_NODE, END)
    return builder.compile()


def initial_state(question, filenames, retriever, llm) -> AgentState:
    """Build the input state for one pipeline run."""
    return {
        "question": question,
        "filenames": list(filenames),
        "retriever": retriever,
        "llm": llm,
        "errors": [],
        "warnings": [],
    }


def run_pipeline(question, filenames, retriever, llm) -> AgentState:
    """Run the whole workflow and return the final state (used by tests/scripts)."""
    graph = build_graph()
    return graph.invoke(initial_state(question, filenames, retriever, llm))
