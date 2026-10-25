"""The three LangGraph agents and the workflow that connects them.

Agents:
* analysis  — per-paper structured summaries (research question, methodology,
  findings, limitations) grounded in retrieved passages.
* comparison — cross-paper comparison, datasets and tentative research gaps,
  built on top of the analyses.
* citation  — verifies generated claims against retrieved passages and flags
  unsupported ones.
"""
from researchpilot.agents.graph import (  # noqa: F401
    AGENT_LABELS,
    build_graph,
    initial_state,
    run_pipeline,
)
