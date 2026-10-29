"""Multi-agent troubleshooting on top of the service manual RAG pipeline.

A LangGraph graph with three agents:
1. Supervisor: classifies intent, extracts equipment and symptoms, routes each turn.
2. Diagnostic agent: searches the manuals, analyzes media and ranks hypotheses.
3. Action agent: records sessions and outcomes in the knowledge graph.

``create_agent_workflow`` is imported on first access, so the state types can be used
without the ``agents`` extra (langgraph) installed.
"""

from typing import Any

from service_manual_rag.agents.state import (
    ActionResult,
    AgentState,
    DiagnosticResult,
    DiagnosticStatus,
    Evidence,
    Hypothesis,
)

__all__ = [
    "ActionResult",
    "AgentState",
    "DiagnosticResult",
    "DiagnosticStatus",
    "Evidence",
    "Hypothesis",
    "create_agent_workflow",
]


def __getattr__(name: str) -> Any:
    if name == "create_agent_workflow":
        from service_manual_rag.agents.workflow import create_agent_workflow

        return create_agent_workflow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
