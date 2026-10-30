"""LangGraph workflow for the supervisor, diagnostic and action agent graph.

Flow per user turn:
    supervisor routes to diagnostic, action or respond
    diagnostic returns to supervisor
    action returns to supervisor
    respond ends the turn
"""

from typing import Any, Literal
from uuid import uuid4

from langgraph.graph import END, StateGraph

from service_manual_rag.agents.action_agent import create_action_agent
from service_manual_rag.agents.diagnostic_agent import create_diagnostic_agent
from service_manual_rag.agents.state import AgentState
from service_manual_rag.agents.supervisor import create_supervisor_agent
from service_manual_rag.utils.logging_config import get_logger

logger = get_logger(__name__)


def supervisor_node(state: AgentState) -> AgentState:
    """Process the latest user message and set ``next_agent``."""
    logger.info("Executing supervisor node")

    # After a specialist agent returns, the graph re-enters the supervisor in the same
    # turn. Routing the turn again would loop, so a processed turn goes to respond.
    if state.get("turn_id") is not None and state.get("last_processed_turn") == state["turn_id"]:
        logger.info("Turn already processed; routing to respond")
        state["next_agent"] = None
        return state
    state["last_processed_turn"] = state.get("turn_id")

    messages = state.get("messages", [])
    if messages and messages[-1]["role"] == "user":
        state = create_supervisor_agent().process_message(state, messages[-1]["content"])
    else:
        state["next_agent"] = None
    return state


async def diagnostic_node(state: AgentState) -> AgentState:
    """Run the diagnostic agent and store its result."""
    logger.info("Executing diagnostic node")
    result = await create_diagnostic_agent().diagnose(state)
    state["diagnostic_result"] = result
    state["evidence"] = [*state.get("evidence", []), *result.new_evidence]
    state["analyzed_media"] = [*state.get("analyzed_media", []), *state.get("uploaded_media", [])]
    state["uploaded_media"] = []
    state["turn_agent"] = "diagnostic"
    state["next_agent"] = None
    return state


def action_node(state: AgentState) -> AgentState:
    """Run the action agent to record the session and store its result."""
    logger.info("Executing action node")
    state["action_result"] = create_action_agent().record_session(state)
    state["turn_agent"] = "action"
    state["next_agent"] = None
    return state


def respond_node(state: AgentState) -> AgentState:
    """Append the supervisor's reply to the messages and wait for the user."""
    logger.info("Executing respond node")
    response = create_supervisor_agent().generate_response(state)
    state.setdefault("messages", []).append({"role": "assistant", "content": response})
    state["pending_user_input"] = True
    return state


def should_continue(state: AgentState) -> Literal["diagnostic", "action", "respond"]:
    """Return the name of the node that runs after the supervisor."""
    next_agent = state.get("next_agent")
    logger.debug("Router: next_agent = %s", next_agent)
    if next_agent in ("diagnostic", "action"):
        return next_agent
    return "respond"


def create_agent_workflow() -> Any:
    """Build and compile the agent graph."""
    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("diagnostic", diagnostic_node)
    workflow.add_node("action", action_node)
    workflow.add_node("respond", respond_node)
    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "diagnostic": "diagnostic",
            "action": "action",
            "respond": "respond",
        },
    )
    workflow.add_edge("diagnostic", "supervisor")
    workflow.add_edge("action", "supervisor")
    workflow.add_edge("respond", END)
    return workflow.compile()


_workflow: Any = None


def get_workflow() -> Any:
    """Return the process-wide compiled workflow, building it on first use."""
    global _workflow
    if _workflow is None:
        _workflow = create_agent_workflow()
    return _workflow


async def run_agent_workflow(state: AgentState) -> AgentState:
    """Run the workflow for one user turn. The new user message must already be in the state.

    Each call is a new turn with its own turn_id, so a repeated message is processed again.
    """
    logger.info("Running agent workflow for session: %s", state.get("session_id"))
    turn_state: AgentState = {
        **state,
        "turn_id": uuid4().hex,
        "turn_agent": None,
        "action_result": None,
    }
    # ainvoke is required because diagnostic_node is async
    return await get_workflow().ainvoke(turn_state)
