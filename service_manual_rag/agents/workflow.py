"""LangGraph workflow for the supervisor, diagnostic and action agent graph.

Flow per user turn:
    supervisor routes to diagnostic, action or respond
    diagnostic returns to supervisor
    action returns to supervisor
    respond ends the turn
"""

from typing import Any, Literal

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

    messages = state.get("messages", [])
    current_message = None
    if messages and messages[-1]["role"] == "user":
        current_message = messages[-1]["content"]

    # After a specialist agent returns, the graph re-enters the supervisor with the same
    # message. Routing it again would loop, so a message that already has a result is
    # sent to the respond node instead.
    last_processed = state.get("last_processed_message")
    has_result = state.get("diagnostic_result") or state.get("action_result")

    if has_result and current_message == last_processed:
        logger.info("Message already processed; routing to respond")
        state["next_agent"] = None
        return state
    if has_result:
        logger.info("New message; clearing previous results")
        state["diagnostic_result"] = None
        state["action_result"] = None

    if current_message:
        state = create_supervisor_agent().process_message(state, current_message)
        state["last_processed_message"] = current_message

    return state


async def diagnostic_node(state: AgentState) -> AgentState:
    """Run the diagnostic agent and store its result."""
    logger.info("Executing diagnostic node")
    state["diagnostic_result"] = await create_diagnostic_agent().diagnose(state)
    state["next_agent"] = None
    return state


def action_node(state: AgentState) -> AgentState:
    """Run the action agent to record the session and store its result."""
    logger.info("Executing action node")
    state["action_result"] = create_action_agent().record_session(state)
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
    """Run the workflow for one user turn. The new user message must already be in the state."""
    logger.info("Running agent workflow for session: %s", state.get("session_id"))
    # ainvoke is required because diagnostic_node is async
    return await get_workflow().ainvoke(state)
