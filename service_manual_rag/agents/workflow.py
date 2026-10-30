"""LangGraph workflow for the supervisor, diagnostic and action agent graph.

Flow per user turn:
    supervisor routes to diagnostic, action or respond
    diagnostic returns to supervisor
    action returns to supervisor
    respond ends the turn
"""

from collections.abc import Callable
from typing import Any, Literal
from uuid import uuid4

from langgraph.graph import END, StateGraph

from service_manual_rag.agents.action_agent import create_action_agent
from service_manual_rag.agents.diagnostic_agent import create_diagnostic_agent
from service_manual_rag.agents.state import AgentState
from service_manual_rag.agents.supervisor import create_supervisor_agent
from service_manual_rag.utils.logging_config import get_logger

logger = get_logger(__name__)


class AgentRuntime:
    """Builds each agent on first use and reuses it for every later turn.

    One runtime belongs to one compiled workflow, so models, the re-ranker and database
    drivers are loaded once per process instead of once per turn. ``close`` releases
    the Neo4j drivers the agents opened.
    """

    def __init__(self):
        self._agents: dict[str, Any] = {}

    def _get(self, name: str, factory: Callable[[], Any]) -> Any:
        if name not in self._agents:
            self._agents[name] = factory()
        return self._agents[name]

    # The factories are looked up at call time so that tests can replace them
    def supervisor(self) -> Any:
        return self._get("supervisor", lambda: create_supervisor_agent())

    def diagnostic(self) -> Any:
        return self._get("diagnostic", lambda: create_diagnostic_agent())

    def action(self) -> Any:
        return self._get("action", lambda: create_action_agent())

    def close(self) -> None:
        """Close every Neo4j manager held by the agents and forget the agents."""
        managers = {}
        for agent in self._agents.values():
            retriever = getattr(getattr(agent, "rag", None), "retriever", None)
            for manager in (getattr(agent, "kg", None), getattr(retriever, "neo4j", None)):
                if manager is not None and hasattr(manager, "close"):
                    managers[id(manager)] = manager
        for manager in managers.values():
            manager.close()
        self._agents = {}


def supervisor_node(state: AgentState, runtime: AgentRuntime) -> AgentState:
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
        state = runtime.supervisor().process_message(state, messages[-1]["content"])
    else:
        state["next_agent"] = None
    return state


async def diagnostic_node(state: AgentState, runtime: AgentRuntime) -> AgentState:
    """Run the diagnostic agent and store its result."""
    logger.info("Executing diagnostic node")
    result = await runtime.diagnostic().diagnose(state)
    state["diagnostic_result"] = result
    state["evidence"] = [*state.get("evidence", []), *result.new_evidence]
    state["analyzed_media"] = [*state.get("analyzed_media", []), *state.get("uploaded_media", [])]
    state["uploaded_media"] = []
    state["turn_agent"] = "diagnostic"
    state["next_agent"] = None
    return state


def action_node(state: AgentState, runtime: AgentRuntime) -> AgentState:
    """Run the action agent to record the session and store its result."""
    logger.info("Executing action node")
    state["action_result"] = runtime.action().record_session(state)
    state["turn_agent"] = "action"
    state["next_agent"] = None
    return state


def respond_node(state: AgentState, runtime: AgentRuntime) -> AgentState:
    """Append the supervisor's reply to the messages and wait for the user."""
    logger.info("Executing respond node")
    response = runtime.supervisor().generate_response(state)
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


def create_agent_workflow(runtime: AgentRuntime | None = None) -> Any:
    """Build and compile the agent graph; its nodes share one AgentRuntime.

    The runtime is available as the ``agent_runtime`` attribute of the result.
    """
    runtime = runtime or AgentRuntime()

    def supervisor(state: AgentState) -> AgentState:
        return supervisor_node(state, runtime)

    async def diagnostic(state: AgentState) -> AgentState:
        return await diagnostic_node(state, runtime)

    def action(state: AgentState) -> AgentState:
        return action_node(state, runtime)

    def respond(state: AgentState) -> AgentState:
        return respond_node(state, runtime)

    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("diagnostic", diagnostic)
    workflow.add_node("action", action)
    workflow.add_node("respond", respond)
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
    compiled = workflow.compile()
    compiled.agent_runtime = runtime
    return compiled


_workflow: Any = None


def get_workflow() -> Any:
    """Return the process-wide compiled workflow, building it on first use."""
    global _workflow
    if _workflow is None:
        _workflow = create_agent_workflow()
    return _workflow


def close_workflow() -> None:
    """Close the agents' database drivers and drop the process-wide workflow."""
    global _workflow
    if _workflow is not None:
        _workflow.agent_runtime.close()
    _workflow = None


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
