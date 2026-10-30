"""Conversational troubleshooting with the supervisor, diagnostic and action agents.

The agent workflow needs the ``agents`` extra (langgraph). In offline mode the agents use
keyword rules and the answers are manual excerpts. The knowledge graph panel needs
GRAPH_ENABLED=true and the optional streamlit-agraph package.
"""

import asyncio

import streamlit as st

from service_manual_rag.agents.session_manager import get_session_manager
from service_manual_rag.agents.state import AgentState, DiagnosticResult, Hypothesis
from service_manual_rag.agents.vlm_service import get_vlm_service
from service_manual_rag.config import get_settings
from service_manual_rag.utils.logging_config import get_logger

logger = get_logger(__name__)

OFFLINE_MESSAGE = (
    "Offline mode: the supervisor and the diagnostic agent use keyword rules, and answers "
    "are excerpts from the indexed manuals. Set OPENAI_API_KEY and restart the app for "
    "model-generated answers and hypotheses."
)
# Errors LocalSessionStore raises when a session file cannot be written or read
SESSION_ERRORS = (OSError, TypeError, ValueError, KeyError)
MEDIA_TYPES = ["jpg", "jpeg", "png", "mp4", "mov"]
MAX_SOURCES = 10
EQUIPMENT_COLOR = "#1f77b4"
DEFAULT_NODE_COLOR = "#7f7f7f"
NODE_COLORS = {
    "Component": "#ff7f0e",
    "Procedure": "#2ca02c",
    "Symptom": "#d62728",
    "FailureMode": "#9467bd",
    "SafetyHazard": "#e377c2",
    "Part": "#8c564b",
    "Tool": "#bcbd22",
    "Section": "#17becf",
    "Skill": "#17becf",
}
NODE_LABEL_KEYS = (
    "name",
    "description",
    "part_name",
    "action_type",
    "hazard_type",
    "tool_name",
    "skill_name",
)
SUBGRAPH_QUERY = """
MATCH (e:Equipment {oem: $oem, model: $model})-[r]-(n)
RETURN e.id AS equipment_id, type(r) AS rel_type, labels(n)[0] AS node_type, n AS node
LIMIT 20
"""
HELP_TEXT = """
**Getting help**
1. Select or enter the equipment in the sidebar.
2. Describe the symptom or ask about maintenance, troubleshooting or specifications.
3. Review the retrieved sources in this panel.

**Tips**
- Be specific about symptoms and when the issue started.
- Mention recent changes or maintenance.
- Upload clear images of the problem area.

**Example:** "What is the torque for the CP-200 impeller nut?"
"""


def get_agent_state() -> AgentState:
    """Return the current session state, starting a new session if it expired."""
    manager = get_session_manager()
    state = manager.get_session(st.session_state.agent_session_id)
    if state is None:
        st.session_state.agent_session_id = manager.create_session()
        state = manager.get_session(st.session_state.agent_session_id)
    return state


def save_agent_state(state: AgentState) -> None:
    get_session_manager().update_session(st.session_state.agent_session_id, state)


def start_new_session() -> None:
    st.session_state.agent_session_id = get_session_manager().create_session()
    st.session_state.chat_history = []


def attach_media(state: AgentState, uploaded_files: list) -> None:
    vlm_service = get_vlm_service()
    media = state.setdefault("uploaded_media", [])
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        path = vlm_service.save_media(file_bytes, f"{state['session_id']}_{uploaded_file.name}")
        media.append(
            {
                "name": uploaded_file.name,
                "path": path,
                "type": uploaded_file.type,
                "size": len(file_bytes),
            }
        )


def process_message(user_input: str, uploaded_files: list) -> str:
    """Run one user turn through the agent workflow and return the reply."""
    from service_manual_rag.agents.workflow import run_agent_workflow

    state = get_agent_state()
    state.setdefault("messages", []).append({"role": "user", "content": user_input})
    if uploaded_files:
        attach_media(state, uploaded_files)

    updated_state = asyncio.run(run_agent_workflow(state))
    save_agent_state(updated_state)

    messages = updated_state.get("messages", [])
    if messages and messages[-1]["role"] == "assistant":
        return messages[-1]["content"]
    return "The agent returned no reply."


def load_onboarded_equipment() -> list[dict]:
    """Return the equipment in the knowledge graph; empty when the graph is disabled."""
    if not get_settings().graph_enabled:
        return []
    from service_manual_rag.knowledge_graph import Neo4JManager

    neo4j = Neo4JManager()
    try:
        return neo4j.get_onboarded_equipment()
    finally:
        neo4j.close()


def set_equipment(state: AgentState, oem: str, model: str) -> None:
    state["equipment_oem"] = oem
    state["equipment_model"] = model
    try:
        save_agent_state(state)
    except SESSION_ERRORS as error:
        st.error(f"Could not save the session: {error}")
        return
    st.rerun()


def show_equipment_selector(state: AgentState) -> None:
    st.header("Equipment")
    try:
        equipment = load_onboarded_equipment()
    except Exception as error:  # noqa: BLE001 (UI boundary: any graph failure is shown)
        st.error(f"Equipment list unavailable: {error}")
        equipment = []

    options = {"Manual entry": None}
    options.update({f"{eq['oem']} {eq['model']} ({eq['category']})": eq for eq in equipment})
    choice = st.selectbox("Select equipment", list(options), help="Onboarded or manual entry")
    selected = options[choice]

    if selected is not None:
        st.info(f"**Selected:** {selected['oem']} {selected['model']}")
        if st.button("Use this equipment", width="stretch"):
            set_equipment(state, selected["oem"], selected["model"])
        return

    oem_col, model_col = st.columns(2)
    oem = oem_col.text_input(
        "OEM", value=state.get("equipment_oem") or "", placeholder="ExampleCo"
    ).strip()
    model = model_col.text_input(
        "Model", value=state.get("equipment_model") or "", placeholder="CP-200"
    ).strip()
    if st.button("Set equipment", width="stretch"):
        if oem and model:
            set_equipment(state, oem, model)
        else:
            st.warning("Enter both OEM and model.")


def show_session_info(state: AgentState) -> None:
    st.header("Session")
    status = state.get("status")
    st.write(f"**Session ID:** `{st.session_state.agent_session_id[:8]}`")
    st.write(f"**Status:** {getattr(status, 'value', status) or 'unknown'}")
    if state.get("equipment_oem") or state.get("equipment_model"):
        equipment = f"{state.get('equipment_oem') or ''} {state.get('equipment_model') or ''}"
        st.write(f"**Equipment:** {equipment.strip()}")
    if state.get("symptoms"):
        st.write(f"**Symptoms:** {', '.join(state['symptoms'][:3])}")
    if st.button("Start new session", width="stretch"):
        try:
            start_new_session()
        except SESSION_ERRORS as error:
            st.error(f"Could not create a session: {error}")
        else:
            st.rerun()


def confidence_label(confidence: float) -> str:
    if confidence > 0.75:
        return "high"
    if confidence > 0.5:
        return "medium"
    return "low"


def show_hypothesis(number: int, hypothesis: Hypothesis) -> None:
    confidence = hypothesis.confidence
    st.markdown(f"**{number}. {hypothesis.description}**")
    st.markdown(f"Confidence: **{confidence:.0%}** ({confidence_label(confidence)})")
    if hypothesis.supporting_evidence:
        st.markdown("**Supporting evidence:**")
        st.markdown("\n".join(f"- {item}" for item in hypothesis.supporting_evidence))
    if hypothesis.tests_suggested:
        st.markdown("**Suggested tests:**")
        st.markdown("\n".join(f"- {item}" for item in hypothesis.tests_suggested))
    st.divider()


def show_diagnostic_details(diagnosis: DiagnosticResult) -> None:
    with st.expander("Diagnostic details", expanded=True):
        if diagnosis.hypotheses:
            st.subheader("Hypotheses")
            ranked = sorted(diagnosis.hypotheses, key=lambda h: h.confidence, reverse=True)
            for number, hypothesis in enumerate(ranked[:5], 1):
                show_hypothesis(number, hypothesis)
        if diagnosis.observations:
            st.subheader("Observations")
            for observation in diagnosis.observations[:5]:
                st.info(observation)
        if diagnosis.safety_warnings:
            st.subheader("Safety warnings")
            for warning in diagnosis.safety_warnings:
                st.warning(warning)
        if diagnosis.diagnosis:
            st.success(f"**Diagnosis:** {diagnosis.diagnosis}")
        if diagnosis.recommended_action:
            st.info(f"**Recommended action:** {diagnosis.recommended_action}")
        if diagnosis.evidence_needed:
            st.warning(f"**Additional evidence needed:** {diagnosis.evidence_needed}")


def show_chat() -> None:
    with st.container(height=500):
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    uploaded_files = st.file_uploader(
        "Upload images or videos",
        type=MEDIA_TYPES,
        accept_multiple_files=True,
        help="Images or videos of the equipment for visual analysis",
    )
    user_input = st.chat_input("Describe the equipment issue...")
    if not user_input:
        return

    with st.spinner("The agent is working..."):
        try:
            response = process_message(user_input, uploaded_files)
        except Exception as error:
            logger.exception("Agent workflow failed")
            st.error(f"The agent failed: {error}")
            return
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.rerun()


def show_sources(diagnosis: DiagnosticResult | None) -> None:
    st.header("Retrieved sources")
    if not (diagnosis and diagnosis.relevant_manual_sections):
        st.info("No sources yet. Ask a question to see the retrieved manual sections here.")
        return
    sections = diagnosis.relevant_manual_sections
    st.caption(f"Found {len(sections)} relevant sections")
    for number, section in enumerate(sections[:MAX_SOURCES], 1):
        title = f"Source {number} (score {section.get('score', 0):.3f})"
        with st.expander(title, expanded=number == 1):
            st.text_area(
                f"Source {number} content",
                section.get("content", ""),
                height=150,
                label_visibility="collapsed",
                key=f"source_{number}_content",
            )
            if section.get("sources"):
                st.markdown(f"**From:** {', '.join(section['sources'])}")
            if section.get("metadata"):
                st.json(section["metadata"])


def query_equipment_subgraph(oem: str, model: str) -> list[dict]:
    from service_manual_rag.knowledge_graph import Neo4JManager

    neo4j = Neo4JManager()
    try:
        return neo4j.execute_query(SUBGRAPH_QUERY, {"oem": oem, "model": model})
    finally:
        neo4j.close()


def node_label(node: dict) -> str:
    for key in NODE_LABEL_KEYS:
        if node.get(key):
            return str(node[key])[:30]
    return str(node.get("id", "node"))[:30]


def node_tooltip(node_type: str, node: dict) -> str:
    lines = [f"<b>{node_type}</b>"]
    lines.extend(
        f"{key}: {str(value)[:50]}" for key, value in node.items() if key != "id" and value
    )
    return "<br>".join(lines)


def build_subgraph(oem: str, model: str, rows: list[dict]) -> tuple[list, list]:
    """Return agraph nodes and edges for the equipment and its direct neighbours."""
    from streamlit_agraph import Edge, Node

    center_id = rows[0]["equipment_id"]
    nodes = {
        center_id: Node(
            id=center_id, label=f"{oem} {model}", size=30, color=EQUIPMENT_COLOR, shape="box"
        )
    }
    edges = []
    for number, row in enumerate(rows):
        node = row["node"] or {}
        node_id = node.get("id") or f"node_{number}"
        if node_id not in nodes:
            nodes[node_id] = Node(
                id=node_id,
                label=node_label(node),
                size=20,
                color=NODE_COLORS.get(row["node_type"], DEFAULT_NODE_COLOR),
                title=node_tooltip(row["node_type"], node),
            )
        edges.append(Edge(source=center_id, target=node_id, label=row["rel_type"]))
    return list(nodes.values()), edges


def show_knowledge_graph(state: AgentState, diagnosis: DiagnosticResult | None) -> None:
    st.header("Knowledge graph")
    if not get_settings().graph_enabled:
        st.info("Disabled. Set GRAPH_ENABLED=true to show the equipment subgraph.")
        return
    oem, model = state.get("equipment_oem"), state.get("equipment_model")
    if not (diagnosis and diagnosis.relevant_manual_sections and oem and model):
        st.info("The equipment subgraph appears here after sources are retrieved.")
        return
    try:
        from streamlit_agraph import Config, agraph
    except ImportError:
        st.warning("Graph visualization needs `streamlit-agraph`: `pip install streamlit-agraph`")
        return
    try:
        rows = query_equipment_subgraph(oem, model)
    except Exception as error:  # noqa: BLE001 (UI boundary: any graph failure is shown)
        st.error(f"Knowledge graph query failed: {error}")
        return
    if not rows:
        st.info("No graph data for this equipment.")
        return
    nodes, edges = build_subgraph(oem, model, rows)
    st.caption(f"Showing {len(nodes)} nodes and {len(edges)} relationships")
    config = Config(width="100%", height=400, directed=True, physics=True, hierarchical=False)
    agraph(nodes=nodes, edges=edges, config=config)


st.set_page_config(page_title="Agent Chat", layout="wide")
st.title("Agent Chat")

if not get_settings().use_openai:
    st.info(OFFLINE_MESSAGE)

st.markdown(
    """
Troubleshooting chat backed by three agents:
- **Supervisor**: guides the conversation and routes each turn.
- **Diagnostic agent**: searches the manuals, analyzes uploaded images or videos and ranks
  hypotheses.
- **Action agent**: records troubleshooting sessions for later reference.
"""
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
try:
    if "agent_session_id" not in st.session_state:
        st.session_state.agent_session_id = get_session_manager().create_session()
    agent_state = get_agent_state()
except SESSION_ERRORS as error:
    st.error(f"Agent session storage failed: {error}")
    st.stop()
diagnostic_result = agent_state.get("diagnostic_result")

with st.sidebar:
    show_equipment_selector(agent_state)
    st.divider()
    show_session_info(agent_state)
    if diagnostic_result:
        show_diagnostic_details(diagnostic_result)

chat_col, sources_col = st.columns([3, 1])
with chat_col:
    show_chat()
with sources_col:
    show_sources(diagnostic_result)
    st.divider()
    show_knowledge_graph(agent_state, diagnostic_result)
    st.divider()
    with st.expander("How to use"):
        st.markdown(HELP_TEXT)

st.divider()
st.caption(
    "The more detail you give about the equipment and symptoms, the better the diagnosis. "
    "Image and video analysis needs GOOGLE_API_KEY."
)
