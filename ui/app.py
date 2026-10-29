"""Streamlit entry point for the service manual RAG UI.

Run from the repository root with ``streamlit run ui/app.py``.
"""

import streamlit as st

from service_manual_rag.config import Settings, get_settings
from service_manual_rag.vector_db import QdrantManager


def check_qdrant() -> tuple[bool, str]:
    """Return (ok, message) for the configured Qdrant collection."""
    try:
        qdrant = QdrantManager()
        if not qdrant.collection_exists():
            return True, "No collection yet"
        return True, f"{qdrant.count_points():,} vectors"
    except Exception as error:  # noqa: BLE001 (UI boundary: status shows any failure)
        return False, str(error)


def check_neo4j(settings: Settings) -> tuple[bool, str]:
    """Return (ok, message) for the knowledge graph; disabled counts as ok."""
    if not settings.graph_enabled:
        return True, "Disabled"
    try:
        from service_manual_rag.knowledge_graph import Neo4JManager

        neo4j = Neo4JManager()
        try:
            if not neo4j.health_check():
                return False, "Health check failed"
            total_nodes = neo4j.get_statistics().get("total_nodes", 0)
            return True, f"{total_nodes:,} nodes"
        finally:
            neo4j.close()
    except Exception as error:  # noqa: BLE001 (UI boundary: status shows any failure)
        return False, str(error)


def llm_mode_label(settings: Settings) -> str:
    return "openai" if settings.use_openai else "offline"


def show_status(settings: Settings) -> bool:
    """Render the status row and return True when every component is available."""
    with st.spinner("Checking system status..."):
        qdrant_ok, qdrant_msg = check_qdrant()
        neo4j_ok, neo4j_msg = check_neo4j(settings)

    qdrant_col, graph_col, mode_col = st.columns(3)
    with qdrant_col:
        st.metric(f"Qdrant ({settings.qdrant_mode})", qdrant_msg if qdrant_ok else "Unavailable")
        if not qdrant_ok:
            st.error(f"Qdrant: {qdrant_msg}")
    with graph_col:
        st.metric("Knowledge graph", neo4j_msg if neo4j_ok else "Unavailable")
        if not neo4j_ok:
            st.error(f"Neo4j: {neo4j_msg}")
    with mode_col:
        st.metric(
            "LLM mode",
            llm_mode_label(settings),
            help="openai: OpenAI embeddings and answers. "
            "offline: hashing embedder and extractive answers, no remote API calls.",
        )
    return qdrant_ok and neo4j_ok


TROUBLESHOOTING = """
**Qdrant.** The default `QDRANT_MODE=local` stores vectors on disk and needs no server.
For `QDRANT_MODE=server`, start the container with `docker compose up -d qdrant`.

**Knowledge graph.** Neo4j is optional. To use it, set `GRAPH_ENABLED=true` and
`NEO4J_PASSWORD`, then start it with `docker compose up -d neo4j`.

**LLM mode.** Set `OPENAI_API_KEY` in the environment or in `.env` for OpenAI mode,
or set `LLM_MODE=offline` to run without remote API calls. Restart the app after a change.
"""

st.set_page_config(
    page_title="Service Manual RAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Service Manual RAG")
st.markdown(
    """
Question answering over equipment service manuals with hybrid vector and
knowledge graph retrieval.

## Pages

1. **Process**: parse a .pdf, .md or .txt manual, chunk it and index it.
2. **Query**: ask a question and get an answer with cited sources.
3. **Documents**: view indexed documents and remove them.
4. **Agent Chat**: conversational troubleshooting with the agent workflow (OpenAI mode only).

Use the sidebar to switch pages.

## System status
"""
)

current_settings = get_settings()
all_ok = show_status(current_settings)

st.markdown("---")
if all_ok:
    st.success("All configured components are available.")
else:
    st.warning("Some components are not available. See the errors above.")
with st.expander("Configuration help", expanded=not all_ok):
    st.markdown(TROUBLESHOOTING)

st.markdown("---")
st.caption("Service Manual RAG | Streamlit, Qdrant, optional Neo4j")
