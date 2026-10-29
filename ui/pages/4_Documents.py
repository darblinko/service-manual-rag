"""View indexed documents, their chunks and knowledge graph links, and delete them."""

import streamlit as st
from utils.document_manager import DocumentManager

LEVEL_LABELS = {
    1: "Level 1 (document)",
    2: "Level 2 (sections)",
    3: "Level 3 (subsections)",
    4: "Level 4 (semantic)",
}
LEVEL_HELP = """
Chunks form a four level hierarchy:

- **Level 1 (document)**: document summary.
- **Level 2 (sections)**: major sections such as Installation or Maintenance.
- **Level 3 (subsections)**: subsections within a section.
- **Level 4 (semantic)**: fine-grained passages of about 500 tokens.

The Process page indexes levels 2 to 4; the command line ingest indexes level 4 by default.
"""


@st.cache_resource
def get_document_manager() -> DocumentManager:
    return DocumentManager()


def document_label(document: dict) -> str:
    return f"{document['oem']} {document['model']}: {document['file_name']}"


def filter_documents(documents: list[dict], search: str) -> list[dict]:
    needle = search.lower()
    return [
        d
        for d in documents
        if needle in d["oem"].lower()
        or needle in d["model"].lower()
        or needle in d["file_name"].lower()
    ]


def show_library_metrics(documents: list[dict]) -> None:
    columns = st.columns(4)
    columns[0].metric("Documents", len(documents))
    columns[1].metric("Chunks", sum(d["chunk_count"] for d in documents))
    columns[2].metric("OEMs", len({d["oem"] for d in documents}))
    columns[3].metric("Models", len({d["model"] for d in documents}))


def show_library_table(documents: list[dict]) -> None:
    rows = [
        {
            "OEM": d["oem"],
            "Model": d["model"],
            "File": d["file_name"],
            "Chunks": d["chunk_count"],
        }
        for d in documents
    ]
    st.dataframe(rows, width="stretch", hide_index=True)


def show_chunks_tab(document: dict) -> None:
    columns = st.columns(len(LEVEL_LABELS))
    for column, (level, label) in zip(columns, LEVEL_LABELS.items()):
        column.metric(label, document["chunks_by_level"].get(level, 0))
    st.caption(f"Document ID: {document['doc_id']}")
    with st.expander("About chunk levels"):
        st.markdown(LEVEL_HELP)


def show_graph_tab(impact: dict | None) -> None:
    if impact is None:
        st.info(
            "No knowledge graph data for this document. The graph is used only when "
            "GRAPH_ENABLED=true and the document was processed with it enabled."
        )
        return
    columns = st.columns(3)
    columns[0].metric("Sections", impact["sections"])
    columns[1].metric("Components", impact["components"])
    columns[2].metric("Other manuals for this equipment", impact["other_manuals"])
    if impact["other_manuals"] == 0:
        st.warning("This is the only manual for its Equipment node. Deleting it orphans that node.")


def show_vector_tab(manager: DocumentManager, document: dict) -> None:
    info = manager.collection_info() or {}
    columns = st.columns(3)
    columns[0].metric("Chunks indexed", document["chunk_count"])
    columns[1].metric("Collection", info.get("name", "n/a"))
    columns[2].metric("Vector size", info.get("vector_size", "n/a"))
    st.caption("Each point stores the chunk embedding, its text and its metadata payload.")


def show_delete_section(manager: DocumentManager, document: dict, impact: dict | None) -> None:
    st.markdown("### Delete document")
    with st.expander("Delete this document", expanded=False):
        lines = [f"- {document['chunk_count']} chunks from the vector database"]
        if impact is not None:
            lines.append(f"- the Manual node and {impact['sections']} Section nodes from Neo4j")
        st.warning("Deletion is permanent. This removes:\n\n" + "\n".join(lines))

        confirm = st.checkbox(
            f"I understand that deleting '{document_label(document)}' cannot be undone",
            key=f"confirm_delete_{document['doc_id']}",
        )
        if st.button("Delete document", type="primary", disabled=not confirm):
            try:
                result = manager.delete_document(document["doc_id"])
            except Exception as error:  # noqa: BLE001 (UI boundary: any failure is shown)
                st.error(f"Deletion failed: {error}")
                return
            message = f"Deleted {result['chunks_deleted']} chunks"
            if result["graph_nodes_deleted"] is not None:
                message += f" and {result['graph_nodes_deleted']} graph nodes"
            st.session_state["documents_flash"] = f"{message} for {document_label(document)}."
            st.rerun()


def load_graph_impact(manager: DocumentManager, doc_id: str) -> dict | None:
    try:
        return manager.get_graph_impact(doc_id)
    except Exception as error:  # noqa: BLE001 (UI boundary: any graph failure is shown)
        st.error(f"Knowledge graph query failed: {error}")
        return None


def show_document(manager: DocumentManager, document: dict) -> None:
    impact = load_graph_impact(manager, document["doc_id"])
    chunks_tab, graph_tab, vector_tab = st.tabs(["Chunks", "Knowledge graph", "Vector database"])
    with chunks_tab:
        show_chunks_tab(document)
    with graph_tab:
        show_graph_tab(impact)
    with vector_tab:
        show_vector_tab(manager, document)
    st.markdown("---")
    show_delete_section(manager, document, impact)


def show_page(manager: DocumentManager) -> None:
    with st.spinner("Loading documents..."):
        documents = manager.get_all_documents()
    if not documents:
        st.info("No documents found. Use the Process page to index documents.")
        return

    show_library_metrics(documents)
    st.markdown("---")
    search = st.text_input("Search documents", placeholder="Search by OEM, model or file name")
    filtered = filter_documents(documents, search) if search else documents
    if not filtered:
        st.warning(f"No documents match '{search}'")
        return

    show_library_table(filtered)
    st.markdown("---")
    st.subheader("Document details")
    by_label = {document_label(d): d for d in filtered}
    selected = st.selectbox("Select a document", options=list(by_label))
    show_document(manager, by_label[selected])


st.title("Document Management")
st.markdown("View indexed documents, their chunks and knowledge graph links, and delete them.")

flash = st.session_state.pop("documents_flash", None)
if flash:
    st.success(flash)

try:
    show_page(get_document_manager())
except Exception as error:  # noqa: BLE001 (UI boundary: any failure is shown)
    st.error(f"Document management is unavailable: {error}")
