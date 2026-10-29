"""Parse a manual, index its chunks and, when enabled, populate the knowledge graph."""

import asyncio
import tempfile
import time
from pathlib import Path

import streamlit as st

from service_manual_rag.config import get_settings
from service_manual_rag.embeddings import VectorIndexer
from service_manual_rag.processing import ChunkLevel, PDFParser, ProcessingPipeline

INDEXED_LEVELS = [ChunkLevel.SECTION, ChunkLevel.SUBSECTION, ChunkLevel.SEMANTIC]
SECONDS_PER_MB = {"fast": 2, "hi_res": 30, "ocr_only": 60}
UPLOAD_TYPES = ["pdf", "md", "markdown", "txt"]
STRATEGY_HELP = """
- **fast**: heuristic parsing, about 5 to 10 seconds per document.
- **hi_res**: layout model parsing, slower and needs 2 to 4 GB of RAM.
- **ocr_only**: OCR for scanned documents, the slowest option.

PDF parsing needs the `pdf` extra (`pip install -e ".[pdf]"`).
"""


def graph_skip_reason() -> str | None:
    """Return why graph population is skipped, or None when it runs."""
    settings = get_settings()
    if not settings.graph_enabled:
        return "GRAPH_ENABLED is false"
    if not settings.use_openai:
        return "entity extraction needs OpenAI mode"
    return None


def save_upload(uploaded_file, directory: Path) -> Path:
    """Write the upload under its own file name, which also sets the document ID."""
    path = directory / Path(uploaded_file.name).name
    path.write_bytes(uploaded_file.getbuffer())
    return path


def populate_graph(chunks: list) -> dict:
    from service_manual_rag.graph_builder import GraphPopulator

    return asyncio.run(GraphPopulator().populate_from_chunks(chunks))


def process_upload(uploaded_file, oem: str, model: str, strategy: str, extract_tables: bool):
    """Run parse, index and graph steps with progress output; return a summary dict."""
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    pipeline = ProcessingPipeline(
        pdf_parser=PDFParser(strategy=strategy, infer_table_structure=extract_tables)
    )

    status_text.text("Step 1 of 3: parsing")
    start = time.time()
    with tempfile.TemporaryDirectory() as directory:
        path = save_upload(uploaded_file, Path(directory))
        parsed_doc, chunks = pipeline.process(path, oem=oem or None, model=model or None)
    timings = {"parse": time.time() - start}
    progress_bar.progress(0.33)

    status_text.text("Step 2 of 3: indexing to the vector database")
    step_start = time.time()
    index_stats = VectorIndexer().index_chunks(chunks, levels_to_index=INDEXED_LEVELS)
    timings["index"] = time.time() - step_start
    progress_bar.progress(0.66)

    graph_stats = None
    skip_reason = graph_skip_reason()
    if skip_reason is None:
        status_text.text("Step 3 of 3: populating the knowledge graph")
        step_start = time.time()
        graph_stats = populate_graph(chunks)
        timings["graph"] = time.time() - step_start
    progress_bar.progress(1.0)
    status_text.empty()

    return {
        "statistics": ProcessingPipeline.get_statistics(parsed_doc, chunks),
        "indexing": index_stats,
        "graph": graph_stats,
        "graph_skipped": skip_reason,
        "timings": {name: round(seconds, 2) for name, seconds in timings.items()},
    }


def show_summary(summary: dict) -> None:
    stats = summary["statistics"]
    columns = st.columns(4)
    columns[0].metric("Pages", stats["total_pages"])
    columns[1].metric("Elements", stats["total_elements"])
    columns[2].metric("Chunks", stats["total_chunks"])
    columns[3].metric("Indexed", summary["indexing"]["indexed_chunks"])

    st.success(f"Processing complete in {sum(summary['timings'].values()):.1f} s.")
    if summary["graph"]:
        st.info(
            f"Knowledge graph: {summary['graph']['components_created']} components and "
            f"{summary['graph']['procedures_created']} procedures created."
        )
    else:
        st.info(f"Knowledge graph step skipped: {summary['graph_skipped']}.")
    with st.expander("Detailed statistics"):
        st.json(summary)


def show_vector_statistics() -> None:
    st.markdown("#### Vector database")
    try:
        indexer = VectorIndexer()
        if not indexer.qdrant_manager.collection_exists():
            st.info("No collection yet. Process a document to create it.")
            return
        stats = indexer.get_statistics()
    except Exception as error:  # noqa: BLE001 (UI boundary: any failure is shown)
        st.warning(f"Vector database statistics unavailable: {error}")
        return
    st.metric("Points", stats["qdrant"]["points_count"])
    st.metric("Collection", stats["qdrant"]["name"])
    st.metric("Embedding model", stats["embeddings"]["model"])


def show_graph_statistics() -> None:
    st.markdown("#### Knowledge graph")
    if not get_settings().graph_enabled:
        st.info("Disabled. Set GRAPH_ENABLED=true to use Neo4j.")
        return
    try:
        from service_manual_rag.knowledge_graph import Neo4JManager

        neo4j = Neo4JManager()
        try:
            graph_stats = neo4j.get_statistics()
        finally:
            neo4j.close()
    except Exception as error:  # noqa: BLE001 (UI boundary: any failure is shown)
        st.warning(f"Knowledge graph statistics unavailable: {error}")
        return
    st.metric("Nodes", graph_stats.get("total_nodes", 0))
    st.metric("Relationships", graph_stats.get("total_relationships", 0))
    st.metric("Equipment nodes", graph_stats.get("Equipment_count", 0))
    st.metric("Manual nodes", graph_stats.get("Manual_count", 0))


st.title("Process and Index Documents")
st.markdown(
    "Parse a manual, create hierarchical chunks and index them into the vector database. "
    "The knowledge graph is populated as well when `GRAPH_ENABLED=true` in OpenAI mode."
)

st.subheader("PDF parsing options")
strategy_col, tables_col = st.columns(2)
with strategy_col:
    strategy = st.selectbox("Parsing strategy", options=list(SECONDS_PER_MB), help=STRATEGY_HELP)
with tables_col:
    extract_tables = st.checkbox(
        "Extract table structure",
        value=True,
        help="Infer table structure for better table extraction; increases processing time.",
    )

st.markdown("---")
st.subheader("Upload document")
uploaded_file = st.file_uploader("Upload a manual (.pdf, .md or .txt)", type=UPLOAD_TYPES)

if uploaded_file:
    oem_col, model_col = st.columns(2)
    front_matter_help = "Optional for Markdown files whose front matter sets oem and model."
    with oem_col:
        oem = st.text_input("OEM", placeholder="ExampleCo", help=front_matter_help).strip()
    with model_col:
        model = st.text_input("Model", placeholder="CP-200", help=front_matter_help).strip()

    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.info(f"File: {uploaded_file.name} ({file_size_mb:.2f} MB)")
    if uploaded_file.name.lower().endswith(".pdf"):
        estimate = file_size_mb * SECONDS_PER_MB[strategy]
        st.caption(f"Estimated PDF parsing time: {estimate:.0f} s")

    if st.button("Process document", type="primary", width="stretch"):
        try:
            show_summary(process_upload(uploaded_file, oem, model, strategy, extract_tables))
        except Exception as error:  # noqa: BLE001 (UI boundary: any failure is shown)
            st.error(f"Processing failed: {error}")

st.markdown("---")
st.subheader("System statistics")
vector_col, graph_col = st.columns(2)
with vector_col:
    show_vector_statistics()
with graph_col:
    show_graph_statistics()
