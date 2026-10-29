"""Ask a question and get an answer with cited sources from the indexed manuals."""

import streamlit as st

from service_manual_rag.rag import RAGPipeline

EXAMPLE_QUESTIONS = [
    "What is the torque for the CP-200 impeller nut?",
    "How often does the CP-200 drive end motor bearing need grease?",
    "The RSC-15 display shows E06. What does that mean?",
    "How much oil does an RSC-15 oil change need, and what grade?",
]
PREVIEW_CHARS = 500


def build_filters(oem: str, model: str) -> dict | None:
    filters = {key: value for key, value in (("oem", oem), ("model", model)) if value}
    return filters or None


def show_sources(sources: list[dict]) -> None:
    st.markdown("### Sources")
    for number, source in enumerate(sources, 1):
        with st.expander(f"Source {number} (RRF score {source['rrf_score']:.3f})"):
            st.markdown(f"**From**: {', '.join(source['sources'])}")
            if "vector_rank" in source:
                st.markdown(f"**Vector rank**: {source['vector_rank']}")
            if "graph_rank" in source:
                st.markdown(f"**Graph rank**: {source['graph_rank']}")
            if source.get("section_name"):
                st.markdown(f"**Section**: {source['section_name']}")
            content = source["content"]
            if len(content) > PREVIEW_CHARS:
                content = content[:PREVIEW_CHARS] + "..."
            st.text(content)


def show_result(result: dict) -> None:
    st.markdown("### Answer")
    st.markdown(result["answer"])
    st.markdown("---")
    show_sources(result["sources"])

    metadata = result["metadata"]
    if metadata.get("graph_error"):
        st.warning(f"Graph search failed; showing vector results only: {metadata['graph_error']}")
    st.markdown("---")
    st.markdown("### Query metadata")
    retrieved_col, used_col, model_col = st.columns(3)
    retrieved_col.metric("Sources retrieved", metadata["total_retrieved"])
    used_col.metric("Sources used", metadata["sources_used"])
    model_col.metric("Answer generator", metadata["model"])


st.title("Query Service Manuals")
st.markdown("Ask a question about the indexed manuals and get an answer with citations.")

question = st.text_area(
    "Ask a question",
    placeholder="For example: What is the torque for the CP-200 impeller nut?",
    height=100,
)

with st.expander("Advanced options"):
    left_col, right_col = st.columns(2)
    with left_col:
        top_k = st.slider("Number of sources", 1, 10, 5)
        oem_filter = st.text_input("Filter by OEM (optional)", placeholder="ExampleCo").strip()
    with right_col:
        model_filter = st.text_input("Filter by model (optional)", placeholder="CP-200").strip()

if st.button("Search") and question:
    with st.spinner("Searching and generating answer..."):
        try:
            result = RAGPipeline().query(
                question, top_k=top_k, filters=build_filters(oem_filter, model_filter)
            )
        except Exception as error:  # noqa: BLE001 (UI boundary: any failure is shown)
            st.error(f"Query failed: {error}")
        else:
            show_result(result)

st.markdown("---")
st.markdown("### Example questions")
for example in EXAMPLE_QUESTIONS:
    st.markdown(f"- {example}")
