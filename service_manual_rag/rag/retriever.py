"""Hybrid retriever: Qdrant vector search plus optional Neo4j full-text graph search.

Both retrievers apply the same OEM, model and doc_id filters.
"""

from typing import Any

from service_manual_rag.config import Settings, get_settings
from service_manual_rag.embeddings import VectorIndexer
from service_manual_rag.utils.logging_config import get_logger

logger = get_logger(__name__)

GRAPH_QUERY = """
CALL db.index.fulltext.queryNodes('fulltext_index', $query)
YIELD node, score
RETURN node, score, labels(node) AS types
LIMIT $limit
"""

# Graph conditions for each vector filter key, matched against the provenance lists that
# GraphPopulator records. Equipment IDs have the form OEM_MODEL in upper case.
GRAPH_FILTER_CONDITIONS = {
    "doc_id": "$doc_id IN coalesce(node.doc_ids, [])",
    "equipment_id": "$equipment_id IN coalesce(node.equipment_ids, [])",
    "equipment_prefix": (
        "any(e IN coalesce(node.equipment_ids, []) WHERE e STARTS WITH $equipment_prefix)"
    ),
    "equipment_suffix": (
        "any(e IN coalesce(node.equipment_ids, []) WHERE e ENDS WITH $equipment_suffix)"
    ),
}
SUPPORTED_GRAPH_FILTERS = ("oem", "model", "doc_id")


def graph_filter_parameters(filters: dict | None) -> dict[str, str]:
    """Translate vector search filters into graph provenance parameters.

    Raises:
        ValueError: A filter key has no graph equivalent; searching without it would
            return other equipment's results.
    """
    filters = {key: value for key, value in (filters or {}).items() if value}
    unsupported = sorted(set(filters) - set(SUPPORTED_GRAPH_FILTERS))
    if unsupported:
        raise ValueError(f"Graph search cannot apply the filters {unsupported}")
    parameters = {}
    oem, model = filters.get("oem", "").upper(), filters.get("model", "").upper()
    if oem and model:
        parameters["equipment_id"] = f"{oem}_{model}"
    elif oem:
        parameters["equipment_prefix"] = f"{oem}_"
    elif model:
        parameters["equipment_suffix"] = f"_{model}"
    if "doc_id" in filters:
        parameters["doc_id"] = filters["doc_id"]
    return parameters


def build_graph_query(parameters: dict[str, str]) -> str:
    """Full-text query restricted to nodes whose provenance matches every parameter."""
    conditions = [GRAPH_FILTER_CONDITIONS[key] for key in parameters]
    if not conditions:
        return GRAPH_QUERY
    where = "WHERE " + " AND ".join(conditions) + "\n"
    return GRAPH_QUERY.replace("RETURN node", where + "RETURN node")


class HybridRetriever:
    """Retrieve from the vector store, the knowledge graph or both, per RETRIEVAL_MODE."""

    def __init__(
        self,
        vector_indexer: VectorIndexer | None = None,
        neo4j_manager: Any | None = None,
        settings: Settings | None = None,
    ):
        settings = settings or get_settings()
        self.mode = settings.retrieval_mode
        if self.mode == "graph" and neo4j_manager is None and not settings.graph_enabled:
            raise ValueError("RETRIEVAL_MODE=graph needs GRAPH_ENABLED=true and a running Neo4j")
        self.vector_indexer = vector_indexer or VectorIndexer()
        if neo4j_manager is None and settings.graph_enabled and self.mode != "vector":
            from service_manual_rag.knowledge_graph import Neo4JManager

            neo4j_manager = Neo4JManager()
        self.neo4j = neo4j_manager

    @property
    def effective_mode(self) -> str:
        """The retrievers that run: fused without a graph client is vector only."""
        if self.mode == "fused" and self.neo4j is None:
            return "vector"
        return self.mode

    def retrieve(
        self,
        query: str,
        vector_limit: int = 10,
        graph_limit: int = 10,
        filters: dict | None = None,
    ) -> dict:
        """Return vector and graph results.

        A failed graph search does not fail the query: the error is logged and
        returned under ``graph_error`` so callers can report it.
        """
        mode = self.effective_mode
        vector_results: list[dict] = []
        if mode != "graph":
            vector_results = self.vector_indexer.search(query, limit=vector_limit, filters=filters)
        graph_results: list[dict] = []
        graph_error = None
        if mode != "vector":
            try:
                graph_results = self._graph_search(query, graph_limit, filters)
            except Exception as error:  # noqa: BLE001
                logger.error(f"Graph search failed, continuing with vector results: {error}")
                graph_error = str(error)

        return {
            "vector_results": vector_results,
            "graph_results": graph_results,
            "graph_error": graph_error,
            "query": query,
        }

    def _graph_search(self, query: str, limit: int, filters: dict | None = None) -> list[dict]:
        filter_parameters = graph_filter_parameters(filters)
        records = self.neo4j.execute_query(
            build_graph_query(filter_parameters),
            {"query": query, "limit": limit, **filter_parameters},
        )
        return [
            {"node": dict(record["node"]), "score": record["score"], "types": record["types"]}
            for record in records
        ]
