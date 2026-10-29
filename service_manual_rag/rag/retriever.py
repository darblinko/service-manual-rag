"""Hybrid retriever: Qdrant vector search plus optional Neo4j full-text graph search."""

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


class HybridRetriever:
    """Retrieve from the vector store and, when GRAPH_ENABLED is true, the knowledge graph."""

    def __init__(
        self,
        vector_indexer: VectorIndexer | None = None,
        neo4j_manager: Any | None = None,
        settings: Settings | None = None,
    ):
        settings = settings or get_settings()
        self.vector_indexer = vector_indexer or VectorIndexer()
        if neo4j_manager is None and settings.graph_enabled:
            from service_manual_rag.knowledge_graph import Neo4JManager

            neo4j_manager = Neo4JManager()
        self.neo4j = neo4j_manager

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
        vector_results = self.vector_indexer.search(query, limit=vector_limit, filters=filters)
        graph_results: list[dict] = []
        graph_error = None
        if self.neo4j is not None:
            try:
                graph_results = self._graph_search(query, graph_limit)
            except Exception as error:  # noqa: BLE001
                logger.error(f"Graph search failed, continuing with vector results: {error}")
                graph_error = str(error)

        return {
            "vector_results": vector_results,
            "graph_results": graph_results,
            "graph_error": graph_error,
            "query": query,
        }

    def _graph_search(self, query: str, limit: int) -> list[dict]:
        records = self.neo4j.execute_query(GRAPH_QUERY, {"query": query, "limit": limit})
        return [
            {"node": dict(record["node"]), "score": record["score"], "types": record["types"]}
            for record in records
        ]
