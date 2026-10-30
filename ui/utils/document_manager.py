"""List and delete indexed documents across the vector store and the optional knowledge graph."""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from service_manual_rag.config import Settings, get_settings
from service_manual_rag.vector_db import QdrantManager

CHUNK_LEVELS = (1, 2, 3, 4)

GRAPH_IMPACT_QUERY = """
MATCH (m:Manual {id: $doc_id})
OPTIONAL MATCH (e:Equipment)-[:HAS_MANUAL]->(m)
OPTIONAL MATCH (m)-[:CONTAINS]->(s:Section)
OPTIONAL MATCH (e)-[:HAS_COMPONENT]->(c:Component)
WITH m, e, count(DISTINCT s) AS section_count, count(DISTINCT c) AS component_count
OPTIONAL MATCH (e)-[:HAS_MANUAL]->(other:Manual)
WHERE other.id <> $doc_id
RETURN section_count, component_count, count(DISTINCT other) AS other_manuals_count
"""

GRAPH_DELETE_QUERY = """
MATCH (m:Manual {id: $doc_id})
OPTIONAL MATCH (m)-[:CONTAINS]->(s:Section)
WITH m, collect(s) AS sections
WITH m, sections, size(sections) + 1 AS nodes_deleted
FOREACH (section IN sections | DETACH DELETE section)
DETACH DELETE m
RETURN nodes_deleted
"""

# Entities extracted only from this document. Equipment counts when this is its only manual.
ENTITY_IMPACT_QUERY = """
MATCH (n)
WHERE coalesce(n.doc_ids, []) = [$doc_id] AND NOT n:Manual AND NOT n:Section
RETURN count(n) AS entity_count
"""

ENTITY_DELETE_QUERY = """
MATCH (n)
WHERE coalesce(n.doc_ids, []) = [$doc_id]
DETACH DELETE n
RETURN count(n) AS nodes_deleted
"""

# Entities that other documents also support drop this document from doc_ids, and
# equipment_ids is rebuilt from the equipment of the documents that remain.
ENTITY_PROVENANCE_UPDATE_QUERY = """
MATCH (n)
WHERE $doc_id IN coalesce(n.doc_ids, [])
WITH n, [d IN n.doc_ids WHERE d <> $doc_id] AS remaining
OPTIONAL MATCH (e:Equipment)-[:HAS_MANUAL]->(m:Manual)
WHERE m.id IN remaining
WITH n, remaining, collect(DISTINCT e.id) AS remaining_equipment
SET n.doc_ids = remaining, n.equipment_ids = remaining_equipment
RETURN count(n) AS nodes_updated
"""


def summarize_document(doc_id: str, payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Build one document record from the payloads of its chunks."""
    first = payloads[0]
    levels = Counter(payload.get("chunk_level") for payload in payloads)
    return {
        "doc_id": doc_id,
        "oem": first.get("oem") or "Unknown",
        "model": first.get("model") or "Unknown",
        "file_name": Path(first.get("file_path") or "").name,
        "chunk_count": len(payloads),
        "chunks_by_level": {level: levels.get(level, 0) for level in CHUNK_LEVELS},
    }


class DocumentManager:
    """Documents are grouped from Qdrant chunk payloads by ``doc_id``.

    The knowledge graph is used only when GRAPH_ENABLED is true. Qdrant errors propagate
    to the caller.
    """

    def __init__(
        self,
        qdrant_manager: QdrantManager | None = None,
        neo4j_manager: Any | None = None,
        settings: Settings | None = None,
    ):
        settings = settings or get_settings()
        self.qdrant = qdrant_manager or QdrantManager(settings=settings)
        if neo4j_manager is None and settings.graph_enabled:
            from service_manual_rag.knowledge_graph import Neo4JManager

            neo4j_manager = Neo4JManager()
        self.neo4j = neo4j_manager

    def collection_info(self) -> dict[str, Any] | None:
        """Return name, vector size and point count, or None before the first ingest."""
        if not self.qdrant.collection_exists():
            return None
        return self.qdrant.get_collection_info()

    def _scroll_all(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        total = self.qdrant.count_points(filters=filters)
        if total == 0:
            return []
        return self.qdrant.scroll_points(filters=filters, limit=total)

    def get_all_documents(self) -> list[dict[str, Any]]:
        """Return one record per indexed document, sorted by OEM, model and file name."""
        if not self.qdrant.collection_exists():
            return []
        payloads_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for point in self._scroll_all():
            payload = point["payload"] or {}
            payloads_by_doc[payload.get("doc_id") or "unknown"].append(payload)
        documents = [summarize_document(doc_id, rows) for doc_id, rows in payloads_by_doc.items()]
        return sorted(documents, key=lambda d: (d["oem"], d["model"], d["file_name"]))

    def get_graph_impact(self, doc_id: str) -> dict[str, int] | None:
        """Return graph node counts tied to the document.

        Returns None when the graph is disabled or holds no Manual node for the document.
        """
        if self.neo4j is None:
            return None
        rows = self.neo4j.execute_query(GRAPH_IMPACT_QUERY, {"doc_id": doc_id})
        if not rows:
            return None
        entity_rows = self.neo4j.execute_query(ENTITY_IMPACT_QUERY, {"doc_id": doc_id})
        return {
            "sections": rows[0].get("section_count", 0),
            "components": rows[0].get("component_count", 0),
            "other_manuals": rows[0].get("other_manuals_count", 0),
            "entities": entity_rows[0].get("entity_count", 0) if entity_rows else 0,
        }

    def delete_document(self, doc_id: str) -> dict[str, int | None]:
        """Delete the document's chunks and, when the graph is enabled, its graph knowledge.

        The graph loses the Manual, its Sections and every entity extracted only from this
        document. Entities that other documents also support stay, without this document.
        """
        point_ids = [point["id"] for point in self._scroll_all({"doc_id": doc_id})]
        if point_ids:
            self.qdrant.delete_points(point_ids)

        graph_nodes_deleted = None
        if self.neo4j is not None:
            graph_nodes_deleted = sum(
                self._run_graph_write(query, doc_id, "nodes_deleted")
                for query in (GRAPH_DELETE_QUERY, ENTITY_DELETE_QUERY)
            )
            self._run_graph_write(ENTITY_PROVENANCE_UPDATE_QUERY, doc_id, "nodes_updated")
        return {"chunks_deleted": len(point_ids), "graph_nodes_deleted": graph_nodes_deleted}

    def _run_graph_write(self, query: str, doc_id: str, count_key: str) -> int:
        rows = self.neo4j.execute_query(query, {"doc_id": doc_id}, write=True)
        return rows[0].get(count_key, 0) if rows else 0
