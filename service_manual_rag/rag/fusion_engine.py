"""Weighted Reciprocal Rank Fusion (RRF) of vector and graph results."""

from service_manual_rag.config import get_settings
from service_manual_rag.utils.logging_config import get_logger

logger = get_logger(__name__)

PAYLOAD_FIELDS = ("section_name", "page_numbers", "oem", "model", "doc_id", "chunk_level")


class FusionEngine:
    """Score each document as the sum of weight / (k + rank) over the lists it appears in."""

    def __init__(self, k: int | None = None):
        """
        Args:
            k: RRF constant. The common default is 60; this project defaults to
               RRF_K=10 because each list holds 10 results or fewer.
        """
        self.k = k if k is not None else get_settings().rrf_k

    def fuse(
        self,
        vector_results: list[dict],
        graph_results: list[dict],
        vector_weight: float | None = None,
        graph_weight: float | None = None,
    ) -> list[dict]:
        settings = get_settings()
        vector_weight = settings.vector_weight if vector_weight is None else vector_weight
        graph_weight = settings.graph_weight if graph_weight is None else graph_weight

        scores: dict[str, float] = {}
        entries: dict[str, dict] = {}

        for rank, result in enumerate(vector_results, start=1):
            payload = result.get("payload") or {}
            doc_id = str(result.get("id") or payload.get("chunk_id"))
            scores[doc_id] = scores.get(doc_id, 0.0) + vector_weight / (self.k + rank)
            entry = entries.setdefault(
                doc_id,
                {"content": payload.get("chunk_text", ""), "sources": []},
            )
            entry.update({field: payload.get(field) for field in PAYLOAD_FIELDS})
            entry.update({"vector_rank": rank, "vector_score": result.get("score", 0)})
            entry["sources"].append("vector")

        for rank, result in enumerate(graph_results, start=1):
            node = result.get("node", {})
            doc_id = node.get("id")
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + graph_weight / (self.k + rank)
            entry = entries.setdefault(doc_id, {"content": str(node), "sources": []})
            entry.update(
                {
                    "graph_rank": rank,
                    "graph_score": result.get("score", 0),
                    "types": result.get("types", []),
                }
            )
            entry["sources"].append("graph")

        ranked = []
        for doc_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
            sources = entries[doc_id]["sources"]
            source_type = "hybrid" if len(sources) > 1 else sources[0]
            ranked.append(
                {"id": doc_id, "rrf_score": score, "source_type": source_type, **entries[doc_id]}
            )

        logger.info(
            f"Fused {len(vector_results)} vector and {len(graph_results)} graph results "
            f"into {len(ranked)}"
        )
        return ranked
