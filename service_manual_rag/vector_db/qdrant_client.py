"""Qdrant vector database operations, in server mode or embedded local mode."""

import atexit
from functools import lru_cache
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct

from service_manual_rag.config import Settings, get_settings
from service_manual_rag.utils.logging_config import get_logger

from .schema import create_collection_config, create_payload_index_config, get_search_params

logger = get_logger(__name__)


@lru_cache
def _cached_client(mode: str, local_path: str, host: str, port: int) -> QdrantClient:
    if mode == "local":
        Path(local_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Opening embedded Qdrant storage at {local_path}")
        client = QdrantClient(path=local_path)
        # Release the storage lock before interpreter shutdown; the client's own
        # finalizer fails if it first runs during shutdown.
        atexit.register(client.close)
        return client
    logger.info(f"Connecting to Qdrant server at {host}:{port}")
    return QdrantClient(host=host, port=port)


def get_qdrant_client(settings: Settings | None = None) -> QdrantClient:
    """Return one shared client per configuration.

    Embedded local storage allows a single client per process, so every
    component that needs Qdrant should obtain its client here.
    """
    settings = settings or get_settings()
    return _cached_client(
        settings.qdrant_mode,
        str(settings.qdrant_local_path),
        settings.qdrant_host,
        settings.qdrant_port,
    )


def build_filter(filters: dict[str, Any] | None) -> Filter | None:
    """Translate {"field": value} pairs into a Qdrant must-match filter."""
    if not filters:
        return None
    conditions = [
        FieldCondition(key=key, match=MatchValue(value=value)) for key, value in filters.items()
    ]
    return Filter(must=conditions)


class QdrantManager:
    """Collection, upsert and search operations for one collection."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        collection_name: str | None = None,
        settings: Settings | None = None,
    ):
        settings = settings or get_settings()
        self.client = client or get_qdrant_client(settings)
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.is_local = client is None and settings.qdrant_mode == "local"

    def health_check(self) -> bool:
        """Return True when the collections endpoint answers."""
        try:
            self.client.get_collections()
            return True
        except Exception as error:  # noqa: BLE001
            logger.error(f"Qdrant health check failed: {error}")
            return False

    def collection_exists(self, collection_name: str | None = None) -> bool:
        name = collection_name or self.collection_name
        return any(c.name == name for c in self.client.get_collections().collections)

    def create_collection(
        self,
        vector_size: int,
        collection_name: str | None = None,
        distance: Distance = Distance.COSINE,
        force: bool = False,
    ) -> None:
        """Create the collection and its payload indexes; recreate it when force is True."""
        name = collection_name or self.collection_name
        if force and self.collection_exists(name):
            logger.warning(f"Deleting existing collection: {name}")
            self.client.delete_collection(name)

        if self.collection_exists(name):
            existing_size = self.vector_size(name)
            if existing_size != vector_size:
                raise ValueError(
                    f"Collection '{name}' has vector size {existing_size}, expected "
                    f"{vector_size}. Re-run ingestion with --reset."
                )
            return

        config = create_collection_config(vector_size, distance)
        self.client.create_collection(
            collection_name=name,
            vectors_config=config["vectors_config"],
            hnsw_config=config["hnsw_config"],
            optimizers_config=config["optimizer_config"],
        )
        # Embedded local mode ignores payload indexes and warns on each call.
        if not self.is_local:
            for index_config in create_payload_index_config():
                self.client.create_payload_index(
                    collection_name=name,
                    field_name=index_config["field_name"],
                    field_schema=index_config["field_schema"],
                )
        logger.info(f"Created collection {name} with vector size {vector_size}")

    def vector_size(self, collection_name: str | None = None) -> int:
        info = self.client.get_collection(collection_name or self.collection_name)
        return info.config.params.vectors.size

    def delete_collection(self, collection_name: str | None = None) -> None:
        name = collection_name or self.collection_name
        if self.collection_exists(name):
            self.client.delete_collection(name)
            logger.info(f"Deleted collection: {name}")

    def get_collection_info(self, collection_name: str | None = None) -> dict[str, Any]:
        name = collection_name or self.collection_name
        info = self.client.get_collection(name)
        return {
            "name": name,
            "vector_size": info.config.params.vectors.size,
            "points_count": info.points_count or 0,
            "status": str(info.status),
        }

    def upsert_points(
        self,
        points: list[dict[str, Any]],
        collection_name: str | None = None,
        batch_size: int = 100,
    ) -> int:
        """Upsert points given as {"id", "vector", "payload"} dicts and return the count."""
        name = collection_name or self.collection_name
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            structs = [
                PointStruct(id=p["id"], vector=p["vector"], payload=p.get("payload", {}))
                for p in batch
            ]
            self.client.upsert(collection_name=name, points=structs)
        logger.info(f"Upserted {len(points)} points to '{name}'")
        return len(points)

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        collection_name: str | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return the nearest points as {"id", "score", "payload"} dicts."""
        results = self.client.query_points(
            collection_name=collection_name or self.collection_name,
            query=query_vector,
            limit=limit,
            query_filter=build_filter(filters),
            score_threshold=score_threshold,
            search_params=None if self.is_local else get_search_params(),
        ).points
        return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results]

    def delete_points(self, point_ids: list[str], collection_name: str | None = None) -> None:
        name = collection_name or self.collection_name
        self.client.delete(collection_name=name, points_selector=point_ids)
        logger.info(f"Deleted {len(point_ids)} points from '{name}'")

    def get_point(self, point_id: str, collection_name: str | None = None) -> dict[str, Any] | None:
        result = self.client.retrieve(
            collection_name=collection_name or self.collection_name, ids=[point_id]
        )
        if not result:
            return None
        return {"id": result[0].id, "vector": result[0].vector, "payload": result[0].payload}

    def count_points(
        self, collection_name: str | None = None, filters: dict[str, Any] | None = None
    ) -> int:
        result = self.client.count(
            collection_name=collection_name or self.collection_name,
            count_filter=build_filter(filters),
            exact=True,
        )
        return result.count

    def scroll_points(
        self,
        collection_name: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        with_vectors: bool = False,
    ) -> list[dict[str, Any]]:
        results, _ = self.client.scroll(
            collection_name=collection_name or self.collection_name,
            scroll_filter=build_filter(filters),
            limit=limit,
            with_vectors=with_vectors,
            with_payload=True,
        )
        return [
            {"id": r.id, "payload": r.payload, "vector": r.vector if with_vectors else None}
            for r in results
        ]
