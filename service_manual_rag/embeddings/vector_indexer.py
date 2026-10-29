"""Embed chunks and index them into Qdrant."""

from service_manual_rag.processing.hierarchical_chunker import Chunk, ChunkLevel
from service_manual_rag.utils.logging_config import get_logger
from service_manual_rag.vector_db import QdrantManager

from .embedding_service import Embedder, create_embedding_service

logger = get_logger(__name__)


def chunk_payload(chunk: Chunk) -> dict:
    """Qdrant payload stored with each chunk vector."""
    return {
        "oem": chunk.oem,
        "model": chunk.model,
        "chunk_id": chunk.chunk_id,
        "chunk_level": int(chunk.level),
        "parent_id": chunk.parent_id,
        "children_ids": chunk.children_ids,
        "section_name": chunk.section_name,
        "page_numbers": chunk.page_numbers,
        "chunk_text": chunk.content,
        "file_path": chunk.file_path,
        "doc_id": chunk.doc_id,
        "element_type": chunk.element_type,
        "table_data": chunk.table_data,
    }


class VectorIndexer:
    """Coordinates the embedding service and the Qdrant collection."""

    def __init__(
        self,
        embedding_service: Embedder | None = None,
        qdrant_manager: QdrantManager | None = None,
    ):
        self.embedding_service = embedding_service or create_embedding_service()
        self.qdrant_manager = qdrant_manager or QdrantManager()

    def ensure_collection(self, reset: bool = False) -> None:
        """Create the collection sized for the active embedder."""
        self.qdrant_manager.create_collection(
            vector_size=self.embedding_service.dimensions, force=reset
        )

    def index_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int = 100,
        levels_to_index: list[ChunkLevel] | None = None,
    ) -> dict:
        """Embed and upsert chunks, optionally limited to some hierarchy levels."""
        selected = [c for c in chunks if not levels_to_index or c.level in levels_to_index]
        self.ensure_collection()

        embeddings = self.embedding_service.embed_batch(
            [chunk.content for chunk in selected], batch_size=batch_size
        )
        points = [
            {"id": chunk.chunk_id, "vector": vector, "payload": chunk_payload(chunk)}
            for chunk, vector in zip(selected, embeddings)
        ]
        upserted = self.qdrant_manager.upsert_points(points, batch_size=batch_size)

        logger.info(f"Indexed {upserted} of {len(chunks)} chunks")
        return {
            "total_chunks": len(chunks),
            "indexed_chunks": upserted,
            "embedding_model": self.embedding_service.model,
        }

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict | None = None,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """Return the chunks nearest to the query text."""
        return self.qdrant_manager.search(
            query_vector=self.embedding_service.embed(query),
            limit=limit,
            filters=filters,
            score_threshold=score_threshold,
        )

    def get_statistics(self) -> dict:
        return {
            "qdrant": self.qdrant_manager.get_collection_info(),
            "embeddings": {
                "model": self.embedding_service.model,
                "cache": self.embedding_service.get_cache_stats(),
            },
        }

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        self.qdrant_manager.delete_points(chunk_ids)
