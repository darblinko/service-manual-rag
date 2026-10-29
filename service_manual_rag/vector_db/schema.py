"""Qdrant vector database schema definitions."""

from dataclasses import dataclass, field
from typing import Any

from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    VectorParams,
)

from service_manual_rag.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class VectorSchema:
    """Schema for vector database payload.

    Defines the structure of metadata stored alongside vectors in Qdrant.
    """

    # Document identifiers
    oem: str
    model: str
    chunk_id: str

    # Chunk hierarchy
    chunk_level: int  # 1-4 (Document, Section, Subsection, Semantic)
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)

    # Content metadata
    section_name: str | None = None
    page_numbers: list[int] = field(default_factory=list)
    chunk_text: str = ""

    # File metadata
    file_path: str = ""
    doc_id: str | None = None

    # Additional metadata
    element_type: str = "text"  # text, table, image, procedure
    table_data: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        """Convert schema to Qdrant payload format.

        Returns:
            dict: Payload dictionary for Qdrant
        """
        return {
            "oem": self.oem,
            "model": self.model,
            "chunk_id": self.chunk_id,
            "chunk_level": self.chunk_level,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "section_name": self.section_name,
            "page_numbers": self.page_numbers,
            "chunk_text": self.chunk_text,
            "file_path": self.file_path,
            "doc_id": self.doc_id,
            "element_type": self.element_type,
            "table_data": self.table_data,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "VectorSchema":
        """Create schema from Qdrant payload.

        Args:
            payload: Qdrant payload dictionary

        Returns:
            VectorSchema: Schema instance
        """
        return cls(
            oem=payload["oem"],
            model=payload["model"],
            chunk_id=payload["chunk_id"],
            chunk_level=payload["chunk_level"],
            parent_id=payload.get("parent_id"),
            children_ids=payload.get("children_ids", []),
            section_name=payload.get("section_name"),
            page_numbers=payload.get("page_numbers", []),
            chunk_text=payload.get("chunk_text", ""),
            file_path=payload.get("file_path", ""),
            doc_id=payload.get("doc_id"),
            element_type=payload.get("element_type", "text"),
            table_data=payload.get("table_data"),
        )


def create_collection_config(
    vector_size: int,
    distance: Distance = Distance.COSINE,
) -> dict[str, Any]:
    """Create Qdrant collection configuration.

    Args:
        vector_size: Size of vectors, which must match the embedding service
        distance: Distance metric for similarity search

    Returns:
        dict: Collection configuration parameters
    """
    # Vector configuration
    vectors_config = VectorParams(
        size=vector_size,
        distance=distance,
    )

    # HNSW index configuration
    # Optimized for accuracy and recall
    hnsw_config = HnswConfigDiff(
        m=16,  # Number of edges per node (higher = better recall, more memory)
        ef_construct=100,  # Construction time parameter (higher = better quality)
        full_scan_threshold=10000,  # Switch to full scan for small collections
    )

    # Optimizer configuration
    optimizer_config = OptimizersConfigDiff(
        default_segment_number=2,
        memmap_threshold=20000,
        indexing_threshold=20000,
    )

    config = {
        "vectors_config": vectors_config,
        "hnsw_config": hnsw_config,
        "optimizer_config": optimizer_config,
    }

    logger.debug(f"Collection config: {config}")
    return config


def get_search_params(
    ef: int = 128,
    exact: bool = False,
) -> dict[str, Any]:
    """Get search parameters for Qdrant queries.

    Args:
        ef: Size of the dynamic candidate list (higher = better accuracy, slower)
        exact: Whether to use exact search (bypasses HNSW index)

    Returns:
        dict: Search parameters
    """
    return {
        "hnsw_ef": ef,
        "exact": exact,
    }


def create_payload_index_config() -> list[dict[str, Any]]:
    """Create payload index configuration for efficient filtering.

    Returns:
        list: List of payload index configurations
    """
    # Index frequently filtered fields
    indexes = [
        {"field_name": "oem", "field_schema": "keyword"},
        {"field_name": "model", "field_schema": "keyword"},
        {"field_name": "chunk_level", "field_schema": "integer"},
        {"field_name": "section_name", "field_schema": "keyword"},
        {"field_name": "element_type", "field_schema": "keyword"},
        {"field_name": "doc_id", "field_schema": "keyword"},
    ]

    logger.debug(f"Creating {len(indexes)} payload indexes")
    return indexes


# Constants for chunk levels
CHUNK_LEVEL_DOCUMENT = 1
CHUNK_LEVEL_SECTION = 2
CHUNK_LEVEL_SUBSECTION = 3
CHUNK_LEVEL_SEMANTIC = 4

# Element types
ELEMENT_TYPE_TEXT = "text"
ELEMENT_TYPE_TABLE = "table"
ELEMENT_TYPE_IMAGE = "image"
ELEMENT_TYPE_PROCEDURE = "procedure"
