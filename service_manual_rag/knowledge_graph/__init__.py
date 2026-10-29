"""Knowledge graph module for Neo4J integration."""

from .graph_schema import (
    NodeType,
    RelationshipType,
    create_constraints,
    create_indexes,
)
from .neo4j_client import Neo4JManager

__all__ = [
    "Neo4JManager",
    "NodeType",
    "RelationshipType",
    "create_constraints",
    "create_indexes",
]
