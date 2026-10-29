"""Knowledge graph construction module.

Builds the Neo4J knowledge graph from processed chunks.
"""

from .entity_extractor import EntityExtractor
from .graph_populator import GraphPopulator

__all__ = ["EntityExtractor", "GraphPopulator"]
