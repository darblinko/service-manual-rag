"""Knowledge graph construction module.

Builds the Neo4J knowledge graph from processed chunks.
"""

from .entity_extractor import EntityExtractor
from .graph_populator import GraphPopulator, open_graph_populator

__all__ = ["EntityExtractor", "GraphPopulator", "open_graph_populator"]
