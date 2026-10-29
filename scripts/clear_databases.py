"""Delete all indexed vectors and, when the graph is enabled, all knowledge graph data.

The Qdrant collection is recreated empty, sized for the active embedder.
"""

import argparse
import sys

from neo4j.exceptions import DriverError, Neo4jError

from service_manual_rag.config import MissingCredentialError, get_settings
from service_manual_rag.embeddings import create_embedding_service
from service_manual_rag.knowledge_graph import Neo4JManager
from service_manual_rag.utils import setup_logging
from service_manual_rag.vector_db import QdrantManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Clear the Neo4j database even when GRAPH_ENABLED is false",
    )
    parser.add_argument("--log-level", default="WARNING", help="Logging level (default WARNING)")
    return parser


def confirm(targets: list[str]) -> bool:
    print("This permanently deletes all data in:")
    for target in targets:
        print(f"  {target}")
    return input("Type 'yes' to continue: ").strip().lower() == "yes"


def clear_qdrant() -> None:
    manager = QdrantManager()
    points = manager.count_points() if manager.collection_exists() else 0
    vector_size = create_embedding_service().dimensions
    manager.create_collection(vector_size=vector_size, force=True)
    print(
        f"Deleted {points} points; recreated empty collection {manager.collection_name} "
        f"with vector size {vector_size}"
    )


def clear_neo4j() -> None:
    manager = Neo4JManager()
    try:
        stats = manager.get_statistics()
        manager.clear_database(confirm=True)
        print(
            f"Deleted {stats['total_nodes']} Neo4j nodes and "
            f"{stats['total_relationships']} relationships"
        )
    finally:
        manager.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(log_level=args.log_level)
    settings = get_settings()
    include_graph = settings.graph_enabled or args.graph

    targets = [f"Qdrant collection '{settings.qdrant_collection_name}'"]
    if include_graph:
        targets.append(f"Neo4j database '{settings.neo4j_database}' at {settings.neo4j_uri}")
    if not confirm(targets):
        print("Cancelled.")
        return 1

    try:
        clear_qdrant()
        if include_graph:
            clear_neo4j()
    except (MissingCredentialError, ValueError, DriverError, Neo4jError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
