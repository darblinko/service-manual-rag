"""Create the Qdrant collection and, when the graph is enabled, the Neo4j schema."""

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
        help="Set up the Neo4j schema even when GRAPH_ENABLED is false",
    )
    parser.add_argument(
        "--reset", action="store_true", help="Delete and recreate the Qdrant collection"
    )
    parser.add_argument("--log-level", default="WARNING", help="Logging level (default WARNING)")
    return parser


def setup_qdrant(reset: bool) -> bool:
    manager = QdrantManager()
    if not manager.health_check():
        print(
            "Qdrant is not reachable. With QDRANT_MODE=server, start it with "
            "'docker compose up -d qdrant'.",
            file=sys.stderr,
        )
        return False
    manager.create_collection(vector_size=create_embedding_service().dimensions, force=reset)
    info = manager.get_collection_info()
    print(
        f"Qdrant collection {info['name']}: vector size {info['vector_size']}, "
        f"{info['points_count']} points, status {info['status']}"
    )
    return True


def setup_neo4j() -> bool:
    manager = Neo4JManager()
    try:
        if not manager.health_check():
            print(
                f"Neo4j is not reachable at {manager.uri}. Set NEO4J_PASSWORD as shown in "
                ".env.example and start it with 'docker compose up -d neo4j'.",
                file=sys.stderr,
            )
            return False
        manager.initialize_schema()
        stats = manager.get_statistics()
        print(
            f"Neo4j schema ready: {stats['total_nodes']} nodes, "
            f"{stats['total_relationships']} relationships"
        )
        return True
    finally:
        manager.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(log_level=args.log_level)
    settings = get_settings()
    try:
        qdrant_ok = setup_qdrant(args.reset)
        graph_ok = setup_neo4j() if settings.graph_enabled or args.graph else True
    except (MissingCredentialError, ValueError, DriverError, Neo4jError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0 if qdrant_ok and graph_ok else 1


if __name__ == "__main__":
    sys.exit(main())
