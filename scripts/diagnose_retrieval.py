"""Report the state of the vector store and knowledge graph and run sample retrievals."""

import argparse
import sys
from collections import Counter
from collections.abc import Callable

from service_manual_rag.config import MissingCredentialError, get_settings
from service_manual_rag.embeddings import VectorIndexer
from service_manual_rag.knowledge_graph import Neo4JManager
from service_manual_rag.rag import HybridRetriever, RAGPipeline
from service_manual_rag.rag.retriever import GRAPH_QUERY
from service_manual_rag.utils import setup_logging

DEFAULT_QUERY = "main components"
LOW_VECTOR_SCORE = 0.3
SAMPLE_SIZE = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Query used by the search checks")
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Check the Neo4j graph even when GRAPH_ENABLED is false",
    )
    parser.add_argument("--log-level", default="WARNING", help="Logging level (default WARNING)")
    return parser


def print_header(title: str) -> None:
    print(f"\n{title}\n{'=' * len(title)}")


def check_vector_store(indexer: VectorIndexer) -> bool:
    manager = indexer.qdrant_manager
    if not manager.collection_exists():
        print(f"Collection '{manager.collection_name}' does not exist. Run ingestion first.")
        return False
    info = manager.get_collection_info()
    print(
        f"Collection {info['name']}: vector size {info['vector_size']}, "
        f"{info['points_count']} points, status {info['status']}"
    )
    if info["points_count"] == 0:
        print("No vectors are indexed. Run ingestion first.")
        return False
    if info["vector_size"] != indexer.embedding_service.dimensions:
        print(
            f"The embedder ({indexer.embedding_service.model}) produces "
            f"{indexer.embedding_service.dimensions} dimensions. Re-run ingestion with --reset."
        )
        return False
    return True


def check_vector_search(indexer: VectorIndexer, query: str) -> bool:
    results = indexer.search(query, limit=SAMPLE_SIZE)
    if not results:
        print("Vector search returned no results.")
        return False
    for number, result in enumerate(results, start=1):
        text = (result["payload"] or {}).get("chunk_text", "")
        print(f"{number}. score {result['score']:.4f}: {text[:100]}")
    top_score = results[0]["score"]
    if top_score < LOW_VECTOR_SCORE:
        print(
            f"Top score {top_score:.4f} is below {LOW_VECTOR_SCORE}. The query may not match "
            "the indexed manuals, or the embedder changed since ingestion."
        )
    return True


def check_graph(manager: Neo4JManager) -> bool:
    if not manager.health_check():
        print(f"Neo4j is not reachable at {manager.uri}.")
        return False
    stats = manager.get_statistics()
    print(f"{stats['total_nodes']} nodes, {stats['total_relationships']} relationships")
    for key, value in sorted(stats.items()):
        if key.endswith("_count"):
            print(f"  {key.removesuffix('_count')}: {value}")
    if stats["total_nodes"] == 0:
        print("The graph is empty. Populate it before using graph search.")
        return False
    return True


def check_graph_search(manager: Neo4JManager, query: str) -> bool:
    records = manager.execute_query(GRAPH_QUERY, {"query": query, "limit": SAMPLE_SIZE})
    if not records:
        print("Full-text graph search returned no results.")
        return False
    for number, record in enumerate(records, start=1):
        node = dict(record["node"])
        print(f"{number}. score {record['score']:.4f}: {record['types']} {node.get('id')}")
    return True


def check_hybrid(query: str, manager: Neo4JManager | None) -> bool:
    retriever = HybridRetriever(neo4j_manager=manager)
    result = RAGPipeline(retriever=retriever).query(query, top_k=10)
    sources = result["sources"]
    counts = Counter(source["source_type"] for source in sources)
    print(f"{len(sources)} fused sources by origin: {dict(counts)}")
    for number, source in enumerate(sources[:SAMPLE_SIZE], start=1):
        print(f"{number}. RRF score {source['rrf_score']:.4f} ({source['source_type']})")
    if result["metadata"]["graph_error"]:
        print(f"Graph search failed: {result['metadata']['graph_error']}")
        return False
    return bool(sources)


def run_check(title: str, check: Callable[..., bool], *args) -> bool:
    """Run one check, reporting an exception as a failed check."""
    print_header(title)
    try:
        return check(*args)
    except Exception as error:  # noqa: BLE001 (any failure is reported as a failed check)
        print(f"Failed: {error}")
        return False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(log_level=args.log_level)
    include_graph = get_settings().graph_enabled or args.graph
    try:
        manager = Neo4JManager() if include_graph else None
    except MissingCredentialError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    indexer = VectorIndexer()
    results = {"Vector store": run_check("Vector store", check_vector_store, indexer)}
    if results["Vector store"]:
        results["Vector search"] = run_check(
            "Vector search", check_vector_search, indexer, args.query
        )

    try:
        if manager is not None:
            results["Graph"] = run_check("Knowledge graph", check_graph, manager)
            if results["Graph"]:
                results["Graph search"] = run_check(
                    "Graph search", check_graph_search, manager, args.query
                )
        if results["Vector store"]:
            results["Hybrid RAG"] = run_check("Hybrid RAG", check_hybrid, args.query, manager)
    finally:
        if manager is not None:
            manager.close()

    print_header("Summary")
    for name, passed in results.items():
        print(f"{name:<14} {'OK' if passed else 'ISSUE'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
