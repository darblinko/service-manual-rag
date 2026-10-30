"""Command line entry point: ingest manuals, ask questions, run the evaluation."""

import argparse
import json
import sys
from pathlib import Path

from service_manual_rag.config import MissingCredentialError, get_settings
from service_manual_rag.embeddings import VectorIndexer
from service_manual_rag.evaluation import RAGASEvaluator, load_golden_set
from service_manual_rag.graph_builder import open_graph_populator
from service_manual_rag.processing import ChunkLevel, ProcessingPipeline
from service_manual_rag.rag import RAGPipeline
from service_manual_rag.utils import setup_logging

DEFAULT_MANUALS = Path("sample_data/manuals")
DEFAULT_GOLDEN_SET = Path("sample_data/golden_set.json")
DEFAULT_OUTPUT = Path("results/evaluation.json")
DEFAULT_QUESTION = "What is the torque for the CP-200 impeller nut?"
SUPPORTED_SUFFIXES = (".md", ".markdown", ".txt", ".pdf")


def manual_files(directory: Path) -> list[Path]:
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No manuals ({', '.join(SUPPORTED_SUFFIXES)}) in {directory}")
    return files


def ingest(manuals: Path, levels: list[int], reset: bool, graph: bool = False) -> dict:
    """Index the manuals; with graph, also extract entities from every chunk into Neo4j."""
    pipeline = ProcessingPipeline()
    indexer = VectorIndexer()
    indexer.ensure_collection(reset=reset)
    populator = open_graph_populator(reset=reset) if graph else None
    selected_levels = [ChunkLevel(level) for level in levels]
    totals = {"files": 0, "chunks": 0, "indexed": 0}
    for path in manual_files(manuals):
        _, chunks = pipeline.process(path)
        stats = indexer.index_chunks(chunks, levels_to_index=selected_levels)
        totals["files"] += 1
        totals["chunks"] += stats["total_chunks"]
        totals["indexed"] += stats["indexed_chunks"]
        print(f"{path.name}: {stats['indexed_chunks']} of {stats['total_chunks']} chunks indexed")
        if populator is not None:
            graph_stats = populator.populate(chunks)
            print(f"{path.name}: graph {graph_stats}")
    if populator is not None:
        populator.close()
    totals["embedding_model"] = indexer.embedding_service.model
    print(
        f"Indexed {totals['indexed']} chunks from {totals['files']} files "
        f"with {totals['embedding_model']}"
    )
    return totals


def ask(question: str, top_k: int | None) -> dict:
    result = RAGPipeline().query(question, top_k=top_k)
    print(f"Q: {question}\n\n{result['answer']}\n")
    for number, source in enumerate(result["sources"], start=1):
        pages = ", ".join(str(p) for p in source.get("page_numbers") or []) or "n/a"
        print(
            f"[Source {number}] {source.get('model')}, {source.get('section_name')}, page {pages}"
        )
    if result["metadata"]["graph_error"]:
        print(f"Graph search failed: {result['metadata']['graph_error']}", file=sys.stderr)
    return result


def evaluate(golden_set_path: Path, output: Path) -> dict:
    settings = get_settings()
    golden_set = load_golden_set(golden_set_path)
    pipeline = RAGPipeline()
    report = RAGASEvaluator().evaluate_pipeline(golden_set, pipeline)
    report = {
        "golden_set": golden_set_path.as_posix(),
        "embedding_model": pipeline.retriever.vector_indexer.embedding_service.model,
        "indexed_chunks": pipeline.retriever.vector_indexer.qdrant_manager.count_points(),
        "final_top_k": settings.final_top_k,
        "graph_enabled": settings.graph_enabled,
        "retrieval_mode": pipeline.retriever.effective_mode,
        **report,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Judge: {report['judge']}, generator: {report['generator']}")
    print(
        f"Retrieval: {report['retrieval_mode']}, graph search errors: {len(report['graph_errors'])}"
    )
    for metric, value in report["aggregate"].items():
        print(f"  {metric:<18} {value:.4f}")
    print(f"Failed judgments: {report['failed_judgments']}. Wrote {output.as_posix()}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="service-manual-rag", description=__doc__)
    parser.add_argument("--log-level", default="WARNING", help="Logging level (default WARNING)")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest_cmd = commands.add_parser("ingest", help="Parse, chunk and index manuals")
    ingest_cmd.add_argument("--manuals", type=Path, default=DEFAULT_MANUALS)
    ingest_cmd.add_argument("--levels", type=int, nargs="+", default=[4], choices=[1, 2, 3, 4])
    ingest_cmd.add_argument(
        "--reset", action="store_true", help="Recreate the collection (and with --graph, the graph)"
    )
    ingest_cmd.add_argument(
        "--graph",
        action="store_true",
        help="Also extract entities with OpenAI into Neo4j (needs NEO4J_PASSWORD)",
    )

    ask_cmd = commands.add_parser("ask", help="Answer a question from the indexed manuals")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument("--top-k", type=int, default=None)

    for name, help_text in (
        ("evaluate", "Score the pipeline on a golden question set"),
        ("quickstart", "Ingest the sample manuals, ask one question, run the evaluation"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET)
        command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    commands.choices["quickstart"].add_argument("--manuals", type=Path, default=DEFAULT_MANUALS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(log_level=args.log_level)
    settings = get_settings()
    print(f"Mode: {'openai' if settings.use_openai else 'offline'}", file=sys.stderr)
    try:
        if args.command == "ingest":
            ingest(args.manuals, args.levels, args.reset, args.graph)
        elif args.command == "ask":
            ask(args.question, args.top_k)
        elif args.command == "evaluate":
            evaluate(args.golden_set, args.output)
        elif args.command == "quickstart":
            ingest(args.manuals, [4], reset=True)
            print()
            ask(DEFAULT_QUESTION, None)
            print()
            evaluate(args.golden_set, args.output)
    except (MissingCredentialError, FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
