"""Score the golden set in each retrieval mode with the OpenAI pipeline and one results file.

Run after ``service-manual-rag ingest --reset --graph`` (and, for the PDF corpus,
``scripts/fetch_public_corpus.py --graph``) with LLM_MODE=openai, GRAPH_ENABLED=true and a
running Neo4j. Each mode answers every question with the chat model and scores the
answer with the same model as judge. The output records the models, the corpus files
with their SHA-256, the chunk and graph node counts, the date, the git revision and the
token usage of the answer and judge calls.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from openai import OpenAI

from service_manual_rag.config import MissingCredentialError, Settings, get_settings
from service_manual_rag.embeddings import VectorIndexer
from service_manual_rag.evaluation import RAGASEvaluator, load_golden_set
from service_manual_rag.evaluation.judges import OpenAIJudge
from service_manual_rag.rag import HybridRetriever, RAGPipeline
from service_manual_rag.rag.answering import OpenAIAnswerGenerator
from service_manual_rag.utils import setup_logging

MODES = ("vector", "graph", "fused")
DEFAULT_GOLDEN_SET = Path("sample_data/golden_set.json")


class UsageCounter:
    """Sum the token usage of every chat completion made through one client."""

    def __init__(self, client: OpenAI):
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._create = client.chat.completions.create
        client.chat.completions.create = self.create

    def create(self, **kwargs):
        response = self._create(**kwargs)
        self.calls += 1
        if response.usage is not None:
            self.prompt_tokens += response.usage.prompt_tokens
            self.completion_tokens += response.usage.completion_tokens
        return response

    def totals(self) -> dict:
        return {
            "chat_calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout

    return {
        "commit": git("rev-parse", "--short", "HEAD").strip(),
        "tracked_changes": bool(git("status", "--porcelain", "--untracked-files=no").strip()),
    }


def evaluate_mode(mode: str, settings: Settings, golden_set: list[dict]) -> dict:
    mode_settings = settings.model_copy(update={"retrieval_mode": mode})
    client = OpenAI(api_key=settings.require_openai_key())
    usage = UsageCounter(client)
    retriever = HybridRetriever(settings=mode_settings)
    pipeline = RAGPipeline(
        retriever=retriever, generator=OpenAIAnswerGenerator(mode_settings, client)
    )
    report = RAGASEvaluator(OpenAIJudge(mode_settings, client)).evaluate_pipeline(
        golden_set, pipeline
    )
    if retriever.neo4j is not None:
        retriever.neo4j.close()
    return {"retrieval_mode": retriever.effective_mode, "usage": usage.totals(), **report}


def graph_statistics(settings: Settings) -> dict:
    from service_manual_rag.knowledge_graph import Neo4JManager

    manager = Neo4JManager()
    try:
        stats = manager.get_statistics()
    finally:
        manager.close()
    return {key: value for key, value in stats.items() if value}


def build_report(settings: Settings, golden_set_path: Path, corpus: list[Path], modes) -> dict:
    golden_set = load_golden_set(golden_set_path)
    runs = {mode: evaluate_mode(mode, settings, golden_set) for mode in modes}
    return {
        "date": datetime.now(UTC).date().isoformat(),
        "git": git_revision(),
        "chat_model": settings.openai_model,
        "judge": f"openai:{settings.openai_model}",
        "embedding_model": settings.openai_embedding_model,
        "embedding_dimensions": settings.openai_embedding_dimensions,
        "corpus": [{"path": path.as_posix(), "sha256": sha256_of(path)} for path in corpus],
        "indexed_chunks": VectorIndexer().qdrant_manager.count_points(),
        "graph": graph_statistics(settings),
        "golden_set": golden_set_path.as_posix(),
        "total_questions": len(golden_set),
        "retrieval_settings": {
            "vector_top_k": settings.vector_top_k,
            "graph_top_k": settings.graph_top_k,
            "final_top_k": settings.final_top_k,
            "rrf_k": settings.rrf_k,
            "vector_weight": settings.vector_weight,
            "graph_weight": settings.graph_weight,
        },
        "runs": runs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument(
        "--corpus", type=Path, nargs="+", required=True, help="Indexed files, for the record"
    )
    parser.add_argument("--log-level", default="WARNING", help="Logging level (default WARNING)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(log_level=args.log_level)
    settings = get_settings()
    if not settings.use_openai or not settings.graph_enabled:
        print("Error: set LLM_MODE=openai and GRAPH_ENABLED=true", file=sys.stderr)
        return 1
    try:
        report = build_report(settings, args.golden_set, args.corpus, args.modes)
    except (MissingCredentialError, FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for mode, run in report["runs"].items():
        means = ", ".join(f"{metric} {value:.3f}" for metric, value in run["aggregate"].items())
        print(f"{mode}: {means}; graph errors {len(run['graph_errors'])}")
    print(f"Wrote {args.output.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
