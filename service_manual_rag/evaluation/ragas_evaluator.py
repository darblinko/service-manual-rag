"""RAGAS-style evaluation of the RAG pipeline against a golden question set.

Metrics follow the RAGAS definitions (context relevance, answer relevance,
faithfulness, answer correctness) and are scored by an LLM judge or by the
offline lexical judge. The ragas package itself is not used. Retrieval is also
scored directly against the golden labels:
- section_hit: 1 when a retrieved passage comes from the expected_section of the
  expected model's manual
- reciprocal_rank: 1 / rank of the first such passage, 0 when there is none
- model_precision: share of retrieved passages from the expected model's manual
"""

from statistics import mean

from service_manual_rag.rag import RAGPipeline
from service_manual_rag.utils.logging_config import get_logger

from .judges import Judge, create_judge

logger = get_logger(__name__)

ANSWER_METRICS = ("context_relevance", "answer_relevance", "faithfulness")
RETRIEVAL_METRICS = ("section_hit", "reciprocal_rank", "model_precision")


def normalized(value: str | None) -> str:
    return (value or "").strip().lower()


def retrieval_scores(
    sources: list[dict], expected_section: str, expected_model: str | None = None
) -> dict[str, float]:
    """Score the ranked sources against the golden section and, when given, model."""

    def same_model(source: dict) -> bool:
        return expected_model is None or normalized(source.get("model")) == normalized(
            expected_model
        )

    scores = {"section_hit": 0.0, "reciprocal_rank": 0.0}
    for rank, source in enumerate(sources, start=1):
        if same_model(source) and normalized(source.get("section_name")) == normalized(
            expected_section
        ):
            scores = {"section_hit": 1.0, "reciprocal_rank": 1.0 / rank}
            break
    if expected_model is not None and sources:
        scores["model_precision"] = sum(same_model(s) for s in sources) / len(sources)
    return scores


def metric_means(rows: list[dict]) -> dict[str, float]:
    """Mean of each metric over the rows that have it, rounded to 4 places."""
    means = {}
    for metric in ANSWER_METRICS + ("correctness",) + RETRIEVAL_METRICS:
        values = [row["scores"][metric] for row in rows if metric in row["scores"]]
        if values:
            means[metric] = round(mean(values), 4)
    return means


class RAGASEvaluator:
    """Run golden questions through a pipeline and score each answer."""

    def __init__(self, judge: Judge | None = None):
        self.judge = judge or create_judge()

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> tuple[dict[str, float], list[str]]:
        """Return (scores, errors). A metric whose judgment fails is left out and its error kept."""
        metrics = ANSWER_METRICS + (("correctness",) if ground_truth else ())
        scores: dict[str, float] = {}
        errors: list[str] = []
        for metric in metrics:
            try:
                scores[metric] = self.judge.score(metric, question, answer, contexts, ground_truth)
            # A judge can fail on network, parsing or range errors; record and continue.
            except Exception as error:  # noqa: BLE001
                logger.error(f"Judge failed on {metric} for '{question[:60]}': {error}")
                errors.append(f"{metric}: {error}")
        return scores, errors

    def evaluate_pipeline(self, golden_set: list[dict], rag_pipeline: RAGPipeline) -> dict:
        """Evaluate every case and return per-question rows plus metric means."""
        rows = []
        for case in golden_set:
            result = rag_pipeline.query(case["question"])
            contexts = [source.get("content", "") for source in result["sources"]]
            scores, errors = self.evaluate_answer(
                case["question"], result["answer"], contexts, case.get("ground_truth")
            )
            if case.get("expected_section"):
                scores.update(
                    retrieval_scores(result["sources"], case["expected_section"], case.get("model"))
                )
            rows.append(
                {
                    "id": case.get("id"),
                    "category": case.get("category"),
                    "question": case["question"],
                    "answer": result["answer"],
                    "retrieved": [
                        {"model": s.get("model"), "section": s.get("section_name")}
                        for s in result["sources"]
                    ],
                    "scores": scores,
                    "errors": errors,
                }
            )

        categories = sorted({row["category"] for row in rows if row["category"]})
        report = {
            "judge": self.judge.name,
            "generator": rag_pipeline.generator.name,
            "total_questions": len(rows),
            "failed_judgments": sum(len(row["errors"]) for row in rows),
            "aggregate": metric_means(rows),
            "by_category": {
                category: metric_means([row for row in rows if row["category"] == category])
                for category in categories
            },
            "per_question": rows,
        }
        logger.info(f"Evaluation complete: {report['aggregate']}")
        return report
