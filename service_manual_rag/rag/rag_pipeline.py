"""RAG pipeline: hybrid retrieval, rank fusion, answer generation."""

from service_manual_rag.config import get_settings
from service_manual_rag.utils.logging_config import get_logger

from .answering import AnswerGenerator, create_answer_generator
from .fusion_engine import FusionEngine
from .retriever import HybridRetriever

logger = get_logger(__name__)


class RAGPipeline:
    """Answer a question from the indexed manuals and return the passages used."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        fusion_engine: FusionEngine | None = None,
        generator: AnswerGenerator | None = None,
    ):
        self.retriever = retriever or HybridRetriever()
        self.fusion = fusion_engine or FusionEngine()
        self.generator = generator or create_answer_generator()

    def query(
        self,
        question: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> dict:
        settings = get_settings()
        top_k = top_k or settings.final_top_k
        retrieval = self.retriever.retrieve(
            question,
            vector_limit=settings.vector_top_k,
            graph_limit=settings.graph_top_k,
            filters=filters,
        )
        fused = self.fusion.fuse(retrieval["vector_results"], retrieval["graph_results"])
        top_results = fused[:top_k]
        answer = self.generator.generate(question, top_results)
        logger.info(f"Answered with {len(top_results)} of {len(fused)} fused passages")

        return {
            "question": question,
            "answer": answer,
            "sources": top_results,
            "metadata": {
                "total_retrieved": len(fused),
                "sources_used": len(top_results),
                "model": self.generator.name,
                "graph_error": retrieval.get("graph_error"),
            },
        }
