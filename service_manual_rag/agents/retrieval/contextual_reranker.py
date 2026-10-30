"""Contextual re-ranking of retrieval results.

Uses a sentence-transformers cross-encoder when that package is installed and the
model loads, and keyword and context scoring otherwise.
"""

from typing import Any

from service_manual_rag.agents.retrieval.query_rewriter import format_equipment_context
from service_manual_rag.config import get_settings
from service_manual_rag.utils.logging_config import get_logger

logger = get_logger(__name__)

MAX_CONTENT_CHARS = 1000


class ContextualReranker:
    """Re-rank results using conversation context.

    The cross-encoder scores query and document pairs where the query includes the
    equipment, symptoms and conversation summary from the context.
    """

    def __init__(
        self,
        model_name: str | None = None,
        use_cross_encoder: bool = True,
    ):
        """Initialize the re-ranker.

        Loading the cross-encoder may download the model from the Hugging Face hub.
        If sentence-transformers is not installed or the model does not load, the
        re-ranker logs a warning, sets use_cross_encoder to False and uses scoring.

        Args:
            model_name: Cross-encoder model name; defaults to ``settings.reranker_model``.
            use_cross_encoder: Whether to try the cross-encoder.
        """
        model_name = model_name or get_settings().reranker_model
        self.model_name = model_name
        self.model = self._load_cross_encoder(model_name) if use_cross_encoder else None
        self.use_cross_encoder = self.model is not None

        mode = f"cross-encoder {model_name}" if self.use_cross_encoder else "score-based ranking"
        logger.info(f"ContextualReranker initialized: {mode}")

    @staticmethod
    def _load_cross_encoder(model_name: str) -> Any | None:
        """Return a loaded CrossEncoder, or None when it is unavailable."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            logger.warning(
                "sentence-transformers is not installed; using score-based ranking. "
                "Install it with: pip install sentence-transformers"
            )
            return None

        try:
            return CrossEncoder(model_name)
        except Exception as e:
            logger.warning(
                f"Cross-encoder model {model_name} did not load "
                f"({type(e).__name__}: {e}); using score-based ranking",
                exc_info=True,
            )
            return None

    def rerank(
        self,
        query: str,
        results: list[dict],
        context: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        """Re-rank results with the cross-encoder or with context scoring.

        The input dictionaries are not modified; each returned result is a copy with
        a cross_encoder_score or enhanced_score and reranked=True.

        Args:
            query: Original query.
            results: Retrieval results.
            context: Optional conversation context.
            top_k: Return the top K results. None returns all.

        Returns:
            Re-ranked results.

        Raises:
            Exception: Cross-encoder prediction failed. The error is logged first.

        Example:
            >>> reranker = ContextualReranker(use_cross_encoder=False)
            >>> context = {
            ...     "equipment_oem": "ExampleCo",
            ...     "equipment_model": "RSC-15",
            ...     "symptoms": ["high discharge temperature"],
            ...     "conversation_summary": "Compressor trips on temperature after 20 minutes",
            ... }
            >>> reranked = reranker.rerank("compressor overheating shutdown", results, context)
        """
        if not results:
            return results

        logger.info(f"Re-ranking {len(results)} results")

        if self.use_cross_encoder:
            return self._rerank_with_cross_encoder(query, results, context, top_k)
        return self._rerank_with_scoring(query, results, context, top_k)

    def _rerank_with_cross_encoder(
        self,
        query: str,
        results: list[dict],
        context: dict[str, Any] | None,
        top_k: int | None,
    ) -> list[dict]:
        """Re-rank using the cross-encoder model.

        Returns:
            Re-ranked copies of the results with cross_encoder_score added.
        """
        rich_query = self._build_contextual_query(query, context or {})
        pairs = [(rich_query, self._extract_content(result)) for result in results]

        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            logger.error(
                f"Cross-encoder prediction failed for {len(pairs)} pairs: {type(e).__name__}: {e}"
            )
            raise

        scored = [
            {**result, "cross_encoder_score": float(score), "reranked": True}
            for result, score in zip(results, scores, strict=True)
        ]
        reranked = sorted(scored, key=lambda x: x["cross_encoder_score"], reverse=True)

        logger.info("Re-ranked with cross-encoder")
        return reranked[:top_k] if top_k else reranked

    def _rerank_with_scoring(
        self,
        query: str,
        results: list[dict],
        context: dict[str, Any] | None,
        top_k: int | None,
    ) -> list[dict]:
        """Re-rank by boosting the retrieval score with keyword and context matches.

        Returns:
            Re-ranked copies of the results with enhanced_score added.
        """
        context = context or {}
        query_words = set(query.lower().split())

        scored = []
        for result in results:
            content = self._extract_content(result).lower()
            base_score = result.get("score", 0) or result.get("rrf_score", 0)
            enhanced_score = float(base_score)

            content_words = set(content.split())
            keyword_overlap = len(query_words & content_words) / max(len(query_words), 1)
            enhanced_score += keyword_overlap * 0.2

            if context.get("equipment_oem") and context.get("equipment_model"):
                oem = context["equipment_oem"].lower()
                model = context["equipment_model"].lower()
                if oem in content or model in content:
                    enhanced_score += 0.15

            symptoms = context.get("symptoms")
            if isinstance(symptoms, list):
                for symptom in symptoms:
                    if symptom.lower() in content:
                        enhanced_score += 0.1
            elif isinstance(symptoms, str) and symptoms.lower() in content:
                enhanced_score += 0.1

            scored.append({**result, "enhanced_score": enhanced_score, "reranked": True})

        reranked = sorted(scored, key=lambda x: x["enhanced_score"], reverse=True)

        logger.info("Re-ranked with context scoring")
        return reranked[:top_k] if top_k else reranked

    def _build_contextual_query(self, query: str, context: dict[str, Any]) -> str:
        """Build the cross-encoder query from the context and the original query."""
        parts = format_equipment_context(context)
        if context.get("conversation_summary"):
            parts.append(f"Context: {context['conversation_summary']}")
        parts.append(f"Query: {query}")
        return "\n".join(parts)

    def _extract_content(self, result: dict) -> str:
        """Return the result text, truncated to MAX_CONTENT_CHARS."""
        content = (
            result.get("content")
            or result.get("text")
            or result.get("chunk_text")
            or str(result.get("node", {}))
        )
        return content[:MAX_CONTENT_CHARS]

    def batch_rerank(
        self,
        queries: list[str],
        results_list: list[list[dict]],
        contexts: list[dict[str, Any]] | None = None,
        top_k: int | None = None,
    ) -> list[list[dict]]:
        """Re-rank several result sets.

        Args:
            queries: Queries, one per result set.
            results_list: Result lists, one per query.
            contexts: Optional contexts, one per query.
            top_k: Return the top K results per query.

        Returns:
            Re-ranked result lists.
        """
        if contexts is None:
            contexts = [None] * len(queries)

        return [
            self.rerank(query, results, context, top_k)
            for query, results, context in zip(queries, results_list, contexts, strict=True)
        ]
