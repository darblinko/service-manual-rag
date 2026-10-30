"""RAG pipeline with query rewriting, re-ranking and conversation memory.

Extends hybrid retrieval and rank fusion with three optional steps:
- query rewriting into several variations,
- contextual re-ranking of the fused results,
- retrieval of similar past troubleshooting sessions from conversation memory.

Answers come from an OpenAI chat model when one is configured. In offline mode
the answer lists the highest ranked manual excerpts without generation.
"""

from typing import Any

from openai import OpenAI

from service_manual_rag.agents.memory import SemanticMemory
from service_manual_rag.agents.retrieval import ContextualReranker, QueryRewriter
from service_manual_rag.agents.retrieval.query_rewriter import format_equipment_context
from service_manual_rag.config import get_settings
from service_manual_rag.utils.logging_config import get_logger

from .fusion_engine import FusionEngine
from .retriever import HybridRetriever

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a technical assistant for industrial equipment service and maintenance.
Use the provided context from service manuals and past resolved cases to answer accurately.
Include specific part numbers, procedures and safety warnings when the context contains them.
If past similar cases are provided, mention the solutions that worked before.
If the context does not contain enough information, say so."""

ANSWER_INSTRUCTIONS = """Provide a technical answer based on the context above. Include:
1. Direct answer to the question
2. Relevant part numbers if applicable
3. Step-by-step procedures if applicable
4. Safety warnings if applicable
5. Source citations (reference the context items)
6. If similar past cases are shown, the solutions that worked before"""

OFFLINE_EXCERPT_CHARS = 400


class EnhancedRAGPipeline:
    """RAG pipeline with query rewriting, re-ranking and conversation memory.

    Each optional step has a flag. A component is constructed only when its flag is
    set and no instance is passed in.
    """

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        fusion_engine: FusionEngine | None = None,
        query_rewriter: QueryRewriter | None = None,
        reranker: ContextualReranker | None = None,
        semantic_memory: SemanticMemory | None = None,
        api_key: str | None = None,
        model: str | None = None,
        use_query_rewriting: bool = True,
        use_reranking: bool = True,
        use_conversation_memory: bool = True,
    ):
        """Initialize the pipeline.

        Args:
            retriever: Hybrid retriever. Defaults to HybridRetriever().
            fusion_engine: Fusion engine. Defaults to FusionEngine().
            query_rewriter: Query rewriter. Built when use_query_rewriting is set.
            reranker: Contextual re-ranker. Built when use_reranking is set.
            semantic_memory: Conversation memory. Built when use_conversation_memory is set.
            api_key: OpenAI API key. Defaults to the configured key.
            model: OpenAI chat model name. Defaults to settings.openai_model.
            use_query_rewriting: Enable query rewriting.
            use_reranking: Enable re-ranking.
            use_conversation_memory: Enable retrieval of similar past sessions.

        Raises:
            MissingCredentialError: OpenAI mode is selected and no key is configured.
        """
        settings = get_settings()
        self.model = model or settings.openai_model
        self.client: OpenAI | None = None
        if api_key or settings.use_openai:
            self.client = OpenAI(api_key=settings.require_openai_key(api_key))

        self.use_query_rewriting = use_query_rewriting
        self.use_reranking = use_reranking
        self.use_conversation_memory = use_conversation_memory

        self.retriever = retriever or HybridRetriever()
        self.fusion = fusion_engine or FusionEngine()
        self.query_rewriter = query_rewriter or (
            QueryRewriter(api_key=api_key) if use_query_rewriting else None
        )
        self.reranker = reranker or (ContextualReranker() if use_reranking else None)
        self.semantic_memory = semantic_memory or (
            SemanticMemory() if use_conversation_memory else None
        )

        logger.info(
            f"EnhancedRAGPipeline initialized: model={self.model if self.client else 'offline'}, "
            f"rewriting={use_query_rewriting}, reranking={use_reranking}, "
            f"memory={use_conversation_memory}"
        )

    def query(
        self,
        question: str,
        top_k: int = 5,
        filters: dict | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict:
        """Run retrieval, the optional steps and answer generation.

        A failure in an optional step is logged, recorded in
        metadata["features"]["errors"] under the step name, and the step is skipped.
        Retrieval and answer generation errors propagate.

        Args:
            question: User question.
            top_k: Number of top results passed to answer generation.
            filters: Metadata filters for vector search, for example {"oem": "ExampleCo"}.
            context: Optional context (equipment_oem, equipment_model, symptoms,
                conversation_summary).

        Returns:
            Dict with question, answer, sources, conversation_memory and metadata.

        Example:
            >>> rag = EnhancedRAGPipeline()
            >>> context = {
            ...     "equipment_oem": "ExampleCo",
            ...     "equipment_model": "CP-200",
            ...     "symptoms": ["mechanical seal leak"],
            ...     "conversation_summary": "User reported a leak at the seal gland",
            ... }
            >>> result = rag.query("How do I replace the mechanical seal?", context=context)
        """
        logger.info(f"Enhanced RAG query: {question[:50]}")

        features: dict[str, Any] = {
            "query_rewriting_used": False,
            "reranking_used": False,
            "conversation_memory_used": False,
            "errors": {},
        }

        queries = self._rewrite_queries(question, context, features)
        all_results = self._retrieve_and_fuse(queries, filters)
        conversation_results = self._retrieve_past_cases(question, context, features)
        all_results = self._rerank(question, all_results, context, features)

        top_results = all_results[:top_k]
        answer = self._generate_answer(
            question=question,
            context_results=top_results,
            conversation_results=conversation_results,
            context=context,
        )

        return {
            "question": question,
            "answer": answer["answer"],
            "sources": top_results,
            "conversation_memory": conversation_results,
            "metadata": {
                "total_retrieved": len(all_results),
                "sources_used": len(top_results),
                "model": self.model if self.client else None,
                "offline": self.client is None,
                "features": features,
            },
        }

    def _rewrite_queries(
        self,
        question: str,
        context: dict[str, Any] | None,
        features: dict[str, Any],
    ) -> list[str]:
        """Return the query variations, or [question] when rewriting is off or fails."""
        if not (self.use_query_rewriting and self.query_rewriter):
            return [question]
        try:
            queries = self.query_rewriter.rewrite(query=question, context=context, num_variations=3)
        except Exception as e:
            logger.exception(f"Query rewriting skipped for '{question[:50]}'")
            features["errors"]["query_rewriting"] = f"{type(e).__name__}: {e}"
            return [question]

        features["query_rewriting_used"] = True
        features["query_variations"] = queries
        logger.info(f"Generated {len(queries)} query variations")
        return queries

    def _retrieve_and_fuse(self, queries: list[str], filters: dict | None) -> list[dict]:
        """Retrieve and fuse results for each query, then deduplicate."""
        all_results = []
        for query in queries:
            retrieval_results = self.retriever.retrieve(
                query,
                vector_limit=10,
                graph_limit=10,
                filters=filters,
            )
            all_results.extend(
                self.fusion.fuse(
                    retrieval_results["vector_results"],
                    retrieval_results["graph_results"],
                )
            )
        return self._deduplicate_results(all_results)

    def _retrieve_past_cases(
        self,
        question: str,
        context: dict[str, Any] | None,
        features: dict[str, Any],
    ) -> list[dict]:
        """Return similar past sessions from conversation memory, or [] when off or failed."""
        if not (self.use_conversation_memory and self.semantic_memory and context):
            return []
        try:
            past_cases = self.semantic_memory.retrieve_for_context(
                current_query=question,
                equipment_oem=context.get("equipment_oem"),
                equipment_model=context.get("equipment_model"),
                symptoms=context.get("symptoms"),
                top_k=3,
            )
        except Exception as e:
            logger.exception(f"Conversation memory retrieval skipped for '{question[:50]}'")
            features["errors"]["conversation_memory"] = f"{type(e).__name__}: {e}"
            return []

        features["conversation_memory_used"] = True
        features["past_cases_found"] = len(past_cases)
        logger.info(f"Retrieved {len(past_cases)} similar past conversations")
        return past_cases

    def _rerank(
        self,
        question: str,
        results: list[dict],
        context: dict[str, Any] | None,
        features: dict[str, Any],
    ) -> list[dict]:
        """Return re-ranked results, or the input order when re-ranking is off or fails."""
        if not (self.use_reranking and self.reranker):
            return results
        try:
            reranked = self.reranker.rerank(
                query=question,
                results=results,
                context=context,
                top_k=None,
            )
        except Exception as e:
            logger.exception(f"Re-ranking of {len(results)} results skipped")
            features["errors"]["reranking"] = f"{type(e).__name__}: {e}"
            return results

        features["reranking_used"] = True
        logger.info(f"Re-ranked {len(reranked)} results")
        return reranked

    def _generate_answer(
        self,
        question: str,
        context_results: list[dict],
        conversation_results: list[dict],
        context: dict[str, Any] | None = None,
    ) -> dict:
        """Generate the answer from manual excerpts and past cases.

        Args:
            question: User question.
            context_results: Retrieved manual excerpts.
            conversation_results: Retrieved past sessions.
            context: Optional equipment and symptom context.

        Returns:
            Dict with the answer text.

        Raises:
            openai.OpenAIError: The OpenAI request failed. The error is logged first.
        """
        if self.client is None:
            return {"answer": self._offline_answer(context_results)}

        conversation_context = ""
        if conversation_results and self.semantic_memory:
            conversation_context = self.semantic_memory.format_for_retrieval(conversation_results)

        user_prompt = self._build_user_prompt(
            question, context_results, conversation_context, context
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
        except Exception as e:
            logger.error(f"Answer generation failed for '{question[:50]}': {type(e).__name__}: {e}")
            raise

        logger.info("Answer generated")
        return {"answer": response.choices[0].message.content}

    def _build_user_prompt(
        self,
        question: str,
        context_results: list[dict],
        conversation_context: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Assemble equipment context, manual excerpts, past cases and the question."""
        parts = format_equipment_context(context or {})
        parts.append(f"\nContext from service manuals:\n\n{self._build_context(context_results)}")
        if conversation_context:
            parts.append(f"\n\n{conversation_context}")
        parts.append(f"\n\nQuestion: {question}")
        parts.append(f"\n\n{ANSWER_INSTRUCTIONS}")
        return "\n".join(parts)

    def _offline_answer(self, results: list[dict]) -> str:
        """Return the top excerpts as the answer when no LLM is configured."""
        if not results:
            return "No relevant content was found in the indexed service manuals."

        excerpts = [
            f"[Source {i}] {result.get('content', '')[:OFFLINE_EXCERPT_CHARS]}"
            for i, result in enumerate(results, start=1)
        ]
        header = "Offline mode: no language model is configured. Most relevant manual excerpts:"
        return "\n\n".join([header, *excerpts])

    def _build_context(self, results: list[dict]) -> str:
        """Format results as numbered sources with their score and retrieval source."""
        context_parts = []

        for i, result in enumerate(results, start=1):
            if "cross_encoder_score" in result:
                score_type, score = "Cross-Encoder", result["cross_encoder_score"]
            elif "enhanced_score" in result:
                score_type, score = "Context", result["enhanced_score"]
            else:
                score_type, score = "RRF", result.get("rrf_score", 0)

            sources = ", ".join(result.get("sources", []))
            context_parts.append(
                f"[Source {i}] ({score_type} Score: {score:.3f}, From: {sources})\n"
                f"{result.get('content', '')}\n"
            )

        return "\n---\n".join(context_parts)

    def _deduplicate_results(self, results: list[dict]) -> list[dict]:
        """Remove results whose first 200 characters of content repeat."""
        seen = set()
        deduplicated = []

        for result in results:
            key = result.get("content", "")[:200]
            if key not in seen:
                seen.add(key)
                deduplicated.append(result)

        return deduplicated
