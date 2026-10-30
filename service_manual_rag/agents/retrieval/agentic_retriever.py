"""Iterative retrieval with quality assessment.

The retriever runs a retrieval function, assesses whether the results answer the
query, and rewrites the query for the next iteration when they do not. The
assessment uses an OpenAI model when one is configured and a result-count
heuristic otherwise.
"""

import json
from collections.abc import Callable
from typing import Any

from openai import OpenAI

from service_manual_rag.agents.retrieval.query_rewriter import format_equipment_context
from service_manual_rag.config import get_settings
from service_manual_rag.utils.logging_config import get_logger

logger = get_logger(__name__)

RetrievalFunction = Callable[[str], list[dict]]

ASSESSMENT_SYSTEM_PROMPT = (
    "You assess information retrieval quality for technical service documentation."
)


class AgenticRetriever:
    """Retrieval loop that assesses results and adapts the query."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        """Initialize the retriever.

        Args:
            api_key: OpenAI API key. Defaults to the configured key.
            model: OpenAI chat model name. Defaults to settings.openai_model.

        Raises:
            MissingCredentialError: OpenAI mode is selected and no key is configured.
        """
        settings = get_settings()
        self.model = model or settings.openai_model
        self.client: OpenAI | None = None
        if api_key or settings.use_openai:
            self.client = OpenAI(api_key=settings.require_openai_key(api_key))

        mode = self.model if self.client else "offline heuristic assessment"
        logger.info(f"AgenticRetriever initialized: {mode}")

    async def retrieve_with_reflection(
        self,
        query: str,
        retrieval_fn: RetrievalFunction,
        context: dict[str, Any] | None = None,
        max_iterations: int = 3,
        min_results: int = 3,
    ) -> dict[str, Any]:
        """Retrieve, assess and adapt the query until results are sufficient.

        Args:
            query: Search query.
            retrieval_fn: Function that takes a query string and returns a list of results.
            context: Optional context information.
            max_iterations: Maximum retrieval iterations.
            min_results: Result count that ends the loop.

        Returns:
            Dict with results, iterations, final_assessment and query_evolution. An
            iteration whose retrieval raised carries an "error" entry.

        Example:
            >>> retriever = AgenticRetriever()
            >>> result = await retriever.retrieve_with_reflection(
            ...     "motor bearing overheating",
            ...     indexer_search,
            ...     context={"equipment_oem": "ExampleCo", "equipment_model": "EM-30"},
            ... )
        """
        logger.info(f"Starting agentic retrieval for: {query[:50]}")

        all_results: list[dict] = []
        iterations: list[dict[str, Any]] = []
        assessment: dict[str, Any] = {}
        current_query = query

        for iteration in range(max_iterations):
            logger.info(f"Iteration {iteration + 1}/{max_iterations}")
            record: dict[str, Any] = {"iteration": iteration + 1, "query": current_query}

            try:
                new_results = retrieval_fn(current_query)
            except Exception as e:
                logger.exception(
                    f"Retrieval failed for '{current_query[:50]}' on iteration {iteration + 1}"
                )
                record["error"] = f"{type(e).__name__}: {e}"
                new_results = []
            if not isinstance(new_results, list):
                new_results = []

            record["results_count"] = len(new_results)
            all_results = self._deduplicate_results(all_results + new_results)

            assessment = self._assess_retrieval_quality(
                query=query,
                results=all_results,
                context=context,
            )
            record["assessment"] = assessment
            iterations.append(record)

            logger.info(f"Assessment: {str(assessment.get('reasoning', 'N/A'))[:100]}")

            if assessment.get("sufficient") or len(all_results) >= min_results:
                logger.info(f"Retrieval sufficient after {iteration + 1} iteration(s)")
                break

            if iteration < max_iterations - 1:
                current_query = self._adapt_query(
                    original_query=query,
                    assessment=assessment,
                    context=context,
                )
                logger.info(f"Adapted query: {current_query}")

        return {
            "results": all_results,
            "iterations": iterations,
            "final_assessment": assessment,
            "query_evolution": [it["query"] for it in iterations],
        }

    def _assess_retrieval_quality(
        self,
        query: str,
        results: list[dict],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assess whether the results are sufficient.

        Returns:
            Assessment dict. "method" is "llm" or "heuristic". When the LLM request
            fails the heuristic result is returned with an "error" entry.
        """
        if not results:
            return {
                "sufficient": False,
                "needs_broader_search": True,
                "needs_more_specific": False,
                "needs_different_source": False,
                "reasoning": "No results found. Need broader search.",
                "method": "heuristic",
            }

        if self.client is None:
            return self._heuristic_assessment(results)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ASSESSMENT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": self._build_assessment_prompt(query, results, context),
                    },
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            assessment = json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.exception(f"Quality assessment failed for '{query[:50]}', using heuristic")
            return {**self._heuristic_assessment(results), "error": f"{type(e).__name__}: {e}"}

        logger.info("Generated quality assessment")
        return {**assessment, "method": "llm"}

    def _build_assessment_prompt(
        self,
        query: str,
        results: list[dict],
        context: dict[str, Any] | None,
    ) -> str:
        """Build the prompt for the LLM quality assessment."""
        context_str = self._format_context(context) if context else ""
        return f"""Assess the quality of these search results for the query.

Query: {query}

{context_str or "No additional context"}

Retrieved {len(results)} results. Top 5 snippets:
{self._format_top_results(results[:5])}

Assess retrieval quality:
1. Are results relevant to the query?
2. Is there enough information to answer the query?
3. Do we need broader search (expand scope)?
4. Do we need more specific search (narrow scope)?
5. Should we try a different approach?

Respond in JSON:
{{
    "sufficient": bool,
    "needs_broader_search": bool,
    "needs_more_specific": bool,
    "needs_different_source": bool,
    "relevance_score": float (0-1),
    "coverage_score": float (0-1),
    "reasoning": str
}}"""

    @staticmethod
    def _heuristic_assessment(results: list[dict]) -> dict[str, Any]:
        """Assess results by count alone."""
        return {
            "sufficient": len(results) >= 3,
            "needs_broader_search": len(results) < 3,
            "needs_more_specific": len(results) > 20,
            "needs_different_source": False,
            "relevance_score": 0.5,
            "coverage_score": 0.5,
            "reasoning": f"Heuristic assessment from result count ({len(results)})",
            "method": "heuristic",
        }

    def _adapt_query(
        self,
        original_query: str,
        assessment: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> str:
        """Choose the next query from the assessment flags."""
        if assessment.get("needs_broader_search"):
            return self._expand_query(original_query, context)
        if assessment.get("needs_more_specific"):
            return self._narrow_query(original_query, context)
        if assessment.get("needs_different_source"):
            return self._rephrase_query(original_query)
        return f"{original_query} troubleshooting"

    def _expand_query(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Broaden the query with general terms and the manufacturer name."""
        expanded = query
        if "failure" not in query.lower():
            expanded += " failure"
        if "problem" not in query.lower():
            expanded += " problem"

        if context and context.get("equipment_oem"):
            expanded = f"{context['equipment_oem']} {expanded}"

        return expanded

    def _narrow_query(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Narrow the query with the first symptom and the equipment model."""
        narrowed = query
        if context:
            symptoms = context.get("symptoms")
            if isinstance(symptoms, list) and symptoms:
                narrowed += f" {symptoms[0]}"
            if context.get("equipment_model"):
                narrowed += f" {context['equipment_model']}"
        return narrowed

    def _rephrase_query(self, query: str) -> str:
        """Replace the first matching term with a synonym."""
        replacements = {
            "leak": "leaking fluid",
            "noise": "abnormal sound",
            "failure": "malfunction",
            "broken": "not working",
            "error": "fault",
        }

        rephrased = query
        for old, new in replacements.items():
            if old in query.lower():
                rephrased = rephrased.lower().replace(old, new)
                break

        return rephrased if rephrased != query else f"{query} service manual"

    def _format_top_results(self, results: list[dict]) -> str:
        """Format up to five results with score and a 150 character snippet."""
        if not results:
            return "No results"

        parts = []
        for i, result in enumerate(results[:5], 1):
            content = self._extract_content(result)
            score = result.get("score") or result.get("rrf_score") or 0
            parts.append(f"{i}. (Score: {score:.3f}) {content[:150]}")

        return "\n".join(parts)

    def _format_context(self, context: dict[str, Any]) -> str:
        """Format equipment and symptoms for prompts."""
        return "\n".join(format_equipment_context(context))

    def _extract_content(self, result: dict) -> str:
        """Return the result text."""
        return (
            result.get("content") or result.get("text") or result.get("chunk_text") or str(result)
        )

    def _deduplicate_results(self, results: list[dict]) -> list[dict]:
        """Remove results whose first 200 characters of content repeat."""
        seen = set()
        deduplicated = []

        for result in results:
            key = self._extract_content(result)[:200]
            if key not in seen:
                seen.add(key)
                deduplicated.append(result)

        return deduplicated

    def retrieve_with_fallback(
        self,
        query: str,
        primary_retrieval_fn: RetrievalFunction,
        fallback_retrieval_fn: RetrievalFunction | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retrieve with the primary function, then the fallback when it fails or is empty.

        Args:
            query: Search query.
            primary_retrieval_fn: Primary retrieval function.
            fallback_retrieval_fn: Optional fallback retrieval function.
            context: Optional context. Not used by the current strategy.

        Returns:
            Dict with results, source ("primary", "fallback" or "none"), success, and
            errors (a list of "<source>: <exception>" strings for attempts that raised).
        """
        logger.info(f"Retrieving with fallback: {query[:50]}")
        errors: list[str] = []
        attempts = [("primary", primary_retrieval_fn)]
        if fallback_retrieval_fn:
            attempts.append(("fallback", fallback_retrieval_fn))

        for source, retrieval_fn in attempts:
            try:
                results = retrieval_fn(query)
            except Exception as e:
                logger.exception(f"{source.capitalize()} retrieval failed for '{query[:50]}'")
                errors.append(f"{source}: {type(e).__name__}: {e}")
                continue
            if results:
                logger.info(f"{source.capitalize()} retrieval returned {len(results)} results")
                return {"results": results, "source": source, "success": True, "errors": errors}

        logger.warning(f"All retrieval attempts returned no results for '{query[:50]}'")
        return {"results": [], "source": "none", "success": False, "errors": errors}
