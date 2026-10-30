"""Query rewriting for service manual retrieval.

Produces query variations with an OpenAI model when one is configured, and with
deterministic keyword expansion otherwise.
"""

import json
from typing import Any

from openai import OpenAI

from service_manual_rag.config import get_settings
from service_manual_rag.utils.logging_config import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You optimize search queries for industrial equipment service manuals, "
    "such as pumps, compressors, motors and their controls."
)

SYNONYMS = {
    "leak": ["leak", "leaking", "seepage", "dripping", "fluid loss"],
    "noise": ["noise", "sound", "knocking", "rattling", "abnormal"],
    "failure": ["failure", "malfunction", "not working", "breakdown"],
    "pressure": ["pressure", "psi", "discharge pressure", "system pressure"],
    "temperature": ["temperature", "temp", "overheating", "heat"],
    "vibration": ["vibration", "shaking", "oscillation", "wobble"],
    "power": ["power", "performance", "output", "horsepower"],
}


def format_equipment_context(context: dict[str, Any]) -> list[str]:
    """Return "Equipment: ..." and "Symptoms: ..." lines for the fields present in context."""
    parts = []
    if context.get("equipment_oem") and context.get("equipment_model"):
        parts.append(f"Equipment: {context['equipment_oem']} {context['equipment_model']}")
    symptoms = context.get("symptoms")
    if symptoms:
        if isinstance(symptoms, list):
            parts.append(f"Symptoms: {', '.join(symptoms)}")
        else:
            parts.append(f"Symptoms: {symptoms}")
    return parts


class QueryRewriter:
    """Rewrite user queries for retrieval.

    Strategies:
    1. Symptom expansion, for example "knocking" to "knocking rattling abnormal noise".
    2. Technical term injection, for example "won't start" to "motor fails to start".
    3. Context incorporation, for example prefixing the equipment model.
    4. Multi-query generation (three variations by default).

    With no OpenAI client (offline mode) only the deterministic expansion runs.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        """Initialize the query rewriter.

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

        mode = self.model if self.client else "offline keyword expansion"
        logger.info(f"QueryRewriter initialized: {mode}")

    def rewrite(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        num_variations: int = 3,
    ) -> list[str]:
        """Generate query variations for service manual search.

        Args:
            query: Original user query.
            context: Optional context (equipment_oem, equipment_model, symptoms,
                conversation_summary).
            num_variations: Number of variations to generate.

        Returns:
            The original query followed by the variations.

        Raises:
            openai.OpenAIError: The OpenAI request failed.
            ValueError: The model response was not valid JSON.
            TypeError: The model response was not a JSON object with a 'variations' list.

        Example:
            >>> rewriter = QueryRewriter()
            >>> context = {
            ...     "equipment_oem": "ExampleCo",
            ...     "equipment_model": "CP-200",
            ...     "symptoms": ["cavitation noise", "low discharge pressure"],
            ... }
            >>> rewriter.rewrite("pump making noise", context)
        """
        context = context or {}
        variations = [query]

        if self.client is None:
            variations.extend(self._simple_expansion(query, context))
            return variations

        prompt = self._build_rewrite_prompt(
            query=query,
            context_str=self._build_context_string(context),
            num_variations=num_variations,
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Query rewriting failed for '{query[:50]}': {type(e).__name__}: {e}")
            raise

        if not isinstance(parsed, dict) or not isinstance(parsed.get("variations"), list):
            message = f"Query rewriting response for '{query[:50]}' has no 'variations' list"
            logger.error(message)
            raise TypeError(message)
        variations.extend(parsed["variations"][:num_variations])

        logger.info(f"Generated {len(variations)} query variations")
        return variations

    def _build_context_string(self, context: dict[str, Any]) -> str:
        """Build the context block for the rewrite prompt."""
        parts = format_equipment_context(context)
        if context.get("conversation_summary"):
            parts.append(f"Context: {context['conversation_summary']}")
        return "\n".join(parts) if parts else "No additional context"

    def _build_rewrite_prompt(
        self,
        query: str,
        context_str: str,
        num_variations: int,
    ) -> str:
        """Build the prompt for query rewriting."""
        return f"""Given this troubleshooting query: "{query}"

{context_str}

Generate {num_variations} variations of this query for searching technical service manuals.
Each variation focuses on a different aspect:

1. Symptom-focused: expand symptom keywords with technical synonyms and related symptoms.
   Example: "knocking" becomes "knocking rattling ticking abnormal noise".

2. Component-focused: name the components and systems that may be involved.
   Example: "won't start" becomes "motor starter contactor overload trip failure to start".

3. Procedure-focused: frame the query as a service or repair procedure.
   Example: "oil leak" becomes "oil leak diagnosis repair procedure troubleshooting".

Guidelines:
- Use technical terminology from service manuals.
- Include the equipment model if relevant.
- Expand abbreviations and add synonyms.
- Keep each variation to 5 to 10 words.
- Use search-friendly keywords.

Respond in JSON format:
{{
    "variations": [
        "variation 1",
        "variation 2",
        "variation 3"
    ]
}}"""

    def _simple_expansion(
        self,
        query: str,
        context: dict[str, Any],
    ) -> list[str]:
        """Expand the query without an LLM.

        Args:
            query: Original query.
            context: Context dictionary.

        Returns:
            Up to two expanded queries.
        """
        variations = []
        query_lower = query.lower()

        if context.get("equipment_oem") and context.get("equipment_model"):
            equipment = f"{context['equipment_oem']} {context['equipment_model']}"
            variations.append(f"{equipment} {query}")

        if "troubleshoot" not in query_lower:
            variations.append(f"{query} troubleshooting")

        if "manual" not in query_lower and "procedure" not in query_lower:
            variations.append(f"{query} service procedure")

        return variations[:2]

    def expand_with_synonyms(self, query: str) -> str:
        """Append maintenance synonyms for terms found in the query.

        Args:
            query: Original query.

        Returns:
            The query followed by up to five synonym terms, or the query unchanged.

        Example:
            >>> QueryRewriter().expand_with_synonyms("seal leak")
            'seal leak leak leaking seepage dripping fluid loss'
        """
        query_lower = query.lower()
        expanded_terms = []
        for term, synonym_list in SYNONYMS.items():
            if term in query_lower:
                expanded_terms.extend(synonym_list)

        if not expanded_terms:
            return query

        unique_terms = list(dict.fromkeys(expanded_terms))
        return f"{query} {' '.join(unique_terms[:5])}"
