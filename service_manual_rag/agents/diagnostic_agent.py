"""Diagnostic agent: searches the manuals, analyzes media and ranks hypotheses.

Steps per turn:
1. Analyze uploaded images and videos with the VLM service.
2. Search the indexed service manuals through the RAG pipeline.
3. Look up similar recorded sessions in the knowledge graph, when it is enabled.
4. Generate hypotheses with the LLM reasoner, or with keyword rules offline.
5. Decide between a diagnosis and a request for more evidence, and add safety warnings.
"""

from datetime import datetime
from typing import Any

from openai import OpenAIError

from service_manual_rag.agents.components import create_graph_manager, create_openai_client
from service_manual_rag.agents.reasoning.llm_reasoner import LLMDiagnosticReasoner
from service_manual_rag.agents.state import (
    AgentState,
    DiagnosticResult,
    DiagnosticStatus,
    Evidence,
    Hypothesis,
)
from service_manual_rag.agents.vlm_service import VLMService, get_vlm_service
from service_manual_rag.config import Settings, get_settings
from service_manual_rag.knowledge_graph.neo4j_client import Neo4JManager
from service_manual_rag.rag.enhanced_rag_pipeline import EnhancedRAGPipeline
from service_manual_rag.utils.logging_config import get_logger

logger = get_logger(__name__)

IMAGE_TYPES = ("image", "jpg", "jpeg", "png")
VIDEO_TYPES = ("video", "mp4", "mov")

DIAGNOSIS_THRESHOLD = 0.75
MORE_EVIDENCE_THRESHOLD = 0.5
# Fused result fields kept with each manual section, for citation in prompts and the UI
SECTION_METADATA = ("section_name", "oem", "model", "doc_id", "page_numbers")

# Similar recorded cases are historical, so their confidence is discounted
SIMILAR_CASE_DISCOUNT = 0.7

MANUAL_EXISTS_QUERY = """
MATCH (e:Equipment {oem: $oem, model: $model})-[:HAS_MANUAL]->(m:Manual)
RETURN count(m) as count
"""

SIMILAR_CASES_QUERY = """
MATCH (s:TroubleshootingSession)
WHERE ANY(symptom IN $symptoms WHERE symptom IN s.symptoms)
RETURN s.symptoms, s.diagnosis, s.resolution, s.confidence
ORDER BY s.confidence DESC
LIMIT 5
"""

# (keywords in the diagnosis, warning)
DIAGNOSIS_WARNINGS = [
    (("electrical", "short"), "ELECTRICAL HAZARD: Ensure power is isolated before inspection"),
    (("pressure", "hydraulic"), "PRESSURE HAZARD: Release pressure before opening system"),
    (
        ("gas", "leak"),
        "LEAK HAZARD: Ensure adequate ventilation and check for flammable vapors",
    ),
]
STRUCTURAL_WARNING = "STRUCTURAL CONCERN: May require engineering assessment"

NEED_DETAILS_MESSAGE = "Please provide more details about the symptoms or equipment behavior"
NEED_CONFIRMATION_MESSAGE = "Visual inspection or additional symptoms would help confirm diagnosis"
INSUFFICIENT_INFORMATION_MESSAGE = (
    "Insufficient information for diagnosis. Please provide: "
    "equipment model, specific symptoms, when the issue started, "
    "and any recent maintenance or changes"
)

# (keywords in the symptoms, hypothesis, base confidence, supporting evidence)
RULE_HYPOTHESES = [
    (
        ("overheating", "hot"),
        "Cooling system failure or blockage",
        0.6,
        "Overheating symptom reported",
    ),
    (("leak",), "Seal or gasket failure", 0.7, "Leak symptom reported"),
    (("noise", "rattle"), "Bearing wear or misalignment", 0.65, "Abnormal noise reported"),
]


def media_kind(media_type: str) -> str | None:
    """Map a MIME type such as 'image/png', or a file extension, to 'image' or 'video'."""
    kind = media_type.lower().split("/")[0]
    if kind in IMAGE_TYPES:
        return "image"
    if kind in VIDEO_TYPES:
        return "video"
    return None


def rule_based_hypotheses(symptoms: list[str], manual_sections: list[dict]) -> list[Hypothesis]:
    """Keyword hypotheses used offline and when the LLM reasoner fails."""
    symptoms_text = " ".join(symptoms).lower()
    boost = 1.2 if manual_sections else 1.0
    hypotheses = []
    for keywords, description, confidence, evidence in RULE_HYPOTHESES:
        if not any(keyword in symptoms_text for keyword in keywords):
            continue
        supporting = [evidence]
        if manual_sections:
            supporting.append(f"Found {len(manual_sections)} relevant manual sections")
        hypotheses.append(
            Hypothesis(
                description=description,
                confidence=min(0.95, confidence * boost),
                supporting_evidence=supporting,
            )
        )
    return hypotheses


def create_rag_pipeline() -> EnhancedRAGPipeline:
    """Build the RAG pipeline the diagnostic agent uses; it runs offline without a key."""
    return EnhancedRAGPipeline(
        use_query_rewriting=True,
        use_reranking=True,
        use_conversation_memory=True,
    )


def create_llm_reasoner(settings: Settings) -> LLMDiagnosticReasoner | None:
    """Return an LLM reasoner when OpenAI is in use, otherwise None."""
    client = create_openai_client(settings)
    if client is None:
        return None
    return LLMDiagnosticReasoner(llm=client, model=settings.openai_model)


class DiagnosticAgent:
    """Combines manual search, media analysis and the knowledge graph into a diagnosis."""

    def __init__(
        self,
        rag_pipeline: Any = None,
        neo4j_manager: Neo4JManager | None = None,
        vlm_service: VLMService | None = None,
        llm_reasoner: LLMDiagnosticReasoner | None = None,
        settings: Settings | None = None,
    ):
        """
        Args:
            rag_pipeline: Object with ``query(question, top_k, filters)`` returning a dict
                with 'answer' and 'sources'; defaults to an EnhancedRAGPipeline.
            neo4j_manager: Knowledge graph manager; created only if GRAPH_ENABLED is true.
            vlm_service: Vision service; defaults to the shared VLM service.
            llm_reasoner: Hypothesis generator; created only if OpenAI is in use.
            settings: Application settings; defaults to ``get_settings()``.

        Raises:
            MissingCredentialError: LLM_MODE=openai and OPENAI_API_KEY is not set.
        """
        settings = settings or get_settings()
        self.llm_reasoner = (
            llm_reasoner if llm_reasoner is not None else create_llm_reasoner(settings)
        )
        self.rag = rag_pipeline if rag_pipeline is not None else create_rag_pipeline()
        self.kg = neo4j_manager if neo4j_manager is not None else create_graph_manager(settings)
        self.vlm = vlm_service if vlm_service is not None else get_vlm_service()
        logger.info(
            "Diagnostic agent initialized (graph=%s, reasoner=%s)",
            "on" if self.kg else "off",
            "llm" if self.llm_reasoner else "keyword rules",
        )

    async def diagnose(self, state: AgentState) -> DiagnosticResult:
        """Run one diagnostic pass over the current state."""
        logger.info("Starting diagnosis for session: %s", state.get("session_id"))

        observations, new_evidence = self._analyze_media(state)
        evidence = [*state.get("evidence", []), *new_evidence]
        hypotheses: list[Hypothesis] = []
        relevant_sections: list[dict] = []

        if state.get("symptoms") or state.get("equipment_oem"):
            manual_results = self._search_manuals(state)
            relevant_sections = manual_results["sections"]
            observations.extend(manual_results["observations"])

        if state.get("symptoms"):
            kg_results = self._query_knowledge_graph(state)
            hypotheses.extend(kg_results["hypotheses"])
            observations.extend(kg_results["observations"])

        hypotheses.extend(await self._generate_hypotheses(state, evidence, relevant_sections))

        diagnosis, recommended_action, evidence_needed = self._make_recommendation(
            hypotheses, state, relevant_sections
        )

        result = DiagnosticResult(
            observations=observations,
            hypotheses=hypotheses,
            diagnosis=diagnosis,
            recommended_action=recommended_action,
            evidence_needed=evidence_needed,
            safety_warnings=self._check_safety_warnings(diagnosis, hypotheses),
            relevant_manual_sections=relevant_sections,
            new_evidence=new_evidence,
        )
        logger.info(
            "Diagnosis complete: %d observations, %d hypotheses, diagnosis=%s",
            len(observations),
            len(hypotheses),
            "yes" if diagnosis else "no",
        )
        return result

    def _analyze_media(self, state: AgentState) -> tuple[list[str], list[Evidence]]:
        """Analyze media not yet analyzed and return (observations, evidence).

        Each finding becomes an observation for the reply and an Evidence item for
        hypothesis generation. A failed analysis becomes an observation only.
        """
        observations = []
        evidence = []
        equipment_context = {
            "oem": state.get("equipment_oem") or "",
            "model": state.get("equipment_model") or "",
        }

        for media in state.get("uploaded_media", []):
            kind = media_kind(media.get("type", ""))
            media_path = media.get("path", "")
            if kind is None:
                logger.info("Skipping media with unsupported type: %s", media.get("type"))
                continue

            try:
                if kind == "image":
                    vlm_result = self.vlm.analyze_image(
                        media_path, analysis_type="defect", equipment_context=equipment_context
                    )
                else:
                    vlm_result = self.vlm.analyze_video(
                        media_path, analysis_type="general", equipment_context=equipment_context
                    )
            except Exception as error:
                logger.exception("%s analysis failed for %s", kind.capitalize(), media_path)
                observations.append(f"Could not analyze {kind}: {error}")
                continue

            label = "Visual inspection" if kind == "image" else "Video analysis"
            for finding in vlm_result.get("findings", []):
                description = finding.get("description", "")
                observation = f"{label}: {description}"
                if kind == "image":
                    observation += f" (confidence: {finding.get('confidence', 0):.2f})"
                observations.append(observation)
                if description:
                    evidence.append(
                        Evidence(kind, description, datetime.now(), metadata={"media": media_path})
                    )

        return observations, evidence

    def _check_manual_exists(self, oem: str, model: str) -> bool:
        """Return True if the knowledge graph links a manual to this equipment."""
        if self.kg is None:
            return False
        results = self.kg.execute_query(
            MANUAL_EXISTS_QUERY, {"oem": oem.upper(), "model": model.upper()}
        )
        exists = bool(results) and results[0].get("count", 0) > 0
        logger.debug("Manual exists for %s %s: %s", oem, model, exists)
        return exists

    @staticmethod
    def _build_manual_query(state: AgentState) -> str:
        """Build the search query from the equipment and either symptoms or the question."""
        query_parts = []
        if state.get("equipment_oem") and state.get("equipment_model"):
            query_parts.append(f"{state['equipment_oem']} {state['equipment_model']}")

        if state.get("status") == DiagnosticStatus.DIAGNOSING:
            query_parts.extend(state.get("symptoms", []))
            return " ".join(query_parts) + " troubleshooting repair"

        # Information question: search with the user's own words
        messages = state.get("messages", [])
        if messages and messages[-1]["role"] == "user":
            query_parts.append(messages[-1]["content"])
        return " ".join(query_parts)

    def _search_manuals(self, state: AgentState) -> dict[str, list]:
        """Search the indexed manuals; report plainly when they do not cover the question."""
        query = self._build_manual_query(state)
        if not query.strip():
            return {"observations": [], "sections": []}

        oem = state.get("equipment_oem")
        model = state.get("equipment_model")
        # OEM and model keep their casing so they match the indexed payload
        filters = {"oem": oem.strip(), "model": model.strip()} if oem and model else None
        logger.info("Searching manuals with query %r and filters %s", query, filters)

        try:
            rag_result = self.rag.query(query, top_k=5, filters=filters)
        except Exception as error:
            logger.exception("Manual search failed for query %r", query)
            return {"observations": [f"Manual search error: {error}"], "sections": []}

        answer = rag_result.get("answer", "")
        sources = rag_result.get("sources", [])
        if not sources:
            return {"observations": [self._no_coverage_message(oem, model)], "sections": []}

        sections = [
            {
                "id": source.get("id"),
                "content": source.get("content", ""),
                "sources": source.get("sources", []),
                "score": source.get("rrf_score", 0.0),
                "metadata": {
                    field: source[field] for field in SECTION_METADATA if source.get(field)
                },
            }
            for source in sources
        ]
        logger.info("Manual search returned %d sources", len(sources))
        observations = [f"Found information in the service manual:\n\n{answer}"] if answer else []
        return {"observations": observations, "sections": sections}

    def _no_coverage_message(self, oem: str | None, model: str | None) -> str:
        """Explain that the indexed manuals hold no passage for this question."""
        if not (oem and model):
            return (
                "The indexed service manuals do not cover this question. Name the equipment "
                "manufacturer and model, or rephrase the question with different keywords."
            )

        try:
            manual_exists = self._check_manual_exists(oem, model)
        except Exception:
            logger.exception("Manual lookup in the knowledge graph failed for %s %s", oem, model)
            manual_exists = False

        if manual_exists:
            return (
                f"The service manual for the {oem} {model} is indexed, but no section "
                "matches this query.\n\n"
                "**Suggestions:**\n"
                "- Rephrase the question with different keywords\n"
                "- Describe in more detail what you are looking for\n"
                "- Check the printed manual or contact the manufacturer"
            )
        return (
            f"The indexed service manuals do not cover this question for the {oem} {model}. "
            "Index the manual for this equipment, or rephrase the question."
        )

    def _query_knowledge_graph(self, state: AgentState) -> dict[str, list]:
        """Turn similar recorded sessions from the knowledge graph into hypotheses."""
        if self.kg is None:
            logger.info("Knowledge graph disabled; skipping similar case lookup")
            return {"observations": [], "hypotheses": []}

        symptoms = state.get("symptoms", [])
        try:
            similar_cases = self.kg.execute_query(SIMILAR_CASES_QUERY, {"symptoms": symptoms})
        except Exception as error:
            logger.exception("Similar case lookup failed for symptoms %s", symptoms)
            return {"observations": [f"Could not query similar cases: {error}"], "hypotheses": []}

        observations = []
        hypotheses = []
        for case in similar_cases:
            case_symptoms = ", ".join(case.get("s.symptoms") or [])
            case_diagnosis = case.get("s.diagnosis") or ""
            case_confidence = case.get("s.confidence") or 0.5
            hypotheses.append(
                Hypothesis(
                    description=case_diagnosis,
                    confidence=case_confidence * SIMILAR_CASE_DISCOUNT,
                    supporting_evidence=[f"Similar case with symptoms: {case_symptoms}"],
                )
            )
            observations.append(f"Found similar case: {case_diagnosis} (symptoms: {case_symptoms})")

        return {"observations": observations, "hypotheses": hypotheses}

    async def _generate_hypotheses(
        self, state: AgentState, evidence: list[Evidence], manual_sections: list[dict]
    ) -> list[Hypothesis]:
        """Generate hypotheses with the LLM reasoner, or with keyword rules."""
        symptoms = state.get("symptoms", [])
        if not symptoms and not evidence:
            logger.info("No symptoms or evidence; skipping hypothesis generation")
            return []

        # The keyword rules read evidence descriptions as additional symptom text
        rule_input = [*symptoms, *(item.content for item in evidence)]
        if self.llm_reasoner is None:
            return rule_based_hypotheses(rule_input, manual_sections)

        try:
            return await self.llm_reasoner.generate_hypotheses(
                symptoms=symptoms,
                evidence=evidence,
                context=state,
                retrieved_docs=manual_sections,
            )
        except OpenAIError:
            logger.exception("LLM hypothesis generation failed; using keyword rules")
            return rule_based_hypotheses(rule_input, manual_sections)

    @staticmethod
    def _make_recommendation(
        hypotheses: list[Hypothesis], state: AgentState, manual_sections: list[dict]
    ) -> tuple[str | None, str | None, str | None]:
        """Return (diagnosis, recommended_action, evidence_needed).

        Coverage is judged from the retrieved manual sections: an information question
        answered from the manuals needs no diagnosis and no further evidence.
        """
        if not hypotheses:
            if manual_sections:
                if state.get("status") == DiagnosticStatus.DIAGNOSING:
                    return None, None, NEED_DETAILS_MESSAGE
                return None, None, None
            if state.get("equipment_oem") and state.get("equipment_model"):
                equipment = f"{state['equipment_oem']} {state['equipment_model']}"
                message = (
                    f"I could not find specific information for the {equipment} in the "
                    "manuals or knowledge base. Please verify the model number, or describe "
                    "the issue in more detail."
                )
                return None, None, message
            return None, None, NEED_DETAILS_MESSAGE

        top_hypothesis = max(hypotheses, key=lambda h: h.confidence)

        if top_hypothesis.confidence > DIAGNOSIS_THRESHOLD:
            diagnosis = top_hypothesis.description
            recommended_action = (
                f"Based on {top_hypothesis.confidence:.0%} confidence: "
                f"{diagnosis}. Recommend following troubleshooting procedure in manual."
            )
            return diagnosis, recommended_action, None

        if top_hypothesis.confidence > MORE_EVIDENCE_THRESHOLD:
            return None, None, NEED_CONFIRMATION_MESSAGE

        return None, None, INSUFFICIENT_INFORMATION_MESSAGE

    @staticmethod
    def _check_safety_warnings(diagnosis: str | None, hypotheses: list[Hypothesis]) -> list[str]:
        """Return safety warnings triggered by the diagnosis or confident hypotheses."""
        warnings = []
        if diagnosis:
            diagnosis_lower = diagnosis.lower()
            for keywords, warning in DIAGNOSIS_WARNINGS:
                if any(keyword in diagnosis_lower for keyword in keywords):
                    warnings.append(warning)

        for hypothesis in hypotheses:
            description = hypothesis.description.lower()
            if hypothesis.confidence > 0.6 and (
                "structural" in description or "crack" in description
            ):
                warnings.append(STRUCTURAL_WARNING)

        return list(dict.fromkeys(warnings))


def create_diagnostic_agent() -> DiagnosticAgent:
    """Create a diagnostic agent from the current settings."""
    return DiagnosticAgent()
