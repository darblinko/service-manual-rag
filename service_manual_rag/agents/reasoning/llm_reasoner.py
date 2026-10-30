"""Hypothesis generation for the diagnostic agent with an OpenAI chat model."""

import re
from typing import Any

from service_manual_rag.agents.state import AgentState, Evidence, Hypothesis
from service_manual_rag.config import get_settings
from service_manual_rag.utils.logging_config import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are an experienced industrial equipment maintenance technician. You generate "
    "diagnostic hypotheses from symptoms, collected evidence and service manual excerpts."
)

DIAGNOSTIC_PROMPT = """EQUIPMENT:
{equipment}

REPORTED SYMPTOMS:
{symptoms}

EVIDENCE COLLECTED:
{evidence}

RELEVANT MANUAL SECTIONS:
{manual_sections}

TASK:
Generate 3 to 5 diagnostic hypotheses that explain the symptoms.
For each hypothesis:
1. Describe the likely root cause
2. Explain which evidence supports it
3. Assign a confidence between 0 and 1 based on the strength of the evidence
4. Suggest additional tests to confirm or rule it out

Consider multiple failure modes and their probabilities.

Use exactly this format for each hypothesis:
1. <root cause> (Confidence: 0.8)
   Evidence: <supporting evidence>
   Tests: <suggested tests>
"""

DEFAULT_CONFIDENCE = 0.6
# Upper bound on the characters of each manual passage placed in the prompt
MAX_PASSAGE_CHARS = 1200
MIN_CONFIDENCE = 0.1
MAX_CONFIDENCE = 0.95


class LLMDiagnosticReasoner:
    """Generates ranked hypotheses by prompting a chat model and parsing its reply."""

    def __init__(self, llm: Any, model: str | None = None):
        """
        Args:
            llm: An OpenAI client, or any object with the same
                ``chat.completions.create`` interface.
            model: Chat model name; defaults to ``settings.openai_model``.
        """
        self.llm = llm
        self.model = model or get_settings().openai_model

    async def generate_hypotheses(
        self,
        symptoms: list[str],
        evidence: list[Evidence],
        context: AgentState,
        retrieved_docs: list[dict],
    ) -> list[Hypothesis]:
        """Return hypotheses sorted by confidence, highest first.

        Raises:
            openai.OpenAIError: The chat completion request failed. The diagnostic agent
                catches this and falls back to its rule-based hypotheses.
        """
        prompt = self._build_diagnostic_prompt(symptoms, evidence, context, retrieved_docs)
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_completion_tokens=1500,
        )
        response_content = response.choices[0].message.content or ""
        logger.debug("LLM response: %s", response_content)

        hypotheses = self._parse_llm_response(response_content)
        for hypothesis in hypotheses:
            if hypothesis.confidence == 0.0:
                hypothesis.confidence = DEFAULT_CONFIDENCE
            hypothesis.confidence = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, hypothesis.confidence))

        logger.info("LLM reasoner generated %d hypotheses", len(hypotheses))
        return sorted(hypotheses, key=lambda h: h.confidence, reverse=True)

    def _build_diagnostic_prompt(
        self,
        symptoms: list[str],
        evidence: list[Evidence],
        context: AgentState,
        retrieved_docs: list[dict],
    ) -> str:
        oem = context.get("equipment_oem") or "Unknown"
        model = context.get("equipment_model") or "Unknown"
        return DIAGNOSTIC_PROMPT.format(
            equipment=f"{oem} {model}",
            symptoms=self._format_symptoms(symptoms),
            evidence=self._format_evidence(evidence),
            manual_sections=self._format_retrieved_docs(retrieved_docs),
        )

    @staticmethod
    def _format_symptoms(symptoms: list[str]) -> str:
        return "\n".join(f"- {symptom}" for symptom in symptoms) if symptoms else "None"

    @staticmethod
    def _format_evidence(evidence: list[Evidence]) -> str:
        if not evidence:
            return "None"
        return "\n".join(f"- Type: {ev.evidence_type}, Content: {ev.content}" for ev in evidence)

    @staticmethod
    def _format_retrieved_docs(docs: list[dict]) -> str:
        """Number each passage, name its manual section, and include its bounded body."""
        if not docs:
            return "None"
        formatted_docs = []
        for number, doc in enumerate(docs, start=1):
            metadata = doc.get("metadata") or {}
            label_parts = [
                str(value)
                for value in (metadata.get("model"), metadata.get("section_name"), doc.get("id"))
                if value
            ]
            label = ", ".join(label_parts) or "Unknown source"
            body = (doc.get("content") or "No content").strip()
            if len(body) > MAX_PASSAGE_CHARS:
                body = body[:MAX_PASSAGE_CHARS].rstrip() + " [truncated]"
            formatted_docs.append(f"[Passage {number}] {label}\n{body}")
        return "\n\n".join(formatted_docs)

    @staticmethod
    def _parse_llm_response(response_content: str) -> list[Hypothesis]:
        """Parse numbered hypotheses in the format requested by DIAGNOSTIC_PROMPT.

        Expected format:
            1. Hypothesis description (Confidence: 0.8)
               Evidence: supporting evidence
               Tests: suggested tests
        """
        hypotheses = []
        current_hypothesis = None

        for raw_line in response_content.split("\n"):
            line = raw_line.strip()

            if line and line[0].isdigit() and "." in line[:3]:
                if current_hypothesis:
                    hypotheses.append(current_hypothesis)

                description = line
                confidence = 0.0
                conf_match = re.search(r"[Cc]onfidence:?\s*([0-9]*\.?[0-9]+)", line)
                if conf_match:
                    confidence = float(conf_match.group(1))
                    description = re.sub(r"\(?[Cc]onfidence:?\s*[0-9.]+\)?", "", line).strip()
                description = re.sub(r"^\d+\.\s*", "", description)

                current_hypothesis = Hypothesis(description=description, confidence=confidence)

            elif current_hypothesis:
                lower = line.lower()
                if lower.startswith("evidence:"):
                    supporting = line[len("evidence:") :].strip()
                    if supporting:
                        current_hypothesis.supporting_evidence.append(supporting)
                elif lower.startswith("test") and ":" in line:
                    test = line[line.find(":") + 1 :].strip()
                    if test:
                        current_hypothesis.tests_suggested.append(test)
                elif line.startswith(("-", "*", "\u2022")):
                    content = line[1:].strip()
                    if content:
                        current_hypothesis.supporting_evidence.append(content)

        if current_hypothesis:
            hypotheses.append(current_hypothesis)

        return hypotheses
