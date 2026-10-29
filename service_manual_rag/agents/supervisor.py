"""Supervisor agent: intent classification, information extraction and routing.

The supervisor owns the conversation with the technician. It extracts equipment and
symptom details from each message, classifies the intent and routes the turn to the
diagnostic agent or the action agent. It has no tools of its own.
"""

import re
from typing import Literal

from openai import OpenAI, OpenAIError

from service_manual_rag.agents.state import AgentState, DiagnosticStatus
from service_manual_rag.config import Settings, get_settings
from service_manual_rag.utils.logging_config import get_logger

logger = get_logger(__name__)

Intent = Literal["troubleshoot", "question", "record", "confirm_action", "unclear"]
VALID_INTENTS = ("troubleshoot", "question", "record", "confirm_action", "unclear")

MANUFACTURER_PATTERNS = [
    r"manufacturer\s+is\s+([A-Za-z][A-Za-z\s]+?)(?:\s+and|\s+model|,|$)",
    r"made\s+by\s+([A-Za-z][A-Za-z\s]+?)(?:\s+model|,|$)",
    r"working\s+on\s+(?:the\s+)?([A-Za-z][A-Za-z\s]+?)\s+[A-Za-z0-9]",
    r"\b([A-Za-z][A-Za-z]+)\s+[A-Za-z0-9]+[-]?[0-9]+",  # "Brand Model" pattern
]

# A model number either contains a hyphen or contains at least one digit.
# Examples: CP-200, RSC-15, EM-30.
MODEL_PATTERN = r"\b([A-Za-z0-9]+-[A-Za-z0-9]+|[A-Za-z]*[0-9]+[A-Za-z]*)\b"

# Tokens that match MODEL_PATTERN but are time windows or versions, not models
STOP_MODELS = ("V1", "V2", "24H", "7D")

SYMPTOM_KEYWORDS = [
    "overheating",
    "overheat",
    "hot",
    "temperature",
    "leak",
    "leaking",
    "drip",
    "dripping",
    "noise",
    "loud",
    "knocking",
    "rattle",
    "rattling",
    "grinding",
    "squeal",
    "squealing",
    "banging",
    "humming",
    "buzzing",
    "vibration",
    "vibrating",
    "shaking",
    "shakes",
    "smoke",
    "smoking",
    "burning smell",
    "error",
    "fault",
    "alarm",
    "warning",
    "code",
    "not starting",
    "won't start",
    "dead",
    "doesn't start",
    "slow",
    "sluggish",
    "weak",
    "loss of power",
    "stalling",
    "stall",
    "dies",
    "pressure drop",
    "low pressure",
    "no pressure",
]

CRITICAL_SYMPTOMS = ("smoke", "fire", "sparks", "explosion", "emergency")
CONFIRM_WORDS = ("yes", "confirm", "go ahead", "do it", "proceed")
DENY_WORDS = ("no", "cancel", "stop", "don't")
QUESTION_PREFIXES = ("what", "when", "how", "which", "where", "who", "is ", "are ", "does", "can")

INTENT_PROMPT = """You are classifying the intent of a technician's message in an \
industrial equipment troubleshooting system.

User Message: "{message}"

Context:
{context}

Classify the intent as ONE of the following:

1. "troubleshoot": User needs diagnostic help with an equipment problem
   Examples: "my pump is leaking", "why is the motor overheating?", "loud knocking noise"

2. "question": Information request that is not troubleshooting
   Examples: "what oil does the compressor use?", "when should I change filters?"

3. "record": User wants to save or log the troubleshooting session
   Examples: "record this session", "save this for later", "log this resolution"

4. "confirm_action": User confirms a suggested action or acknowledges a diagnosis
   Examples: "yes, do it", "proceed", "create the record", "okay thanks", "got it"

5. "unclear": Intent is ambiguous or the user needs guidance
   Examples: "hello", "help", "I don't know"

Return ONLY the intent category as a single word \
(troubleshoot, question, record, confirm_action, or unclear).

Intent:"""


def parse_known_oems(known_oems: str) -> list[str]:
    """Split the comma-separated KNOWN_OEMS setting into names, keeping their casing."""
    return [name.strip() for name in known_oems.split(",") if name.strip()]


def find_known_oem(message: str, known_oems: list[str]) -> str | None:
    """Return the first known OEM named in the message, matched case-insensitively."""
    for oem in known_oems:
        if re.search(r"\b" + re.escape(oem) + r"\b", message, re.IGNORECASE):
            return oem
    return None


class SupervisorAgent:
    """Routes each turn to a specialist agent and formats the reply to the user.

    When ``settings.use_openai`` is true, ambiguous intents are classified by the
    OpenAI chat model. Otherwise a deterministic keyword classifier is used.
    """

    def __init__(self, settings: Settings | None = None, llm_client: OpenAI | None = None):
        self.settings = settings or get_settings()
        self.model = self.settings.openai_model
        self.known_oems = parse_known_oems(self.settings.known_oems)
        if llm_client is not None:
            self.llm_client: OpenAI | None = llm_client
        elif self.settings.use_openai:
            self.llm_client = OpenAI(api_key=self.settings.require_openai_key())
        else:
            self.llm_client = None
        logger.info(
            "Supervisor agent initialized (intent classifier: %s)",
            "openai" if self.llm_client else "keyword rules",
        )

    def process_message(self, state: AgentState, user_message: str) -> AgentState:
        """Update the state from one user message and set ``next_agent``.

        The caller appends the message to ``state['messages']`` before this runs.
        """
        logger.info("Supervisor processing message: %s", user_message[:100])

        # Classify first so a plain question is not mixed with older troubleshooting context
        intent = self._classify_intent(user_message, state)
        logger.info("Classified intent: %s", intent)

        if intent in ("question", "unclear"):
            logger.info("Non-troubleshooting intent; clearing diagnostic context")
            state["symptoms"] = []
            state["hypotheses"] = []
            state["diagnostic_result"] = None
            state["evidence"] = []
            state["status"] = DiagnosticStatus.IDLE

        prev_oem = state.get("equipment_oem")
        prev_model = state.get("equipment_model")

        self._extract_information(state, user_message)

        new_oem = state.get("equipment_oem")
        new_model = state.get("equipment_model")
        oem_changed = prev_oem and new_oem and prev_oem != new_oem
        model_changed = prev_model and new_model and prev_model != new_model
        if oem_changed or model_changed:
            logger.info(
                "Equipment changed from %s %s to %s %s; clearing previous results",
                prev_oem,
                prev_model,
                new_oem,
                new_model,
            )
            state["diagnostic_result"] = None
            state["action_result"] = None
            state["hypotheses"] = []
            state["symptoms"] = []

        # Uploaded media goes to the diagnostic agent for visual analysis regardless of intent
        if state.get("uploaded_media"):
            state["next_agent"] = "diagnostic"
            state["status"] = DiagnosticStatus.DIAGNOSING
            logger.info(
                "Routing %d media file(s) to the diagnostic agent", len(state["uploaded_media"])
            )
            return state

        if intent == "troubleshoot":
            state["next_agent"] = "diagnostic"
            state["status"] = DiagnosticStatus.DIAGNOSING
        elif intent in ("record", "confirm_action"):
            state["next_agent"] = "action"
        elif intent == "question":
            state["next_agent"] = "diagnostic"
            state["status"] = DiagnosticStatus.GATHERING
        else:
            state["next_agent"] = None
            state["pending_user_input"] = True
        logger.info("Next agent: %s", state["next_agent"])

        return state

    def _extract_information(self, state: AgentState, message: str) -> None:
        """Extract OEM, model and symptoms from the message into the state in place."""
        message_lower = message.lower()

        if not state.get("equipment_oem"):
            oem = find_known_oem(message, self.known_oems)
            if oem:
                state["equipment_oem"] = oem
                logger.debug("Extracted OEM from KNOWN_OEMS: %s", oem)

        if not state.get("equipment_oem"):
            oem = self._extract_oem_by_pattern(message)
            if oem:
                state["equipment_oem"] = oem
                logger.debug("Extracted OEM by pattern: %s", oem)

        if not state.get("equipment_model"):
            for candidate in re.findall(MODEL_PATTERN, message.upper()):
                has_digit = any(char.isdigit() for char in candidate)
                if len(candidate) >= 2 and candidate not in STOP_MODELS and has_digit:
                    state["equipment_model"] = candidate
                    logger.debug("Extracted model: %s", candidate)
                    break

        if "symptoms" not in state:
            state["symptoms"] = []
        for keyword in SYMPTOM_KEYWORDS:
            if keyword in message_lower and keyword not in state["symptoms"]:
                state["symptoms"].append(keyword)
                logger.debug("Extracted symptom: %s", keyword)

    @staticmethod
    def _extract_oem_by_pattern(message: str) -> str | None:
        """Find a manufacturer name from phrasing such as 'made by X' or 'X CP-200'."""
        for pattern in MANUFACTURER_PATTERNS:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                oem = match.group(1).strip()
                oem = re.sub(
                    r"\s+(mfg|manufacturing|inc|corp|corporation|ltd)\.?$",
                    "",
                    oem,
                    flags=re.IGNORECASE,
                )
                return oem.title()
        return None

    def _classify_intent(self, message: str, state: AgentState) -> Intent:
        """Classify intent with fixed rules first and the LLM for ambiguous messages."""
        message_lower = message.lower().strip()

        diagnostic_result = state.get("diagnostic_result")
        if diagnostic_result and diagnostic_result.recommended_action:
            if message_lower in CONFIRM_WORDS:
                return "confirm_action"
            if message_lower in DENY_WORDS:
                return "unclear"

        if any(symptom in message_lower for symptom in CRITICAL_SYMPTOMS):
            return "troubleshoot"

        if message_lower.startswith(("record", "save", "log")) and "session" in message_lower:
            return "record"

        if self.llm_client is None:
            return self._rule_classify_intent(message_lower)
        return self._llm_classify_intent(message, state)

    @staticmethod
    def _rule_classify_intent(message_lower: str) -> Intent:
        """Deterministic intent classifier used when no LLM is configured."""
        if any(keyword in message_lower for keyword in SYMPTOM_KEYWORDS):
            return "troubleshoot"
        if message_lower.endswith("?") or message_lower.startswith(QUESTION_PREFIXES):
            return "question"
        return "unclear"

    def _build_intent_context(self, state: AgentState) -> str:
        context_parts = []
        if state.get("equipment_oem"):
            equipment = f"{state['equipment_oem']} {state.get('equipment_model') or ''}"
            context_parts.append(f"Equipment: {equipment.strip()}")
        if state.get("symptoms"):
            context_parts.append(f"Symptoms mentioned: {', '.join(state['symptoms'])}")

        diagnostic_result = state.get("diagnostic_result")
        if diagnostic_result:
            context_parts.append("SYSTEM STATUS: Diagnostic analysis just completed.")
            context_parts.append("The user is likely responding to the diagnosis.")
            if diagnostic_result.diagnosis:
                context_parts.append(f"Diagnosis provided: {diagnostic_result.diagnosis}")
            elif diagnostic_result.hypotheses:
                context_parts.append("Hypotheses generated but no final diagnosis yet")

        return "\n".join(context_parts) if context_parts else "No context available yet"

    def _llm_classify_intent(self, message: str, state: AgentState) -> Intent:
        """Classify intent with the chat model; fall back to the keyword rules on API errors."""
        prompt = INTENT_PROMPT.format(message=message, context=self._build_intent_context(state))
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise intent classifier. "
                        "Return only the intent category.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=10,
            )
        except OpenAIError:
            logger.exception("LLM intent classification failed; using keyword rules")
            return self._rule_classify_intent(message.lower().strip())

        intent = (response.choices[0].message.content or "").strip().lower()
        if intent not in VALID_INTENTS:
            logger.warning("LLM returned invalid intent %r; using 'unclear'", intent)
            return "unclear"
        return intent

    def generate_response(self, state: AgentState) -> str:
        """Generate the reply to the user from the current state."""
        if state.get("diagnostic_result"):
            return self._format_diagnostic_response(state)

        if state.get("action_result"):
            return self._format_action_response(state)

        if not state.get("equipment_oem") and not state.get("symptoms"):
            return (
                "I'm here to help you troubleshoot equipment issues. "
                "Could you tell me:\n"
                "1. What equipment are you working on? (manufacturer and model)\n"
                "2. What problem are you experiencing?"
            )

        if state.get("equipment_oem") and not state.get("symptoms"):
            equipment = f"{state.get('equipment_oem', '')} {state.get('equipment_model') or ''}"
            equipment = equipment.strip()
            if state.get("status") == DiagnosticStatus.DIAGNOSING:
                return (
                    f"I understand you're working on a {equipment}. "
                    "What issue are you experiencing with it?"
                )
            return f"I'll search the manual for information about the {equipment}."

        return "I'm analyzing the issue. One moment."

    def _format_diagnostic_response(self, state: AgentState) -> str:
        result = state["diagnostic_result"]
        response_parts = []

        if result.observations:
            response_parts.append("**Observations:**")
            for observation in result.observations[:3]:
                response_parts.append(f"- {observation}")

        if result.hypotheses:
            response_parts.append("\n**Possible Causes:**")
            ranked = sorted(result.hypotheses, key=lambda h: h.confidence, reverse=True)
            for hypothesis in ranked[:3]:
                response_parts.append(
                    f"- {hypothesis.description} (confidence: {hypothesis.confidence:.0%})"
                )

        if result.diagnosis:
            response_parts.append(f"\n**Diagnosis:** {result.diagnosis}")

        if result.recommended_action:
            response_parts.append(f"\n**Recommended Action:**\n{result.recommended_action}")

        if result.evidence_needed:
            response_parts.append(f"\n**Additional Information Needed:**\n{result.evidence_needed}")

        if result.safety_warnings:
            response_parts.append("\n**Safety Warnings:**")
            response_parts.extend(result.safety_warnings)

        if result.diagnosis and result.recommended_action:
            response_parts.append(
                "\n\nWould you like me to record this troubleshooting session for future reference?"
            )

        return "\n".join(response_parts)

    def _format_action_response(self, state: AgentState) -> str:
        result = state["action_result"]
        if result.success:
            return (
                f"{result.details}\n\n"
                f"Session recorded with ID: {result.record_id}\n\n"
                "Is there anything else I can help you with?"
            )
        return (
            f"Action failed: {result.details}\n\n"
            "Please try again or contact support if the issue persists."
        )


def create_supervisor_agent() -> SupervisorAgent:
    """Create a supervisor agent from the current settings."""
    return SupervisorAgent()
