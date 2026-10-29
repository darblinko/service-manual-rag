"""Action agent: records troubleshooting sessions and outcomes in the knowledge graph.

Every operation that writes a record goes through this agent, which keeps writes behind
one boundary. The supervisor asks the user for confirmation before routing here.
Recording needs the optional Neo4j knowledge graph; when ``GRAPH_ENABLED`` is false the
agent returns an unsuccessful ``ActionResult`` that says so.
"""

import uuid
from datetime import datetime

from service_manual_rag.agents.components import create_graph_manager
from service_manual_rag.agents.state import ActionResult, AgentState
from service_manual_rag.config import Settings, get_settings
from service_manual_rag.knowledge_graph.neo4j_client import Neo4JManager
from service_manual_rag.utils.logging_config import get_logger

logger = get_logger(__name__)

GRAPH_DISABLED_DETAILS = (
    "The knowledge graph is disabled (GRAPH_ENABLED=false), so the session was not recorded."
)

CREATE_SESSION_QUERY = """
CREATE (s:TroubleshootingSession {
    session_id: $session_id,
    user_id: $user_id,
    equipment_oem: $equipment_oem,
    equipment_model: $equipment_model,
    symptoms: $symptoms,
    diagnosis: $diagnosis,
    confidence: $confidence,
    timestamp: datetime($timestamp),
    resolution: $resolution
})
RETURN s.session_id as id
"""

LINK_EQUIPMENT_QUERY = """
MERGE (e:Equipment {oem: $oem, model: $model})
WITH e
MATCH (s:TroubleshootingSession {session_id: $session_id})
MERGE (e)-[:HAS_SESSION]->(s)
"""

UPDATE_OUTCOME_QUERY = """
MATCH (s:TroubleshootingSession {session_id: $session_id})
SET s.outcome_success = $outcome_success,
    s.outcome_timestamp = datetime($timestamp),
    s.outcome_notes = $notes
RETURN s.session_id as id
"""


class ActionAgent:
    """Writes troubleshooting sessions and outcomes to the knowledge graph."""

    def __init__(
        self,
        neo4j_manager: Neo4JManager | None = None,
        settings: Settings | None = None,
    ):
        """
        Args:
            neo4j_manager: Knowledge graph manager. When omitted, one is created only if
                ``settings.graph_enabled`` is true.
            settings: Application settings; defaults to ``get_settings()``.
        """
        settings = settings or get_settings()
        self.kg = neo4j_manager if neo4j_manager is not None else create_graph_manager(settings)
        logger.info("Action agent initialized (graph=%s)", "on" if self.kg else "off")

    def record_session(self, state: AgentState) -> ActionResult:
        """Record the session as a TroubleshootingSession node for use in later cases."""
        session_id = state.get("session_id") or str(uuid.uuid4())
        logger.info("Recording troubleshooting session: %s", session_id)

        if self.kg is None:
            logger.info("Skipping session recording: knowledge graph disabled")
            return ActionResult("record_session", success=False, details=GRAPH_DISABLED_DETAILS)

        diag_result = state.get("diagnostic_result")
        if not diag_result:
            return ActionResult(
                "record_session", success=False, details="No diagnostic result to record"
            )

        equipment_oem = state.get("equipment_oem") or "Unknown"
        equipment_model = state.get("equipment_model") or "Unknown"
        hypotheses = diag_result.hypotheses or []
        params = {
            "session_id": session_id,
            "user_id": state.get("user_id", "unknown"),
            "equipment_oem": equipment_oem,
            "equipment_model": equipment_model,
            "symptoms": state.get("symptoms", []),
            "diagnosis": diag_result.diagnosis or "Diagnosis pending",
            "confidence": max((h.confidence for h in hypotheses), default=0.5),
            "timestamp": datetime.now().isoformat(),
            "resolution": diag_result.recommended_action or "No resolution yet",
        }

        try:
            result = self.kg.execute_query(CREATE_SESSION_QUERY, params, write=True)
            if not result:
                return ActionResult(
                    "record_session",
                    success=False,
                    details="The knowledge graph returned no session node",
                )
            record_id = result[0].get("id", session_id)
            self.kg.execute_query(
                LINK_EQUIPMENT_QUERY,
                {"oem": equipment_oem, "model": equipment_model, "session_id": session_id},
                write=True,
            )
        except Exception as error:
            logger.exception("Failed to record session %s", session_id)
            return ActionResult(
                "record_session", success=False, details=f"Failed to record session: {error}"
            )

        logger.info("Troubleshooting session recorded: %s", record_id)
        return ActionResult(
            "record_session",
            success=True,
            details="Troubleshooting session recorded successfully",
            record_id=record_id,
        )

    def record_outcome(
        self,
        session_id: str,
        outcome_success: bool,
        symptom_description: str,
        solution_description: str,
        equipment_oem: str | None = None,
        equipment_model: str | None = None,
        notes: str | None = None,
    ) -> ActionResult:
        """Record the outcome of a session and update knowledge graph weights.

        A successful outcome strengthens the symptom to solution path in the graph, so
        later diagnoses rank that solution higher. A failed outcome is stored on the
        session node and does not change weights.

        Args:
            session_id: Troubleshooting session ID
            outcome_success: True if the solution resolved the problem
            symptom_description: Description of the symptom, for example "motor overheating"
            solution_description: Description of the solution, for example "clean cooling fan"
            equipment_oem: Optional equipment OEM
            equipment_model: Optional equipment model
            notes: Optional notes about the outcome

        Example:
            >>> agent.record_outcome(
            ...     session_id="sess_123",
            ...     outcome_success=True,
            ...     symptom_description="discharge pressure low",
            ...     solution_description="replace worn impeller",
            ...     equipment_oem="ExampleCo",
            ...     equipment_model="CP-200",
            ... )
        """
        outcome_label = "SUCCESS" if outcome_success else "FAILURE"
        logger.info("Recording outcome for session %s: %s", session_id, outcome_label)

        if self.kg is None:
            logger.info("Skipping outcome recording: knowledge graph disabled")
            return ActionResult("record_outcome", success=False, details=GRAPH_DISABLED_DETAILS)

        equipment_id = None
        if equipment_oem and equipment_model:
            equipment_id = f"{equipment_oem.upper()}_{equipment_model.upper()}"

        params = {
            "session_id": session_id,
            "outcome_success": outcome_success,
            "timestamp": datetime.now().isoformat(),
            "notes": notes or "",
        }
        path = f"'{symptom_description}' to '{solution_description}'"

        try:
            result = self.kg.execute_query(UPDATE_OUTCOME_QUERY, params, write=True)
            if not result:
                logger.warning("Session %s not found; continuing with weight update", session_id)

            if outcome_success:
                weight_updated = self.kg.strengthen_diagnostic_path(
                    symptom_description=symptom_description,
                    solution_description=solution_description,
                    equipment_id=equipment_id,
                )
        except Exception as error:
            logger.exception("Failed to record outcome for session %s", session_id)
            return ActionResult(
                "record_outcome", success=False, details=f"Failed to record outcome: {error}"
            )

        if not outcome_success:
            details = (
                f"Outcome recorded: FAILURE. Solution '{solution_description}' did not "
                f"resolve '{symptom_description}'. The outcome is stored for later analysis."
            )
        elif weight_updated:
            details = (
                f"Outcome recorded: SUCCESS. Strengthened diagnostic path {path}. "
                "Later diagnoses will rank this solution higher."
            )
        else:
            details = (
                "Outcome recorded: SUCCESS. The path was not strengthened because the "
                "symptom or solution is not in the graph."
            )
        logger.info("Outcome recorded for session %s", session_id)

        return ActionResult("record_outcome", success=True, details=details, record_id=session_id)


def create_action_agent() -> ActionAgent:
    """Create an action agent from the current settings."""
    return ActionAgent()
