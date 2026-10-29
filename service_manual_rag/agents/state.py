"""Shared state that flows through the supervisor, diagnostic and action agent graph."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TypedDict


class DiagnosticStatus(str, Enum):
    """Current status of the diagnostic session."""

    IDLE = "idle"
    GATHERING = "gathering"
    DIAGNOSING = "diagnosing"
    CONFIRMING = "confirming"
    RESOLVED = "resolved"
    FAILED = "failed"


@dataclass
class Evidence:
    """Evidence collected during diagnosis."""

    evidence_type: str  # 'text', 'image', 'video', 'telemetry', 'audio'
    content: str  # Text description or file path
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "evidence_type": self.evidence_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Evidence":
        return cls(
            evidence_type=data["evidence_type"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Hypothesis:
    """A diagnostic hypothesis with a confidence between 0.0 and 1.0."""

    description: str
    confidence: float
    supporting_evidence: list[str] = field(default_factory=list)
    tests_suggested: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "confidence": self.confidence,
            "supporting_evidence": self.supporting_evidence,
            "tests_suggested": self.tests_suggested,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Hypothesis":
        return cls(
            description=data["description"],
            confidence=data["confidence"],
            supporting_evidence=data.get("supporting_evidence", []),
            tests_suggested=data.get("tests_suggested", []),
        )


@dataclass
class DiagnosticResult:
    """Result returned by the diagnostic agent."""

    observations: list[str]
    hypotheses: list[Hypothesis]
    diagnosis: str | None = None
    recommended_action: str | None = None
    evidence_needed: str | None = None
    safety_warnings: list[str] = field(default_factory=list)
    relevant_manual_sections: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "observations": self.observations,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "diagnosis": self.diagnosis,
            "recommended_action": self.recommended_action,
            "evidence_needed": self.evidence_needed,
            "safety_warnings": self.safety_warnings,
            "relevant_manual_sections": self.relevant_manual_sections,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DiagnosticResult":
        return cls(
            observations=data.get("observations", []),
            hypotheses=[Hypothesis.from_dict(h) for h in data.get("hypotheses", [])],
            diagnosis=data.get("diagnosis"),
            recommended_action=data.get("recommended_action"),
            evidence_needed=data.get("evidence_needed"),
            safety_warnings=data.get("safety_warnings", []),
            relevant_manual_sections=data.get("relevant_manual_sections", []),
        )


@dataclass
class ActionResult:
    """Result returned by the action agent."""

    action_type: str  # 'record_session', 'record_outcome'
    success: bool
    details: str
    record_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "success": self.success,
            "details": self.details,
            "record_id": self.record_id,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ActionResult":
        return cls(
            action_type=data["action_type"],
            success=data["success"],
            details=data["details"],
            record_id=data.get("record_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


class AgentState(TypedDict, total=False):
    """State shared by every node of the agent graph and updated on each turn."""

    # Session identity
    session_id: str
    user_id: str
    equipment_oem: str | None
    equipment_model: str | None

    # Conversation: list of {role, content} messages
    messages: list[dict[str, str]]

    # Diagnostic state
    status: DiagnosticStatus
    symptoms: list[str]
    evidence: list[Evidence]
    hypotheses: list[Hypothesis]

    # Agent results, updated each turn
    diagnostic_result: DiagnosticResult | None
    action_result: ActionResult | None

    # Workflow control
    next_agent: str | None
    pending_user_input: bool
    # The supervisor skips a message it has already processed, which prevents routing loops
    last_processed_message: str | None

    # Context
    uploaded_media: list[dict[str, str]]
    session_start: datetime
    last_updated: datetime


def create_initial_state(session_id: str, user_id: str = "default_user") -> AgentState:
    """Create the initial state for a new session."""
    now = datetime.now()
    return AgentState(
        session_id=session_id,
        user_id=user_id,
        equipment_oem=None,
        equipment_model=None,
        messages=[],
        status=DiagnosticStatus.IDLE,
        symptoms=[],
        evidence=[],
        hypotheses=[],
        diagnostic_result=None,
        action_result=None,
        next_agent=None,
        pending_user_input=False,
        last_processed_message=None,
        uploaded_media=[],
        session_start=now,
        last_updated=now,
    )
