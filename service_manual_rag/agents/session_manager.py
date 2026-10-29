"""Agent session storage in local JSON files with an in-memory cache."""

import json
import uuid
from datetime import datetime
from pathlib import Path

from service_manual_rag.agents.state import (
    ActionResult,
    AgentState,
    DiagnosticResult,
    DiagnosticStatus,
    Evidence,
    Hypothesis,
    create_initial_state,
)
from service_manual_rag.utils.logging_config import get_logger

logger = get_logger(__name__)

DATETIME_FIELDS = ("session_start", "last_updated")


class LocalSessionStore:
    """Stores one JSON file per session in a local directory."""

    def __init__(self, storage_dir: str = "data/agent_sessions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Session store initialized: %s", self.storage_dir)

    def _session_file(self, session_id: str) -> Path:
        return self.storage_dir / f"{session_id}.json"

    def save_session(self, session_id: str, state: AgentState) -> None:
        """Write the session state to its JSON file.

        Raises:
            OSError: The file could not be written.
            TypeError: The state holds a value that cannot be serialized to JSON.
        """
        serializable_state = self._make_serializable(state)
        try:
            with open(self._session_file(session_id), "w", encoding="utf-8") as handle:
                json.dump(serializable_state, handle, indent=2)
        except (OSError, TypeError) as error:
            logger.error("Failed to save session %s: %s", session_id, error)
            raise
        logger.debug("Session saved: %s", session_id)

    def load_session(self, session_id: str) -> AgentState | None:
        """Load a session state; return None when no file exists for the session.

        Raises:
            OSError: The file exists but could not be read.
            ValueError: The file does not hold a valid session state.
        """
        session_file = self._session_file(session_id)
        if not session_file.exists():
            logger.warning("Session not found: %s", session_id)
            return None

        try:
            with open(session_file, encoding="utf-8") as handle:
                state = self._deserialize_state(json.load(handle))
        except (OSError, ValueError, KeyError) as error:
            logger.error("Failed to load session %s: %s", session_id, error)
            raise
        logger.debug("Session loaded: %s", session_id)
        return state

    def delete_session(self, session_id: str) -> bool:
        """Delete a session file; return False when it did not exist."""
        session_file = self._session_file(session_id)
        if not session_file.exists():
            return False
        session_file.unlink()
        logger.info("Session deleted: %s", session_id)
        return True

    @staticmethod
    def _make_serializable(state: AgentState) -> dict:
        """Convert dataclasses, enums and datetimes in the state to JSON types."""
        serializable = dict(state)

        for field_name in DATETIME_FIELDS:
            if isinstance(serializable.get(field_name), datetime):
                serializable[field_name] = serializable[field_name].isoformat()

        for list_field in ("evidence", "hypotheses"):
            if list_field in serializable:
                serializable[list_field] = [
                    item.to_dict() if hasattr(item, "to_dict") else item
                    for item in serializable[list_field]
                ]

        for result_field in ("diagnostic_result", "action_result"):
            value = serializable.get(result_field)
            if hasattr(value, "to_dict"):
                serializable[result_field] = value.to_dict()

        if "status" in serializable:
            status = serializable["status"]
            serializable["status"] = status.value if hasattr(status, "value") else str(status)

        return serializable

    @staticmethod
    def _deserialize_state(state_dict: dict) -> AgentState:
        """Rebuild datetimes, enums and dataclasses from a loaded JSON dict."""
        state = dict(state_dict)

        for field_name in DATETIME_FIELDS:
            if isinstance(state.get(field_name), str):
                state[field_name] = datetime.fromisoformat(state[field_name])

        if isinstance(state.get("status"), str):
            state["status"] = DiagnosticStatus(state["status"])

        state["evidence"] = [Evidence.from_dict(item) for item in state.get("evidence", [])]
        state["hypotheses"] = [Hypothesis.from_dict(item) for item in state.get("hypotheses", [])]
        if state.get("diagnostic_result"):
            state["diagnostic_result"] = DiagnosticResult.from_dict(state["diagnostic_result"])
        if state.get("action_result"):
            state["action_result"] = ActionResult.from_dict(state["action_result"])

        return AgentState(**state)


class SessionManager:
    """Session manager with an in-memory cache in front of the local store."""

    def __init__(self, store: LocalSessionStore | None = None):
        self.store = store or LocalSessionStore()
        self._cache: dict[str, AgentState] = {}

    def create_session(self, user_id: str = "default_user") -> str:
        """Create and persist a new session; return its ID."""
        session_id = str(uuid.uuid4())
        state = create_initial_state(session_id, user_id)
        self.store.save_session(session_id, state)
        self._cache[session_id] = state
        logger.info("Created session %s for user %s", session_id, user_id)
        return session_id

    def get_session(self, session_id: str) -> AgentState | None:
        """Return the session state from the cache or the store, or None if unknown."""
        if session_id in self._cache:
            return self._cache[session_id]

        state = self.store.load_session(session_id)
        if state:
            self._cache[session_id] = state
        return state

    def update_session(self, session_id: str, state: AgentState) -> None:
        """Update the cached state and persist it."""
        state["last_updated"] = datetime.now()
        self._cache[session_id] = state
        self.store.save_session(session_id, state)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session from the cache and the store."""
        self._cache.pop(session_id, None)
        return self.store.delete_session(session_id)


_session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Return the process-wide session manager, creating it on first use."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
