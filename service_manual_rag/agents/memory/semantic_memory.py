"""Conversation memory: stores conversation turns as embeddings and retrieves similar cases.

Turns are stored in the Qdrant collection ``conversation_memory`` through the shared
client from ``get_qdrant_client()``, so the store works with embedded local Qdrant and
the offline hashing embedder.
"""

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from service_manual_rag.embeddings import Embedder, create_embedding_service
from service_manual_rag.utils.logging_config import get_logger
from service_manual_rag.vector_db.qdrant_client import get_qdrant_client

logger = get_logger(__name__)


@dataclass
class ConversationTurn:
    """A single conversation turn with metadata."""

    turn_id: str
    session_id: str
    timestamp: datetime
    role: str  # "user" or "assistant"
    content: str
    equipment_oem: str | None = None
    equipment_model: str | None = None
    symptoms: list[str] = field(default_factory=list)
    resolution_success: bool | None = None
    diagnostic_result: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        """Convert to a dictionary for storage."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationTurn":
        """Create from a stored dictionary."""
        data = data.copy()
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class ConversationSession:
    """Complete conversation session with metadata."""

    session_id: str
    user_id: str
    equipment_oem: str | None
    equipment_model: str | None
    symptoms: list[str]
    resolution_success: bool
    resolution_summary: str | None = None
    turns: list[ConversationTurn] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    success_rate: float = 0.0

    def to_dict(self) -> dict:
        """Convert to a dictionary for storage."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["turns"] = [turn.to_dict() for turn in self.turns]
        return data


def turn_point_id(turn_id: str) -> str:
    """Return a Qdrant point ID that is stable for a turn ID across processes."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"conversation-turn:{turn_id}"))


class ConversationMemoryStore:
    """Stores conversation turns in Qdrant and searches them by semantic similarity."""

    COLLECTION_NAME = "conversation_memory"

    def __init__(
        self,
        qdrant_client: QdrantClient | None = None,
        embedding_generator: Embedder | None = None,
    ):
        """Connect to Qdrant and create the collection if it does not exist.

        Args:
            qdrant_client: Qdrant client; defaults to the shared ``get_qdrant_client()``.
            embedding_generator: Embedder; defaults to ``create_embedding_service()``.
        """
        self.client = qdrant_client or get_qdrant_client()
        self.embedder = embedding_generator or create_embedding_service()
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the conversation memory collection sized to the embedder."""
        if self.client.collection_exists(self.COLLECTION_NAME):
            return
        logger.info(
            "Creating collection %s with %d dimensions",
            self.COLLECTION_NAME,
            self.embedder.dimensions,
        )
        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(size=self.embedder.dimensions, distance=Distance.COSINE),
        )

    def store_conversation_turn(
        self,
        turn: ConversationTurn,
        embedding: list[float] | None = None,
    ) -> str:
        """Store a conversation turn and return its point ID.

        Args:
            turn: Conversation turn to store
            embedding: Optional precomputed embedding of ``turn.content``
        """
        if embedding is None:
            embedding = self.embedder.embed(turn.content)

        point_id = turn_point_id(turn.turn_id)
        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[PointStruct(id=point_id, vector=embedding, payload=turn.to_dict())],
        )
        logger.debug("Stored conversation turn: %s", turn.turn_id)
        return point_id

    def store_conversation_session(self, session: ConversationSession) -> None:
        """Store every turn of a session, copying the session metadata onto each turn."""
        logger.info("Storing session %s with %d turns", session.session_id, len(session.turns))
        for turn in session.turns:
            turn.equipment_oem = session.equipment_oem
            turn.equipment_model = session.equipment_model
            turn.symptoms = session.symptoms
            turn.resolution_success = session.resolution_success
            self.store_conversation_turn(turn)

    def search_similar_conversations(
        self,
        query: str,
        equipment: str | None = None,
        symptoms: list[str] | None = None,
        successful_only: bool = True,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search for past conversation turns similar to the query.

        Args:
            query: Query text or conversation context
            equipment: "<OEM> <model>", for example "ExampleCo CP-200"; the first word is
                matched against the OEM and the rest against the model
            symptoms: Current symptoms; accepted for interface compatibility and not
                applied as a filter
            successful_only: Only return turns from successful resolutions
            limit: Maximum number of results

        Returns:
            List of similar conversation turns with metadata

        Example:
            >>> store = ConversationMemoryStore()
            >>> similar = store.search_similar_conversations(
            ...     "pump discharge pressure dropping",
            ...     equipment="ExampleCo CP-200",
            ...     successful_only=True,
            ...     limit=5,
            ... )
        """
        query_embedding = self.embedder.embed(query)
        search_filter = self._build_filter(equipment, successful_only)

        results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_embedding,
            query_filter=search_filter,
            limit=limit,
        ).points

        formatted_results = []
        for result in results:
            payload = result.payload or {}
            formatted_results.append(
                {
                    "score": result.score,
                    "content": payload.get("content"),
                    "session_id": payload.get("session_id"),
                    "timestamp": payload.get("timestamp"),
                    "equipment": (
                        f"{payload.get('equipment_oem')} {payload.get('equipment_model')}"
                    ),
                    "symptoms": payload.get("symptoms", []),
                    "resolution_success": payload.get("resolution_success"),
                    "diagnostic_result": payload.get("diagnostic_result"),
                }
            )

        logger.debug("Found %d similar conversations", len(formatted_results))
        return formatted_results

    @staticmethod
    def _build_filter(equipment: str | None, successful_only: bool) -> Filter | None:
        must_conditions = []
        if equipment:
            parts = equipment.split()
            if len(parts) >= 2:
                must_conditions.append(
                    FieldCondition(key="equipment_oem", match=MatchValue(value=parts[0]))
                )
                must_conditions.append(
                    FieldCondition(
                        key="equipment_model", match=MatchValue(value=" ".join(parts[1:]))
                    )
                )
        if successful_only:
            must_conditions.append(
                FieldCondition(key="resolution_success", match=MatchValue(value=True))
            )
        return Filter(must=must_conditions) if must_conditions else None

    def get_successful_resolutions(
        self,
        equipment: str | None = None,
        symptoms: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return turns from successful resolutions, ranked against the symptoms."""
        query = "successful resolution"
        if symptoms:
            query = f"{query} {' '.join(symptoms)}"

        return self.search_similar_conversations(
            query=query,
            equipment=equipment,
            symptoms=symptoms,
            successful_only=True,
            limit=limit,
        )


class SemanticMemory:
    """Retrieves relevant past conversations and stores completed sessions."""

    def __init__(self, memory_store: ConversationMemoryStore | None = None):
        """
        Args:
            memory_store: Conversation memory store; one is created when omitted.
        """
        self.store = memory_store or ConversationMemoryStore()

    def retrieve_for_context(
        self,
        current_query: str,
        equipment_oem: str | None = None,
        equipment_model: str | None = None,
        symptoms: list[str] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Retrieve past successful conversations relevant to the current context.

        Args:
            current_query: Current user query or conversation context
            equipment_oem: Equipment manufacturer
            equipment_model: Equipment model
            symptoms: Current symptoms
            top_k: Number of past conversations to retrieve

        Example:
            >>> memory = SemanticMemory()
            >>> past_cases = memory.retrieve_for_context(
            ...     "compressor discharge temperature high",
            ...     equipment_oem="ExampleCo",
            ...     equipment_model="RSC-15",
            ...     symptoms=["overheating"],
            ...     top_k=3,
            ... )
        """
        equipment = None
        if equipment_oem and equipment_model:
            equipment = f"{equipment_oem} {equipment_model}"

        return self.store.search_similar_conversations(
            query=current_query,
            equipment=equipment,
            symptoms=symptoms,
            successful_only=True,
            limit=top_k,
        )

    def store_session_result(
        self,
        session_id: str,
        user_id: str,
        messages: list[dict[str, str]],
        equipment_oem: str | None = None,
        equipment_model: str | None = None,
        symptoms: list[str] | None = None,
        resolution_success: bool = False,
        resolution_summary: str | None = None,
    ) -> None:
        """Store a completed session in memory.

        Args:
            session_id: Session identifier
            user_id: User identifier
            messages: List of {role, content} conversation messages
            equipment_oem: Equipment manufacturer
            equipment_model: Equipment model
            symptoms: List of symptoms
            resolution_success: Whether the resolution was successful
            resolution_summary: Summary of the resolution
        """
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            equipment_oem=equipment_oem,
            equipment_model=equipment_model,
            symptoms=symptoms or [],
            resolution_success=resolution_success,
            resolution_summary=resolution_summary,
        )
        for index, message in enumerate(messages):
            session.turns.append(
                ConversationTurn(
                    turn_id=f"{session_id}_turn_{index}",
                    session_id=session_id,
                    timestamp=datetime.now(),
                    role=message.get("role", "user"),
                    content=message.get("content", ""),
                )
            )

        self.store.store_conversation_session(session)
        logger.info("Stored session result: %s", session_id)

    def format_for_retrieval(self, past_conversations: list[dict[str, Any]]) -> str:
        """Format past conversations as text for the RAG context."""
        if not past_conversations:
            return "No similar past cases found."

        parts = ["SIMILAR PAST CASES:"]
        for index, conv in enumerate(past_conversations, 1):
            parts.append(f"\nCase {index} (Similarity: {conv['score']:.2f}):")
            parts.append(f"  Equipment: {conv.get('equipment', 'Unknown')}")
            parts.append(f"  Symptoms: {', '.join(conv.get('symptoms', []))}")

            diag = conv.get("diagnostic_result")
            if isinstance(diag, dict):
                if diag.get("diagnosis"):
                    parts.append(f"  Diagnosis: {diag['diagnosis']}")
                if diag.get("recommended_action"):
                    parts.append(f"  Resolution: {diag['recommended_action']}")

            parts.append(f"  Success: {'Yes' if conv.get('resolution_success') else 'No'}")

        return "\n".join(parts)
