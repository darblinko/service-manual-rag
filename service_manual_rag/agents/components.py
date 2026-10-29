"""Factories for the optional services the agents depend on.

Each factory reads the settings and returns None, or an offline component, when the
service is not configured. Nothing here connects to a service at import time.
"""

from openai import OpenAI

from service_manual_rag.config import Settings
from service_manual_rag.knowledge_graph.neo4j_client import Neo4JManager
from service_manual_rag.utils.logging_config import get_logger

logger = get_logger(__name__)


def create_graph_manager(settings: Settings) -> Neo4JManager | None:
    """Return a Neo4JManager when ``settings.graph_enabled`` is true, otherwise None."""
    if not settings.graph_enabled:
        logger.info("Knowledge graph disabled (GRAPH_ENABLED=false); graph steps are skipped")
        return None
    return Neo4JManager()


def create_openai_client(settings: Settings) -> OpenAI | None:
    """Return an OpenAI client when ``settings.use_openai`` is true, otherwise None.

    Raises:
        MissingCredentialError: OpenAI is required by LLM_MODE but OPENAI_API_KEY is not set.
    """
    if not settings.use_openai:
        return None
    return OpenAI(api_key=settings.require_openai_key())
