"""Every test runs offline: no API keys, embedded Qdrant in a temp dir, no graph."""

import pytest

from service_manual_rag.config import get_settings
from service_manual_rag.vector_db import qdrant_client

CREDENTIAL_VARIABLES = ("OPENAI_API_KEY", "GOOGLE_API_KEY", "NEO4J_PASSWORD")


@pytest.fixture(autouse=True)
def offline_settings(monkeypatch, tmp_path):
    for name in CREDENTIAL_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_MODE", "offline")
    monkeypatch.setenv("GRAPH_ENABLED", "false")
    monkeypatch.setenv("QDRANT_MODE", "local")
    monkeypatch.setenv("QDRANT_LOCAL_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    qdrant_client._cached_client.cache_clear()
    yield get_settings()
    get_settings.cache_clear()
    qdrant_client._cached_client.cache_clear()
