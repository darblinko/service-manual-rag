"""Application settings loaded from environment variables and an optional .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MissingCredentialError(RuntimeError):
    """Raised when a component needs a credential that is not configured."""


class OfflineModeError(MissingCredentialError):
    """Raised when LLM_MODE=offline and a component asks for a remote service."""


class Settings(BaseSettings):
    """Application settings.

    Every field maps to an upper-case environment variable of the same name,
    for example ``openai_api_key`` reads ``OPENAI_API_KEY``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM mode: "auto" uses OpenAI when OPENAI_API_KEY is set and the offline
    # deterministic components otherwise. "offline" never calls a remote API.
    llm_mode: Literal["auto", "openai", "offline"] = Field(default="auto")

    # OpenAI. The chat model serves answers, agents, entity extraction and the judge.
    # Call sites set temperature, so the default is a non-reasoning model.
    openai_api_key: str | None = Field(default=None)
    openai_model: str = Field(default="gpt-4.1-mini")
    openai_embedding_model: str = Field(default="text-embedding-3-large")
    openai_embedding_dimensions: int = Field(default=3072)
    openai_rpm: int = Field(default=10000, description="Requests per minute limit")

    # Offline hashing embedder
    hash_embedding_dimensions: int = Field(default=512)

    # Google Gemini, used only by the optional vision service
    google_api_key: str | None = Field(default=None)
    gemini_model: str = Field(default="gemini-2.5-flash")

    # Cross-encoder for the agents' re-ranker (sentence-transformers, optional)
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Qdrant: "local" stores vectors on disk through qdrant-client with no server
    qdrant_mode: Literal["local", "server"] = Field(default="local")
    qdrant_local_path: Path = Field(default=Path("data/qdrant_local"))
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_collection_name: str = Field(default="service_manuals")

    # Neo4j knowledge graph, optional
    graph_enabled: bool = Field(default=False)
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str | None = Field(default=None)
    neo4j_database: str = Field(default="neo4j")

    # Document processing. PDF_STRATEGY is the Unstructured partition strategy: hi_res
    # runs a layout model (downloaded on first use) and infers table structure; fast
    # reads the PDF text layer only.
    pdf_strategy: Literal["hi_res", "fast", "ocr_only", "auto"] = Field(default="hi_res")
    chunk_size_tokens: int = Field(default=500)
    chunk_overlap_tokens: int = Field(default=50)
    min_chunk_size_tokens: int = Field(default=100)
    max_chunk_size_tokens: int = Field(default=1000)

    # Retrieval and fusion
    vector_top_k: int = Field(default=10)
    graph_top_k: int = Field(default=10)
    final_top_k: int = Field(default=5)
    vector_weight: float = Field(default=0.6)
    graph_weight: float = Field(default=0.4)
    rrf_k: int = Field(default=10)
    # vector: Qdrant only. graph: Neo4j full-text search only, needs GRAPH_ENABLED=true.
    # fused: both lists merged with RRF; vector only while GRAPH_ENABLED is false.
    retrieval_mode: Literal["vector", "graph", "fused"] = Field(default="fused")

    # Comma-separated manufacturer names the supervisor agent recognizes in messages
    known_oems: str = Field(default="ExampleCo")

    # Paths
    data_dir: Path = Field(default=Path("data"))
    cache_dir: Path = Field(default=Path("data/cache"))

    # Logging
    log_level: str = Field(default="INFO")
    log_file: Path | None = Field(default=None)
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    @field_validator("vector_weight", "graph_weight")
    @classmethod
    def validate_weights(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("Weights must be between 0 and 1")
        return value

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if value.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return value.upper()

    @property
    def use_openai(self) -> bool:
        """True when components should call the OpenAI API."""
        if self.llm_mode == "offline":
            return False
        if self.llm_mode == "openai":
            return True
        return bool(self.openai_api_key)

    @property
    def offline(self) -> bool:
        """True when LLM_MODE=offline, which forbids remote API calls and model downloads."""
        return self.llm_mode == "offline"

    def require_online(self, service: str) -> None:
        """Raise OfflineModeError when LLM_MODE=offline.

        Args:
            service: Name of the remote service, used in the error message.
        """
        if self.offline:
            raise OfflineModeError(
                f"{service} is not available with LLM_MODE=offline, which makes no remote "
                "API calls and downloads no models. Set LLM_MODE=auto or LLM_MODE=openai."
            )

    def require_openai_key(self, api_key: str | None = None) -> str:
        """Return the explicit key or the configured OpenAI key.

        Raises:
            OfflineModeError: LLM_MODE=offline.
            MissingCredentialError: Neither key is set.
        """
        self.require_online("The OpenAI API")
        key = api_key or self.openai_api_key
        if not key:
            raise MissingCredentialError(
                "OPENAI_API_KEY is not set. Set it in the environment or in .env, "
                "or set LLM_MODE=offline to use the deterministic offline components."
            )
        return key

    def require_neo4j_password(self) -> str:
        """Return the Neo4j password or raise a clear error naming the variable."""
        if not self.neo4j_password:
            raise MissingCredentialError(
                "NEO4J_PASSWORD is not set. The knowledge graph is optional; "
                "leave GRAPH_ENABLED=false to run with vector retrieval only."
            )
        return self.neo4j_password


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()
