"""Embedding services: OpenAI embeddings with a disk cache, and an offline hashing embedder."""

import hashlib
import json
import math
import time
from itertools import pairwise
from pathlib import Path
from typing import Protocol

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from service_manual_rag.config import Settings, get_settings
from service_manual_rag.utils.logging_config import get_logger
from service_manual_rag.utils.text import STOPWORDS, tokenize

logger = get_logger(__name__)


class Embedder(Protocol):
    """Interface shared by every embedding service."""

    model: str
    dimensions: int

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]: ...

    def get_cache_stats(self) -> dict: ...


class HashingEmbeddingService:
    """Deterministic feature-hashing embedder that needs no network or model download.

    Stopwords are dropped; the remaining unigrams and bigrams are hashed with
    BLAKE2b into a fixed number of signed buckets, weighted by
    1 + log(term frequency), and L2-normalized. It captures
    lexical overlap only, so it is a stand-in for semantic embeddings in tests
    and offline demos.
    """

    def __init__(self, dimensions: int = 512):
        if dimensions < 16:
            raise ValueError("dimensions must be at least 16")
        self.model = f"hashing-bigram-{dimensions}"
        self.dimensions = dimensions

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        sign = 1.0 if value & 1 else -1.0
        return (value >> 1) % self.dimensions, sign

    def embed(self, text: str) -> list[float]:
        tokens = [token for token in tokenize(text) if token not in STOPWORDS]
        features = tokens + [f"{a} {b}" for a, b in pairwise(tokens)]
        counts: dict[str, int] = {}
        for feature in features:
            counts[feature] = counts.get(feature, 0) + 1

        vector = [0.0] * self.dimensions
        for feature, count in counts.items():
            index, sign = self._bucket(feature)
            vector[index] += sign * (1.0 + math.log(count))

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            return vector
        return [v / norm for v in vector]

    def embed_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def get_cache_stats(self) -> dict:
        return {"cache_files": 0, "total_size": 0, "total_size_mb": 0.0}


class EmbeddingService:
    """OpenAI embedding service with a per-text disk cache and a requests-per-minute limit."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        cache_dir: Path | None = None,
        use_cache: bool = True,
        settings: Settings | None = None,
    ):
        settings = settings or get_settings()
        self.model = model or settings.openai_embedding_model
        self.dimensions = dimensions or settings.openai_embedding_dimensions
        self.cache_dir = cache_dir or settings.cache_dir / "embeddings"
        self.use_cache = use_cache
        self.client = OpenAI(api_key=api_key or settings.require_openai_key())
        self.rpm_limit = settings.openai_rpm
        self._request_times: list[float] = []

        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized EmbeddingService: model={self.model}, dim={self.dimensions}")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True
    )
    def embed(self, text: str) -> list[float]:
        cached = self._get_cached_embedding(text)
        if cached:
            return cached

        self._wait_for_rate_limit()
        response = self.client.embeddings.create(
            input=text, model=self.model, dimensions=self.dimensions
        )
        embedding = response.data[0].embedding
        self._cache_embedding(text, embedding)
        self._request_times.append(time.time())
        return embedding

    def embed_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        logger.info(f"Creating embeddings for {len(texts)} texts")
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            embeddings.extend(self._embed_one_batch(texts[start : start + batch_size]))
        logger.info(f"Created {len(embeddings)} embeddings")
        return embeddings

    def _embed_one_batch(self, batch: list[str]) -> list[list[float]]:
        results: dict[int, list[float]] = {}
        uncached_indices = []
        for index, text in enumerate(batch):
            cached = self._get_cached_embedding(text)
            if cached:
                results[index] = cached
            else:
                uncached_indices.append(index)

        if uncached_indices:
            self._wait_for_rate_limit()
            response = self.client.embeddings.create(
                input=[batch[i] for i in uncached_indices],
                model=self.model,
                dimensions=self.dimensions,
            )
            for index, item in zip(uncached_indices, response.data):
                results[index] = item.embedding
                self._cache_embedding(batch[index], item.embedding)
            self._request_times.append(time.time())

        return [results[i] for i in range(len(batch))]

    def _wait_for_rate_limit(self) -> None:
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 60]
        if len(self._request_times) >= self.rpm_limit:
            sleep_time = 60 - (now - self._request_times[0])
            if sleep_time > 0:
                logger.debug(f"Rate limit: waiting {sleep_time:.1f}s")
                time.sleep(sleep_time)

    def _cache_file(self, text: str) -> Path:
        key = hashlib.sha256(f"{text}|{self.model}|{self.dimensions}".encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _get_cached_embedding(self, text: str) -> list[float] | None:
        if not self.use_cache:
            return None
        cache_file = self._cache_file(text)
        if not cache_file.exists():
            return None
        try:
            return json.loads(cache_file.read_text())["embedding"]
        except (OSError, ValueError, KeyError) as error:
            logger.warning(f"Ignoring unreadable cache file {cache_file}: {error}")
            return None

    def _cache_embedding(self, text: str, embedding: list[float]) -> None:
        if not self.use_cache:
            return
        data = {"model": self.model, "dimensions": self.dimensions, "embedding": embedding}
        self._cache_file(text).write_text(json.dumps(data))

    def clear_cache(self) -> int:
        """Delete cached embeddings and return the number of files removed."""
        if not self.cache_dir.exists():
            return 0
        files = list(self.cache_dir.glob("*.json"))
        for cache_file in files:
            cache_file.unlink()
        logger.info(f"Cleared {len(files)} cache files")
        return len(files)

    def get_cache_stats(self) -> dict:
        if not self.cache_dir.exists():
            return {"cache_files": 0, "total_size": 0, "total_size_mb": 0.0}
        files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in files)
        return {
            "cache_files": len(files),
            "total_size": total_size,
            "total_size_mb": total_size / (1024 * 1024),
        }


def create_embedding_service(settings: Settings | None = None) -> Embedder:
    """Return the OpenAI embedder when settings allow it, else the offline hashing embedder."""
    settings = settings or get_settings()
    if settings.use_openai:
        return EmbeddingService(settings=settings)
    return HashingEmbeddingService(dimensions=settings.hash_embedding_dimensions)
