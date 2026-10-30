import math

import pytest

from service_manual_rag.config import MissingCredentialError, get_settings
from service_manual_rag.embeddings import (
    EmbeddingService,
    HashingEmbeddingService,
    VectorIndexer,
    create_embedding_service,
)
from service_manual_rag.processing import Chunk, ChunkLevel
from service_manual_rag.rag import ExtractiveAnswerGenerator, FusionEngine, HybridRetriever
from service_manual_rag.rag.answering import NO_CONTEXT_ANSWER
from service_manual_rag.vector_db import QdrantManager


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hashing_embedder_is_deterministic_and_normalized():
    embedder = HashingEmbeddingService(dimensions=64)
    vector = embedder.embed("Replace the impeller seal")
    assert vector == embedder.embed("Replace the impeller seal")
    assert len(vector) == 64
    assert math.isclose(sum(v * v for v in vector), 1.0, rel_tol=1e-9)
    assert embedder.embed("") == [0.0] * 64


def test_hashing_embedder_ranks_lexical_overlap_higher():
    embedder = HashingEmbeddingService()
    query = embedder.embed("impeller nut torque")
    related = embedder.embed("Tighten the impeller nut to the listed torque")
    unrelated = embedder.embed("Check the motor insulation resistance")
    assert cosine(query, related) > cosine(query, unrelated)


def test_offline_mode_selects_the_hashing_embedder():
    assert isinstance(create_embedding_service(), HashingEmbeddingService)


def test_openai_embedder_without_key_raises_a_clear_error(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "openai")
    get_settings.cache_clear()
    with pytest.raises(MissingCredentialError, match="OPENAI_API_KEY"):
        EmbeddingService()


def test_neo4j_password_is_required_only_when_asked_for():
    with pytest.raises(MissingCredentialError, match="NEO4J_PASSWORD"):
        get_settings().require_neo4j_password()


def make_chunk(number, text):
    return Chunk(
        chunk_id=f"00000000-0000-0000-0000-{number:012d}",
        level=ChunkLevel.SEMANTIC,
        content=text,
        oem="ExampleCo",
        model="T-1",
        section_name=f"Section {number}",
        doc_id="doc",
        token_count=len(text.split()),
    )


def test_indexer_round_trip_on_embedded_qdrant():
    indexer = VectorIndexer()
    chunks = [
        make_chunk(1, "Tighten the impeller nut to 45 N m."),
        make_chunk(2, "Measure motor insulation resistance with a 500 V tester."),
    ]
    stats = indexer.index_chunks(chunks)
    assert stats["indexed_chunks"] == 2

    results = indexer.search("impeller nut torque", limit=2)
    assert results[0]["payload"]["section_name"] == "Section 1"
    assert indexer.get_statistics()["qdrant"]["points_count"] == 2


def test_reindexing_a_document_removes_its_superseded_chunks():
    indexer = VectorIndexer()
    indexer.index_chunks([make_chunk(1, "Tighten the impeller nut to 45 N m.")])
    other = make_chunk(9, "Measure motor insulation resistance.")
    other.doc_id = "other-doc"
    indexer.index_chunks([other])

    stats = indexer.index_chunks([make_chunk(2, "Tighten the impeller nut to 55 N m.")])

    texts = [hit["payload"]["chunk_text"] for hit in indexer.search("impeller nut", limit=10)]
    assert "Tighten the impeller nut to 55 N m." in texts
    assert "Tighten the impeller nut to 45 N m." not in texts
    assert "Measure motor insulation resistance." in texts
    assert stats["removed_chunks"] == 1


def test_create_collection_rejects_a_different_vector_size():
    manager = QdrantManager()
    manager.create_collection(vector_size=32)
    with pytest.raises(ValueError):
        manager.create_collection(vector_size=64)


def test_fusion_rewards_documents_found_by_both_retrievers():
    vector = [
        {"id": "a", "score": 0.9, "payload": {"chunk_text": "A"}},
        {"id": "b", "score": 0.8, "payload": {"chunk_text": "B"}},
    ]
    graph = [{"node": {"id": "b"}, "score": 3.0, "types": ["Procedure"]}]
    fused = FusionEngine(k=10).fuse(vector, graph, vector_weight=0.6, graph_weight=0.4)

    assert [entry["id"] for entry in fused] == ["b", "a"]
    assert fused[0]["source_type"] == "hybrid"
    assert math.isclose(fused[0]["rrf_score"], 0.6 / 12 + 0.4 / 11)


def test_extractive_answer_cites_only_retrieved_text():
    passages = [
        {"content": "Tighten the impeller nut to 45 N m. Use a calibrated wrench."},
        {"content": "Measure insulation resistance before start-up."},
    ]
    answer = ExtractiveAnswerGenerator(max_sentences=1).generate("impeller nut torque", passages)
    assert answer == "Tighten the impeller nut to 45 N m. [Source 1]"


def test_extractive_answer_reports_missing_context():
    assert ExtractiveAnswerGenerator().generate("coolant", []) == NO_CONTEXT_ANSWER


class FakeVectorIndexer:
    def __init__(self):
        self.calls = 0

    def search(self, query, limit=10, filters=None):
        self.calls += 1
        return [{"id": "v1", "score": 0.9, "payload": {"chunk_text": "vector passage"}}]


class FakeGraph:
    def __init__(self):
        self.calls = 0

    def execute_query(self, query, parameters):
        self.calls += 1
        return [{"node": {"id": "g1"}, "score": 2.0, "types": ["Procedure"]}]


@pytest.mark.parametrize(
    ("mode", "vector_calls", "graph_calls"),
    [("vector", 1, 0), ("graph", 0, 1), ("fused", 1, 1)],
)
def test_retrieval_mode_selects_the_retrievers(monkeypatch, mode, vector_calls, graph_calls):
    monkeypatch.setenv("RETRIEVAL_MODE", mode)
    get_settings.cache_clear()
    vector, graph = FakeVectorIndexer(), FakeGraph()
    retriever = HybridRetriever(vector_indexer=vector, neo4j_manager=graph)

    result = retriever.retrieve("impeller nut torque")

    assert retriever.effective_mode == mode
    assert (vector.calls, graph.calls) == (vector_calls, graph_calls)
    assert len(result["vector_results"]) == vector_calls
    assert len(result["graph_results"]) == graph_calls


def test_fused_mode_without_graph_is_vector_only():
    retriever = HybridRetriever(vector_indexer=FakeVectorIndexer())
    assert retriever.effective_mode == "vector"


def test_graph_mode_without_graph_enabled_raises(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MODE", "graph")
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="GRAPH_ENABLED"):
        HybridRetriever(vector_indexer=FakeVectorIndexer())
