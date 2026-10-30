"""Population to retrieval contract of the knowledge graph.

The in-memory tests use a Neo4JManager stand-in that stores nodes and relationships.
test_neo4j_round_trip runs the same contract against a real Neo4j when
SMR_TEST_NEO4J_URI and SMR_TEST_NEO4J_PASSWORD name a throwaway database; it clears it.
"""

import os

import pytest

from service_manual_rag.graph_builder import GraphPopulator
from service_manual_rag.knowledge_graph.graph_schema import create_indexes
from service_manual_rag.processing.hierarchical_chunker import Chunk
from service_manual_rag.rag.retriever import HybridRetriever
from ui.utils.document_manager import DocumentManager


class InMemoryGraph:
    """Neo4JManager stand-in for the node, relationship and provenance calls."""

    def __init__(self):
        self.nodes = {}
        self.relationships = []
        self.queries = []

    def create_node(self, node_type, properties):
        self.nodes[properties["id"]] = {"label": node_type.value, **properties}
        return properties

    def merge_node(self, node_type, match_properties, set_properties=None):
        existing = next(
            (
                node_id
                for node_id, node in self.nodes.items()
                if node["label"] == node_type.value
                and all(node.get(k) == v for k, v in match_properties.items())
            ),
            None,
        )
        node = self.nodes.pop(existing, {"label": node_type.value})
        node = {**node, **match_properties, **(set_properties or {})}
        self.nodes[node["id"]] = node
        return node

    def create_relationship(self, from_type, from_id, rel_type, to_type, to_id, properties=None):
        if from_id not in self.nodes or to_id not in self.nodes:
            return False
        self.relationships.append((from_id, rel_type.value, to_id))
        return True

    def add_provenance(self, node_type, node_id, doc_id, equipment_id):
        node = self.nodes[node_id]
        for key, value in (("doc_ids", doc_id), ("equipment_ids", equipment_id)):
            values = node.get(key, [])
            if value and value not in values:
                node[key] = [*values, value]

    def execute_query(self, query, parameters=None, write=False):
        self.queries.append((query, parameters or {}))
        return []

    def close(self):
        pass

    def of_label(self, label):
        return [node for node in self.nodes.values() if node["label"] == label]


class StubExtractor:
    """Returns fixed entities for each semantic chunk, keyed by chunk ID."""

    def __init__(self, entities_by_chunk):
        self.entities_by_chunk = entities_by_chunk

    async def extract_entities_batch(self, chunks):
        return [self.entities_by_chunk.get(chunk.chunk_id, {}) for chunk in chunks]


TWO_PAIRS = {
    "symptoms": [
        {"description": "Gland drips", "solutions": ["Replace the mechanical seal"]},
        {"description": "Motor trips on start", "solutions": ["Reset the overload relay"]},
    ],
    "solutions": [
        {"fix_description": "Replace the mechanical seal", "parts_needed": []},
        {"fix_description": "Reset the overload relay", "parts_needed": []},
    ],
}

SEAL_PROCEDURE = {
    "procedures": [
        {
            "action_type": "replace",
            "steps": "Drain the casing and press the new seal onto the shaft sleeve",
            "difficulty": "medium",
            "required_tools": ["seal installation sleeve"],
        }
    ],
}


def manual_chunks(doc_id, oem, model, semantic_text):
    common = {"doc_id": doc_id, "oem": oem, "model": model, "file_path": f"{doc_id}.md"}
    return [
        Chunk(chunk_id=f"{doc_id}-1", level=1, content=semantic_text, **common),
        Chunk(
            chunk_id=f"{doc_id}-2", level=2, content=semantic_text, section_name="Seal", **common
        ),
        Chunk(chunk_id=f"{doc_id}-4", level=4, content=semantic_text, **common),
    ]


def populate(graph, entities_by_chunk, *chunk_lists):
    populator = GraphPopulator(
        neo4j_manager=graph, entity_extractor=StubExtractor(entities_by_chunk)
    )
    try:
        return populator.populate([chunk for chunks in chunk_lists for chunk in chunks])
    finally:
        populator.close()


def solved_by_pairs(graph):
    return {
        (graph.nodes[a]["description"], graph.nodes[b]["fix_description"])
        for a, rel, b in graph.relationships
        if rel == "SOLVED_BY"
    }


def test_each_symptom_is_linked_only_to_its_own_solution():
    graph = InMemoryGraph()
    stats = populate(
        graph, {"doc-a-4": TWO_PAIRS}, manual_chunks("doc-a", "ExampleCo", "CP-200", "x")
    )

    assert solved_by_pairs(graph) == {
        ("Gland drips", "Replace the mechanical seal"),
        ("Motor trips on start", "Reset the overload relay"),
    }
    assert stats["solutions_created"] == 2


def test_a_solution_named_by_no_symptom_is_kept_without_a_link():
    entities = {**TWO_PAIRS, "symptoms": [{"description": "Gland drips", "solutions": []}]}
    graph = InMemoryGraph()
    populate(graph, {"doc-a-4": entities}, manual_chunks("doc-a", "ExampleCo", "CP-200", "x"))

    assert solved_by_pairs(graph) == set()
    assert len(graph.of_label("Solution")) == 2


def test_every_node_records_the_documents_and_equipment_it_came_from():
    graph = InMemoryGraph()
    populate(
        graph,
        {"doc-a-4": {**TWO_PAIRS, **SEAL_PROCEDURE}, "doc-b-4": SEAL_PROCEDURE},
        manual_chunks("doc-a", "ExampleCo", "CP-200", "x"),
        manual_chunks("doc-b", "ExampleCo", "RSC-15", "y"),
    )

    for node in graph.nodes.values():
        assert node.get("doc_ids"), node
        assert node.get("equipment_ids"), node
    (tool,) = graph.of_label("Tool")
    assert tool["doc_ids"] == ["doc-a", "doc-b"]
    assert tool["equipment_ids"] == ["EXAMPLECO_CP-200", "EXAMPLECO_RSC-15"]
    symptom = graph.of_label("Symptom")[0]
    assert symptom["equipment_ids"] == ["EXAMPLECO_CP-200"]


class RecordingGraph:
    def __init__(self, rows=()):
        self.calls = []
        self.rows = list(rows)

    def execute_query(self, query, parameters=None, write=False):
        self.calls.append((query, parameters or {}))
        return self.rows


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({"oem": "ExampleCo", "model": "CP-200"}, {"equipment_id": "EXAMPLECO_CP-200"}),
        ({"oem": "ExampleCo"}, {"equipment_prefix": "EXAMPLECO_"}),
        ({"model": "CP-200"}, {"equipment_suffix": "_CP-200"}),
        ({"doc_id": "doc-a"}, {"doc_id": "doc-a"}),
    ],
)
def test_graph_search_applies_the_vector_filters(monkeypatch, filters, expected):
    monkeypatch.setenv("RETRIEVAL_MODE", "graph")
    from service_manual_rag.config import get_settings

    get_settings.cache_clear()
    graph = RecordingGraph()
    HybridRetriever(vector_indexer=object(), neo4j_manager=graph).retrieve("seal", filters=filters)

    query, parameters = graph.calls[0]
    for key, value in expected.items():
        assert parameters[key] == value
        assert f"${key}" in query


def test_an_unsupported_graph_filter_is_an_error_not_an_unfiltered_search(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MODE", "graph")
    from service_manual_rag.config import get_settings

    get_settings.cache_clear()
    graph = RecordingGraph(rows=[{"node": {"id": "g1"}, "score": 1.0, "types": ["Symptom"]}])
    retriever = HybridRetriever(vector_indexer=object(), neo4j_manager=graph)

    result = retriever.retrieve("seal", filters={"chunk_level": 4})

    assert result["graph_results"] == []
    assert "chunk_level" in result["graph_error"]
    assert graph.calls == []


def test_fulltext_index_covers_procedure_text():
    (fulltext,) = [query for query in create_indexes() if "FULLTEXT" in query]
    for field in ("n.action_type", "n.steps", "n.difficulty"):
        assert field in fulltext


class EmptyQdrant:
    def count_points(self, filters=None):
        return 0


def test_deleting_a_document_also_removes_its_extracted_knowledge():
    graph = RecordingGraph(rows=[{"nodes_deleted": 3}])
    manager = DocumentManager(qdrant_manager=EmptyQdrant(), neo4j_manager=graph)

    manager.delete_document("doc-a")

    queries = [query for query, parameters in graph.calls if parameters == {"doc_id": "doc-a"}]
    assert any("doc_ids" in query for query in queries)


def test_process_page_initializes_the_schema_and_closes_the_populator(monkeypatch):
    import importlib.util
    from pathlib import Path

    events = []

    class Populator:
        def populate(self, chunks):
            events.append("populate")
            return {}

        def close(self):
            events.append("close")

    def open_populator(reset=False):
        events.append(("open", reset))
        return Populator()

    monkeypatch.setattr("service_manual_rag.graph_builder.open_graph_populator", open_populator)
    path = Path(__file__).parents[1] / "ui" / "pages" / "2_Process.py"
    spec = importlib.util.spec_from_file_location("process_page", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.populate_graph([])

    assert events == [("open", False), "populate", "close"]


NEO4J_URI = os.environ.get("SMR_TEST_NEO4J_URI")


@pytest.mark.skipif(not NEO4J_URI, reason="SMR_TEST_NEO4J_URI is not set")
def test_neo4j_round_trip(monkeypatch):
    from service_manual_rag.config import get_settings
    from service_manual_rag.knowledge_graph import Neo4JManager

    graph = Neo4JManager(uri=NEO4J_URI, password=os.environ["SMR_TEST_NEO4J_PASSWORD"])
    graph.clear_database(confirm=True)
    graph.initialize_schema()
    graph.execute_query("CALL db.awaitIndexes(60)")
    populator = GraphPopulator(
        neo4j_manager=graph,
        entity_extractor=StubExtractor(
            {"doc-a-4": {**TWO_PAIRS, **SEAL_PROCEDURE}, "doc-b-4": SEAL_PROCEDURE}
        ),
    )
    populator.populate(
        manual_chunks("doc-a", "ExampleCo", "CP-200", "x")
        + manual_chunks("doc-b", "ExampleCo", "RSC-15", "y")
    )

    monkeypatch.setenv("RETRIEVAL_MODE", "graph")
    get_settings.cache_clear()
    retriever = HybridRetriever(vector_indexer=object(), neo4j_manager=graph)

    def procedures(filters=None):
        results = retriever.retrieve("shaft sleeve", filters=filters)
        assert results["graph_error"] is None
        return [r["node"]["doc_ids"] for r in results["graph_results"] if "Procedure" in r["types"]]

    assert sorted(procedures()) == [["doc-a"], ["doc-b"]]
    assert procedures({"oem": "ExampleCo", "model": "RSC-15"}) == [["doc-b"]]
    assert procedures({"doc_id": "doc-a"}) == [["doc-a"]]

    pairs = graph.execute_query(
        "MATCH (s:Symptom)-[:SOLVED_BY]->(f:Solution) "
        "RETURN s.description AS symptom, f.fix_description AS fix"
    )
    assert {(row["symptom"], row["fix"]) for row in pairs} == {
        ("Gland drips", "Replace the mechanical seal"),
        ("Motor trips on start", "Reset the overload relay"),
    }

    DocumentManager(qdrant_manager=EmptyQdrant(), neo4j_manager=graph).delete_document("doc-a")

    assert procedures() == [["doc-b"]]
    assert graph.execute_query("MATCH (n) WHERE 'doc-a' IN n.doc_ids RETURN n") == []
    tool = graph.execute_query("MATCH (t:Tool) RETURN t.doc_ids AS docs, t.equipment_ids AS eq")
    assert tool == [{"docs": ["doc-b"], "eq": ["EXAMPLECO_RSC-15"]}]
    assert graph.execute_query("MATCH (e:Equipment) RETURN e.id AS id") == [
        {"id": "EXAMPLECO_RSC-15"}
    ]
    graph.clear_database(confirm=True)
    populator.close()
