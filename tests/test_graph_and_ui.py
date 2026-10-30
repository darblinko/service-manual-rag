"""Offline checks for the graph clients, the Streamlit pages and the maintenance scripts."""

import asyncio
import importlib.util
import py_compile
from pathlib import Path
from types import SimpleNamespace

import pytest
from neo4j import GraphDatabase

from service_manual_rag.config import MissingCredentialError, OfflineModeError, get_settings
from service_manual_rag.embeddings import VectorIndexer, create_embedding_service
from service_manual_rag.graph_builder import EntityExtractor, GraphPopulator
from service_manual_rag.knowledge_graph import Neo4JManager
from service_manual_rag.processing import ProcessingPipeline
from service_manual_rag.vector_db import QdrantManager

REPO_ROOT = Path(__file__).parents[1]
UI_FILES = sorted((REPO_ROOT / "ui").rglob("*.py"))
SCRIPT_FILES = sorted((REPO_ROOT / "scripts").glob("*.py"))
SAMPLE_MANUAL = REPO_ROOT / "sample_data" / "manuals" / "exampleco-cp-200-centrifugal-pump.md"


def load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"scripts_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_neo4j_manager_requires_password_before_connecting(monkeypatch):
    def refuse_connection(*args, **kwargs):
        raise AssertionError("Neo4JManager opened a driver without a password")

    monkeypatch.setattr(GraphDatabase, "driver", refuse_connection)
    with pytest.raises(MissingCredentialError, match="NEO4J_PASSWORD"):
        Neo4JManager()


def test_entity_extractor_requires_openai_key(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "openai")
    get_settings.cache_clear()
    with pytest.raises(MissingCredentialError, match="OPENAI_API_KEY"):
        EntityExtractor()


def test_entity_extractor_refuses_offline_mode_even_with_a_key():
    with pytest.raises(OfflineModeError, match="LLM_MODE=offline"):
        EntityExtractor(api_key="placeholder-key")


def test_ui_and_script_files_found():
    assert UI_FILES, "no files under ui/"
    assert SCRIPT_FILES, "no files under scripts/"


@pytest.mark.parametrize(
    "path", UI_FILES + SCRIPT_FILES, ids=lambda p: p.relative_to(REPO_ROOT).as_posix()
)
def test_file_compiles(path, tmp_path):
    py_compile.compile(str(path), cfile=str(tmp_path / "compiled.pyc"), doraise=True)


def run_page(relative_path: str):
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(REPO_ROOT / "ui" / relative_path), default_timeout=60)
    return app.run()


def test_home_page_shows_offline_mode():
    app = run_page("app.py")
    assert not app.exception
    assert any(metric.value == "offline" for metric in app.metric)


def test_agent_chat_page_answers_offline_from_the_indexed_manual():
    pytest.importorskip("langgraph")
    _, chunks = ProcessingPipeline().process(SAMPLE_MANUAL)
    VectorIndexer().index_chunks(chunks)

    app = run_page("pages/5_Agent_Chat.py")
    assert not app.exception
    assert any("Offline mode" in info.value for info in app.info)

    app.chat_input[0].set_value("What is the impeller nut torque on the ExampleCo CP-200?").run()

    assert not app.exception
    reply = app.chat_message[-1]
    assert reply.name == "assistant"
    reply_text = " ".join(md.value for md in reply.markdown)
    assert "Fastener torque values" in reply_text
    assert "Impeller nut | M16 | 55 N·m" in reply_text
    assert "could not find" not in reply_text.lower()
    assert "Additional Information Needed" not in reply_text


@pytest.mark.parametrize("path", SCRIPT_FILES, ids=lambda p: p.name)
def test_script_exposes_main(path):
    module = load_script(path.stem)
    assert callable(module.main)


def test_setup_databases_creates_collection():
    assert load_script("setup_databases").main([]) == 0

    manager = QdrantManager()
    assert manager.collection_exists()
    assert manager.get_collection_info()["vector_size"] == create_embedding_service().dimensions


def test_setup_databases_graph_flag_without_password_reports_error(capsys):
    assert load_script("setup_databases").main(["--graph"]) == 1
    assert "NEO4J_PASSWORD" in capsys.readouterr().err


def test_clear_databases_recreates_empty_collection(monkeypatch):
    indexer = VectorIndexer()
    _, chunks = ProcessingPipeline().process(SAMPLE_MANUAL)
    indexer.index_chunks(chunks)
    assert indexer.qdrant_manager.count_points() > 0

    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    assert load_script("clear_databases").main([]) == 0
    assert indexer.qdrant_manager.count_points() == 0


def test_clear_databases_cancels_without_confirmation(monkeypatch):
    VectorIndexer().ensure_collection()
    monkeypatch.setattr("builtins.input", lambda prompt: "no")
    assert load_script("clear_databases").main([]) == 1
    assert QdrantManager().collection_exists()


def test_diagnose_retrieval_reports_indexed_manual(capsys):
    _, chunks = ProcessingPipeline().process(SAMPLE_MANUAL)
    VectorIndexer().index_chunks(chunks)

    assert load_script("diagnose_retrieval").main(["--query", "impeller"]) == 0
    output = capsys.readouterr().out
    assert "Hybrid RAG" in output
    assert "ISSUE" not in output


def test_diagnose_retrieval_fails_without_collection():
    assert load_script("diagnose_retrieval").main([]) == 1


class FakeCompletions:
    def create(self, **kwargs):
        usage = SimpleNamespace(prompt_tokens=7, completion_tokens=3)
        return SimpleNamespace(usage=usage, kwargs=kwargs)


def test_hosted_evaluation_counts_chat_usage():
    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    counter = load_script("hosted_evaluation").UsageCounter(client)

    response = client.chat.completions.create(model="m", messages=[])
    client.chat.completions.create(model="m", messages=[])

    assert response.kwargs["model"] == "m"
    assert counter.totals() == {"chat_calls": 2, "prompt_tokens": 14, "completion_tokens": 6}


def test_hosted_evaluation_requires_openai_and_graph(capsys, tmp_path):
    exit_code = load_script("hosted_evaluation").main(
        ["--output", str(tmp_path / "out.json"), "--corpus", str(SAMPLE_MANUAL)]
    )
    assert exit_code == 1
    assert "LLM_MODE=openai" in capsys.readouterr().err


class LoopRecordingExtractor:
    def __init__(self):
        self.loops = []

    async def extract_entities_batch(self, chunks):
        self.loops.append(asyncio.get_running_loop())
        return [{} for _ in chunks]


class ClosingGraph:
    closed = False

    def close(self):
        self.closed = True


def test_graph_populator_runs_every_populate_call_on_one_event_loop():
    extractor = LoopRecordingExtractor()
    graph = ClosingGraph()
    populator = GraphPopulator(neo4j_manager=graph, entity_extractor=extractor)

    populator.populate([])
    populator.populate([])
    populator.close()

    assert len(extractor.loops) == 4
    assert len(set(map(id, extractor.loops))) == 1
    assert graph.closed
