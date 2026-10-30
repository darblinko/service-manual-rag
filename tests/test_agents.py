"""Offline tests for the agents package and the enhanced RAG pipeline."""

import asyncio
import importlib
import subprocess
import sys
from datetime import datetime
from types import SimpleNamespace

import pytest

from service_manual_rag.agents import components
from service_manual_rag.agents.action_agent import GRAPH_DISABLED_DETAILS, ActionAgent
from service_manual_rag.agents.diagnostic_agent import (
    DiagnosticAgent,
    media_kind,
    rule_based_hypotheses,
)
from service_manual_rag.agents.memory import ConversationMemoryStore, SemanticMemory
from service_manual_rag.agents.reasoning.llm_reasoner import LLMDiagnosticReasoner
from service_manual_rag.agents.retrieval.contextual_reranker import ContextualReranker
from service_manual_rag.agents.session_manager import LocalSessionStore, SessionManager
from service_manual_rag.agents.state import (
    DiagnosticResult,
    DiagnosticStatus,
    Evidence,
    Hypothesis,
    create_initial_state,
)
from service_manual_rag.agents.supervisor import SupervisorAgent, find_known_oem, parse_known_oems
from service_manual_rag.agents.vlm_service import VLMService
from service_manual_rag.config import MissingCredentialError, OfflineModeError, get_settings

SCOPE_MODULES = [
    "service_manual_rag.agents",
    "service_manual_rag.agents.action_agent",
    "service_manual_rag.agents.components",
    "service_manual_rag.agents.diagnostic_agent",
    "service_manual_rag.agents.memory",
    "service_manual_rag.agents.memory.semantic_memory",
    "service_manual_rag.agents.reasoning",
    "service_manual_rag.agents.reasoning.llm_reasoner",
    "service_manual_rag.agents.retrieval",
    "service_manual_rag.agents.retrieval.agentic_retriever",
    "service_manual_rag.agents.retrieval.contextual_reranker",
    "service_manual_rag.agents.retrieval.query_rewriter",
    "service_manual_rag.agents.session_manager",
    "service_manual_rag.agents.state",
    "service_manual_rag.agents.supervisor",
    "service_manual_rag.agents.vlm_service",
    "service_manual_rag.agents.workflow",
    "service_manual_rag.rag.enhanced_rag_pipeline",
]


class StubRag:
    """RAG pipeline stand-in that returns a fixed result and records each call."""

    def __init__(self, answer="", sources=None):
        self.answer = answer
        self.sources = sources or []
        self.calls = []

    def query(self, question, top_k=None, filters=None):
        self.calls.append({"question": question, "top_k": top_k, "filters": filters})
        return {"question": question, "answer": self.answer, "sources": self.sources}


class StubChatClient:
    """Object with the OpenAI ``chat.completions.create`` shape returning fixed text."""

    def __init__(self, content):
        self.requests = []
        message = SimpleNamespace(content=content)
        response = SimpleNamespace(choices=[SimpleNamespace(message=message)])

        def create(**kwargs):
            self.requests.append(kwargs)
            return response

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def use_settings(monkeypatch, **env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    return get_settings()


@pytest.mark.parametrize("module_name", SCOPE_MODULES)
def test_module_imports_without_keys(module_name):
    importlib.import_module(module_name)


def test_import_does_not_load_optional_vision_packages(tmp_path):
    code = (
        "import sys\n"
        "import service_manual_rag.agents.workflow\n"
        "import service_manual_rag.agents.vlm_service\n"
        "import service_manual_rag.rag.enhanced_rag_pipeline\n"
        "loaded = [m for m in ('google.generativeai', 'PIL') if m in sys.modules]\n"
        "print(','.join(loaded))\n"
    )
    env = {"PATH": "", "LLM_MODE": "offline", "QDRANT_LOCAL_PATH": str(tmp_path / "q")}
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == ""


def test_create_initial_state_defaults():
    state = create_initial_state("session-1", user_id="tech-7")

    assert state["session_id"] == "session-1"
    assert state["user_id"] == "tech-7"
    assert state["status"] == DiagnosticStatus.IDLE
    assert state["equipment_oem"] is None
    assert state["equipment_model"] is None
    assert state["messages"] == []
    assert state["symptoms"] == []
    assert state["uploaded_media"] == []
    assert state["diagnostic_result"] is None
    assert state["action_result"] is None
    assert state["next_agent"] is None
    assert state["pending_user_input"] is False
    assert state["session_start"] == state["last_updated"]
    assert not any(key.startswith("auto_ingestion") for key in state)


def test_create_initial_state_returns_independent_lists():
    first = create_initial_state("a")
    second = create_initial_state("b")
    first["symptoms"].append("leak")
    assert second["symptoms"] == []


def test_parse_known_oems_keeps_casing_and_drops_blanks():
    assert parse_known_oems(" ExampleCo, Acme Pumps ,,") == ["ExampleCo", "Acme Pumps"]


def test_find_known_oem_is_case_insensitive_with_word_boundaries():
    oems = ["ExampleCo", "Acme"]
    assert find_known_oem("the exampleco unit", oems) == "ExampleCo"
    assert find_known_oem("an ACME motor", oems) == "Acme"
    assert find_known_oem("an AcmeCorp motor", ["Acme"]) is None


def test_supervisor_offline_does_not_construct_llm_client():
    supervisor = SupervisorAgent()
    assert supervisor.llm_client is None


def test_supervisor_extracts_oem_model_and_symptoms():
    supervisor = SupervisorAgent()
    state = create_initial_state("s")

    supervisor._extract_information(
        state, "My exampleco CP-200 pump is leaking and making a grinding noise"
    )

    assert state["equipment_oem"] == "ExampleCo"
    assert state["equipment_model"] == "CP-200"
    assert {"leak", "leaking", "grinding", "noise"} <= set(state["symptoms"])


def test_supervisor_does_not_take_an_article_as_the_manufacturer():
    supervisor = SupervisorAgent()
    state = create_initial_state("s")

    supervisor._extract_information(state, "The CP-200 discharge pressure is low")

    assert state["equipment_oem"] is None
    assert state["equipment_model"] == "CP-200"


def test_supervisor_reads_manufacturer_from_made_by_phrase():
    supervisor = SupervisorAgent()
    state = create_initial_state("s")

    supervisor._extract_information(state, "The unit is made by Northwind, model NW-40")

    assert state["equipment_oem"] == "Northwind"
    assert state["equipment_model"] == "NW-40"


def test_supervisor_uses_known_oems_setting(monkeypatch):
    use_settings(monkeypatch, KNOWN_OEMS="ExampleCo, Northwind Compressors")
    supervisor = SupervisorAgent()
    state = create_initial_state("s")

    supervisor._extract_information(state, "northwind compressors RSC-15 is overheating")

    assert state["equipment_oem"] == "Northwind Compressors"
    assert state["equipment_model"] == "RSC-15"
    assert "overheating" in state["symptoms"]


def test_supervisor_routes_troubleshooting_to_diagnostic():
    supervisor = SupervisorAgent()
    state = create_initial_state("s")

    state = supervisor.process_message(state, "The ExampleCo EM-30 motor is overheating")

    assert state["next_agent"] == "diagnostic"
    assert state["status"] == DiagnosticStatus.DIAGNOSING
    assert state["equipment_model"] == "EM-30"


def test_supervisor_routes_question_to_diagnostic_gathering():
    supervisor = SupervisorAgent()
    state = create_initial_state("s")

    state = supervisor.process_message(state, "Which lubricant does the RSC-15 use?")

    assert state["next_agent"] == "diagnostic"
    assert state["status"] == DiagnosticStatus.GATHERING


def test_supervisor_llm_classifier_with_stub_client():
    client = StubChatClient("record")
    supervisor = SupervisorAgent(llm_client=client)
    state = create_initial_state("s")
    assert supervisor._classify_intent("please file this one", state) == "record"
    assert client.requests[0]["model"] == get_settings().openai_model
    assert client.requests[0]["max_completion_tokens"] == 10
    assert "max_tokens" not in client.requests[0]

    invalid = SupervisorAgent(llm_client=StubChatClient("dance"))
    assert invalid._classify_intent("please file this one", state) == "unclear"


def test_supervisor_formats_response_without_decorative_symbols():
    supervisor = SupervisorAgent()
    state = create_initial_state("s")
    state["diagnostic_result"] = DiagnosticResult(
        observations=["Seal area wet"],
        hypotheses=[Hypothesis("Mechanical seal worn", 0.8)],
        diagnosis="Mechanical seal worn",
        recommended_action="Replace the mechanical seal",
        safety_warnings=["PRESSURE HAZARD: Release pressure before opening system"],
    )
    state["turn_agent"] = "diagnostic"

    response = supervisor.generate_response(state)

    assert "Mechanical seal worn" in response
    assert response.isascii()


@pytest.mark.parametrize(
    "build",
    [
        lambda: SupervisorAgent(),
        lambda: DiagnosticAgent(rag_pipeline=StubRag()),
        lambda: components.create_openai_client(get_settings()),
    ],
    ids=["supervisor", "diagnostic", "openai-client"],
)
def test_openai_mode_without_key_raises(monkeypatch, build):
    use_settings(monkeypatch, LLM_MODE="openai")
    with pytest.raises(MissingCredentialError, match="OPENAI_API_KEY"):
        build()


def test_query_rewriter_and_enhanced_pipeline_require_key_in_openai_mode(monkeypatch):
    from service_manual_rag.agents.retrieval import QueryRewriter
    from service_manual_rag.rag.enhanced_rag_pipeline import EnhancedRAGPipeline

    use_settings(monkeypatch, LLM_MODE="openai")
    with pytest.raises(MissingCredentialError):
        QueryRewriter()
    with pytest.raises(MissingCredentialError):
        EnhancedRAGPipeline()


def test_model_names_come_from_the_environment(monkeypatch, tmp_path):
    settings = use_settings(
        monkeypatch,
        OPENAI_MODEL="chat-model-x",
        GEMINI_MODEL="vision-model-y",
        RERANKER_MODEL="reranker-z",
    )

    assert settings.openai_model == "chat-model-x"
    assert SupervisorAgent(llm_client=StubChatClient("record")).model == "chat-model-x"
    assert VLMService(media_dir=tmp_path).model_name == "vision-model-y"
    assert ContextualReranker(use_cross_encoder=False).model_name == "reranker-z"


def test_default_model_names_are_current(monkeypatch):
    for name in ("OPENAI_MODEL", "GEMINI_MODEL", "RERANKER_MODEL"):
        monkeypatch.delenv(name, raising=False)
    settings = use_settings(monkeypatch)

    assert settings.openai_model == "gpt-4.1-mini"
    assert settings.gemini_model == "gemini-2.5-flash"
    assert settings.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_llm_reasoner_limits_completion_tokens():
    client = StubChatClient("1. Worn impeller (Confidence: 0.6)")
    reasoner = LLMDiagnosticReasoner(llm=client, model="test-model")

    asyncio.run(reasoner.generate_hypotheses(["noise"], [], create_initial_state("s"), []))

    assert client.requests[0]["max_completion_tokens"] == 1500
    assert "max_tokens" not in client.requests[0]


def test_vlm_service_needs_google_key_only_for_analysis(monkeypatch, tmp_path):
    use_settings(monkeypatch, LLM_MODE="auto")
    service = VLMService(media_dir=tmp_path / "media")

    saved = service.save_media(b"data", "../escape.png")
    assert saved == str(tmp_path / "media" / "escape.png")

    with pytest.raises(MissingCredentialError, match="GOOGLE_API_KEY"):
        service.analyze_image(saved)


def test_vlm_service_refuses_gemini_in_offline_mode_even_with_a_key(monkeypatch, tmp_path):
    use_settings(monkeypatch, GOOGLE_API_KEY="placeholder-key")
    service = VLMService(media_dir=tmp_path)
    monkeypatch.setitem(sys.modules, "google.generativeai", None)

    with pytest.raises(OfflineModeError, match="LLM_MODE=offline"):
        service.analyze_image("missing.png")
    with pytest.raises(OfflineModeError):
        service.analyze_video("missing.mp4")


def test_reranker_does_not_load_a_cross_encoder_in_offline_mode(monkeypatch):
    def refuse_load(model_name):
        raise AssertionError("cross-encoder loaded in offline mode")

    monkeypatch.setattr(ContextualReranker, "_load_cross_encoder", staticmethod(refuse_load))

    assert ContextualReranker(use_cross_encoder=True).use_cross_encoder is False


def test_explicit_openai_key_is_refused_in_offline_mode():
    from service_manual_rag.agents.retrieval import QueryRewriter

    with pytest.raises(OfflineModeError):
        QueryRewriter(api_key="placeholder-key")


def test_graph_manager_not_constructed_when_graph_disabled():
    assert components.create_graph_manager(get_settings()) is None


def test_action_agent_reports_disabled_graph():
    agent = ActionAgent()
    state = create_initial_state("s")

    result = agent.record_session(state)

    assert agent.kg is None
    assert result.success is False
    assert result.details == GRAPH_DISABLED_DETAILS


@pytest.mark.parametrize(
    ("media_type", "expected"),
    [("image/png", "image"), ("jpg", "image"), ("video/mp4", "video"), ("text/plain", None)],
)
def test_media_kind(media_type, expected):
    assert media_kind(media_type) == expected


def test_rule_based_hypotheses_boosted_by_manual_sections():
    without_sections = rule_based_hypotheses(["leaking"], [])
    with_sections = rule_based_hypotheses(["leaking"], [{"content": "seal"}])

    assert [h.description for h in without_sections] == ["Seal or gasket failure"]
    assert with_sections[0].confidence > without_sections[0].confidence


class FailingVision:
    def analyze_image(self, *args, **kwargs):
        raise MissingCredentialError("GOOGLE_API_KEY is not set.")


def diagnose(agent, state):
    return asyncio.run(agent.diagnose(state))


def test_diagnostic_agent_reports_uncovered_question_offline():
    rag = StubRag()
    agent = DiagnosticAgent(rag_pipeline=rag, vlm_service=FailingVision())
    state = create_initial_state("s")
    state.update(
        equipment_oem="ExampleCo",
        equipment_model="CP-200",
        symptoms=["leak"],
        status=DiagnosticStatus.DIAGNOSING,
        uploaded_media=[{"type": "image/png", "path": "missing.png"}],
    )

    result = diagnose(agent, state)

    assert agent.kg is None
    assert agent.llm_reasoner is None
    assert rag.calls[0]["filters"] == {"oem": "ExampleCo", "model": "CP-200"}
    assert any("do not cover" in obs for obs in result.observations)
    assert any("Could not analyze image" in obs for obs in result.observations)
    assert [h.description for h in result.hypotheses] == ["Seal or gasket failure"]


def test_diagnostic_agent_uses_manual_sources():
    sources = [{"content": "Replace the mechanical seal.", "sources": ["vector"], "rrf_score": 0.1}]
    agent = DiagnosticAgent(
        rag_pipeline=StubRag(answer="Replace the seal [Source 1]", sources=sources),
        vlm_service=FailingVision(),
    )
    state = create_initial_state("s")
    state.update(symptoms=["leaking"], status=DiagnosticStatus.DIAGNOSING)

    result = diagnose(agent, state)

    assert result.relevant_manual_sections[0]["content"] == "Replace the mechanical seal."
    assert result.observations[0].startswith("Found information in the service manual")
    assert result.diagnosis == "Seal or gasket failure"
    assert "LEAK HAZARD" in result.safety_warnings[0]


def test_llm_reasoner_parses_and_ranks_hypotheses():
    reply = (
        "1. Worn impeller (Confidence: 0.6)\n"
        "   Evidence: discharge pressure low\n"
        "   Tests: measure shut-off head\n"
        "2. Clogged suction strainer (Confidence: 0.9)\n"
        "   - noise at inlet\n"
    )
    client = StubChatClient(reply)
    reasoner = LLMDiagnosticReasoner(llm=client, model="test-model")
    state = create_initial_state("s")
    state.update(equipment_oem="ExampleCo", equipment_model="CP-200")
    evidence = [Evidence("text", "gauge reads 20 psi", datetime.now())]

    hypotheses = asyncio.run(
        reasoner.generate_hypotheses(["low pressure"], evidence, state, [{"content": "x"}])
    )

    assert [h.description for h in hypotheses] == ["Clogged suction strainer", "Worn impeller"]
    assert hypotheses[0].confidence == pytest.approx(0.9)
    assert hypotheses[1].supporting_evidence == ["discharge pressure low"]
    assert hypotheses[1].tests_suggested == ["measure shut-off head"]
    assert client.requests[0]["model"] == "test-model"
    assert "ExampleCo CP-200" in client.requests[0]["messages"][1]["content"]


def test_session_round_trip_restores_dataclasses(tmp_path):
    manager = SessionManager(store=LocalSessionStore(str(tmp_path / "sessions")))
    session_id = manager.create_session("tech-1")
    state = manager.get_session(session_id)
    state["status"] = DiagnosticStatus.DIAGNOSING
    state["hypotheses"] = [Hypothesis("Bearing wear", 0.7)]
    state["diagnostic_result"] = DiagnosticResult(
        observations=["noise"], hypotheses=[Hypothesis("Bearing wear", 0.7)]
    )
    manager.update_session(session_id, state)

    loaded = LocalSessionStore(str(tmp_path / "sessions")).load_session(session_id)

    assert loaded["status"] == DiagnosticStatus.DIAGNOSING
    assert isinstance(loaded["session_start"], datetime)
    assert loaded["hypotheses"][0].description == "Bearing wear"
    assert loaded["diagnostic_result"].hypotheses[0].confidence == 0.7


def test_load_missing_session_returns_none(tmp_path):
    assert LocalSessionStore(str(tmp_path)).load_session("unknown") is None


def test_semantic_memory_stores_and_retrieves_offline():
    memory = SemanticMemory(ConversationMemoryStore())
    memory.store_session_result(
        session_id="past-1",
        user_id="tech-1",
        messages=[{"role": "user", "content": "RSC-15 discharge temperature high, trips"}],
        equipment_oem="ExampleCo",
        equipment_model="RSC-15",
        symptoms=["overheating"],
        resolution_success=True,
    )

    found = memory.retrieve_for_context(
        "compressor discharge temperature high",
        equipment_oem="ExampleCo",
        equipment_model="RSC-15",
    )

    assert len(found) == 1
    assert found[0]["session_id"] == "past-1"
    assert "Case 1" in memory.format_for_retrieval(found)


def test_workflow_runs_one_turn_offline(monkeypatch):
    from service_manual_rag.agents import workflow

    monkeypatch.setattr(workflow, "_workflow", None)
    monkeypatch.setattr(
        workflow,
        "create_diagnostic_agent",
        lambda: DiagnosticAgent(rag_pipeline=StubRag(), vlm_service=FailingVision()),
    )
    state = create_initial_state("s")
    state["messages"].append({"role": "user", "content": "ExampleCo CP-200 is leaking"})

    result = asyncio.run(workflow.run_agent_workflow(state))

    assert result["messages"][-1]["role"] == "assistant"
    assert "Seal or gasket failure" in result["messages"][-1]["content"]
    assert result["pending_user_input"] is True
