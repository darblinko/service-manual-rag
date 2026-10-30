"""Multi-turn tests of the agent workflow, offline, with stand-ins for search and the graph."""

import asyncio

import pytest
from test_agents import FailingVision, StubChatClient, StubRag

from service_manual_rag.agents.action_agent import ActionAgent
from service_manual_rag.agents.diagnostic_agent import DiagnosticAgent
from service_manual_rag.agents.reasoning.llm_reasoner import LLMDiagnosticReasoner
from service_manual_rag.agents.state import DiagnosticStatus, add_media, create_initial_state

pytest.importorskip("langgraph")

SEAL_SOURCES = [
    {
        "id": "chunk-7",
        "content": "Seal replacement\nReplace the mechanical seal when the gland drips.",
        "sources": ["vector"],
        "rrf_score": 0.1,
        "section_name": "4.2 Mechanical seal",
        "model": "CP-200",
        "doc_id": "doc-cp200",
        "page_numbers": [12],
    }
]


class RecordingGraph:
    """Neo4JManager stand-in that records queries and returns a session node on create."""

    def __init__(self):
        self.queries = []

    def execute_query(self, query, params=None, write=False):
        self.queries.append((query, params or {}))
        if "CREATE (s:TroubleshootingSession" in query:
            return [{"id": params["session_id"]}]
        return []


class FindingVision:
    """Vision stand-in that reports one finding per analyzed image."""

    def __init__(self, description):
        self.description = description
        self.calls = 0

    def analyze_image(self, *args, **kwargs):
        self.calls += 1
        return {"findings": [{"description": self.description, "confidence": 0.9}]}


class Conversation:
    """Runs user turns through the compiled workflow with injected agents."""

    def __init__(self, monkeypatch, rag=None, vision=None, graph=None):
        from service_manual_rag.agents import workflow

        self.workflow = workflow
        self.rag = rag or StubRag()
        self.vision = vision or FailingVision()
        self.graph = graph or RecordingGraph()
        monkeypatch.setattr(workflow, "_workflow", None)
        monkeypatch.setattr(
            workflow,
            "create_diagnostic_agent",
            lambda: DiagnosticAgent(rag_pipeline=self.rag, vlm_service=self.vision),
        )
        monkeypatch.setattr(
            workflow, "create_action_agent", lambda: ActionAgent(neo4j_manager=self.graph)
        )
        self.state = create_initial_state("s")

    def say(self, message, media=None):
        self.state["messages"].append({"role": "user", "content": message})
        for entry in media or []:
            self.state = add_media(self.state, entry)
        self.state = asyncio.run(self.workflow.run_agent_workflow(self.state))
        return self.state["messages"][-1]["content"]

    def recorded_sessions(self):
        return [params for query, params in self.graph.queries if "CREATE" in query]


def test_record_after_a_diagnosis_writes_that_diagnosis(monkeypatch):
    chat = Conversation(monkeypatch, rag=StubRag(answer="Replace the seal", sources=SEAL_SOURCES))
    first = chat.say("ExampleCo CP-200 is leaking")
    assert "Seal or gasket failure" in first

    reply = chat.say("record this session")

    assert "Session recorded with ID: s" in reply
    assert chat.recorded_sessions()[0]["diagnosis"] == "Seal or gasket failure"


def test_confirmation_records_the_diagnosis_it_confirms(monkeypatch):
    chat = Conversation(monkeypatch, rag=StubRag(answer="Replace the seal", sources=SEAL_SOURCES))
    assert "record this troubleshooting session" in chat.say("ExampleCo CP-200 is leaking")

    reply = chat.say("yes")

    assert "Session recorded" in reply
    assert chat.recorded_sessions()[0]["diagnosis"] == "Seal or gasket failure"


def test_a_repeated_message_is_processed_as_a_new_turn(monkeypatch):
    chat = Conversation(monkeypatch)
    chat.say("ExampleCo CP-200 is leaking")
    chat.say("ExampleCo CP-200 is leaking")

    assert len(chat.rag.calls) == 2
    assert [m["role"] for m in chat.state["messages"]] == ["user", "assistant"] * 2


def test_naming_other_equipment_changes_the_search_filter(monkeypatch):
    chat = Conversation(monkeypatch)
    chat.say("ExampleCo CP-200 is leaking")

    chat.say("Now the RSC-15 is overheating")

    assert chat.rag.calls[-1]["filters"] == {"oem": "ExampleCo", "model": "RSC-15"}
    assert "leak" not in chat.state["symptoms"]
    assert "overheating" in chat.state["symptoms"]


def test_uploaded_media_is_analyzed_once_and_kept_as_evidence(monkeypatch):
    vision = FindingVision("seal visibly cracked")
    chat = Conversation(monkeypatch, vision=vision)
    photo = {"name": "seal.png", "path": "media/s_seal.png", "type": "image/png", "size": 10}
    chat.say("ExampleCo CP-200 is leaking", media=[photo])

    reply = chat.say("record this session", media=[photo])

    assert vision.calls == 1
    assert "Session recorded" in reply
    assert chat.state["uploaded_media"] == []
    assert chat.state["analyzed_media"] == [photo]
    assert [ev.content for ev in chat.state["evidence"]] == ["seal visibly cracked"]


def test_visual_findings_reach_the_hypothesis_prompt():
    client = StubChatClient("1. Cracked seal face (Confidence: 0.8)")
    agent = DiagnosticAgent(
        rag_pipeline=StubRag(),
        vlm_service=FindingVision("seal visibly cracked"),
        llm_reasoner=LLMDiagnosticReasoner(llm=client, model="test-model"),
    )
    state = create_initial_state("s")
    state.update(
        symptoms=["leak"],
        status=DiagnosticStatus.DIAGNOSING,
        uploaded_media=[{"type": "image/png", "path": "seal.png"}],
    )

    result = asyncio.run(agent.diagnose(state))

    assert "seal visibly cracked" in client.requests[0]["messages"][1]["content"]
    assert [ev.content for ev in result.new_evidence] == ["seal visibly cracked"]


def test_a_factual_answer_does_not_claim_missing_information():
    rag = StubRag(answer="Torque the impeller nut to 45 N m [Source 1]", sources=SEAL_SOURCES)
    agent = DiagnosticAgent(rag_pipeline=rag, vlm_service=FailingVision())
    state = create_initial_state("s")
    state.update(equipment_oem="ExampleCo", equipment_model="CP-200")
    state["messages"].append({"role": "user", "content": "What is the impeller nut torque?"})
    state["status"] = DiagnosticStatus.GATHERING

    result = asyncio.run(agent.diagnose(state))

    assert result.evidence_needed is None
    assert not any("could not find" in obs.lower() for obs in result.observations)


def test_reasoner_receives_passage_bodies_with_source_identifiers():
    formatted = LLMDiagnosticReasoner._format_retrieved_docs(
        [{**SEAL_SOURCES[0], "metadata": {"section_name": "4.2 Mechanical seal"}}]
    )

    assert "Replace the mechanical seal when the gland drips." in formatted
    assert "4.2 Mechanical seal" in formatted


def test_add_media_skips_an_attachment_already_seen():
    photo = {"name": "seal.png", "path": "media/s_seal.png", "type": "image/png", "size": 10}
    state = add_media(create_initial_state("s"), photo)
    state = add_media(state, dict(photo))
    assert state["uploaded_media"] == [photo]

    state = {**state, "uploaded_media": [], "analyzed_media": [photo]}
    assert add_media(state, dict(photo))["uploaded_media"] == []


class ClosableGraph(RecordingGraph):
    def __init__(self):
        super().__init__()
        self.closed = False

    def close(self):
        self.closed = True


def test_agents_are_built_once_per_workflow_and_closed_with_it(monkeypatch):
    from service_manual_rag.agents import workflow

    graph = ClosableGraph()
    built = []

    def build_diagnostic():
        built.append("diagnostic")
        return DiagnosticAgent(rag_pipeline=StubRag(), vlm_service=FailingVision())

    monkeypatch.setattr(workflow, "_workflow", None)
    monkeypatch.setattr(workflow, "create_diagnostic_agent", build_diagnostic)
    monkeypatch.setattr(workflow, "create_action_agent", lambda: ActionAgent(neo4j_manager=graph))
    state = create_initial_state("s")
    for message in ("ExampleCo CP-200 is leaking", "It is still leaking", "record this session"):
        state["messages"].append({"role": "user", "content": message})
        state = asyncio.run(workflow.run_agent_workflow(state))

    assert built == ["diagnostic"]
    workflow.close_workflow()
    assert graph.closed
    assert workflow._workflow is None
