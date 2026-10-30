import json
from pathlib import Path

from service_manual_rag.cli import main

SAMPLE_DATA = Path(__file__).parents[1] / "sample_data"


def test_quickstart_runs_end_to_end_offline(tmp_path, capsys):
    output = tmp_path / "evaluation.json"
    exit_code = main(
        [
            "quickstart",
            "--manuals",
            str(SAMPLE_DATA / "manuals"),
            "--golden-set",
            str(SAMPLE_DATA / "golden_set.json"),
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    golden_set = json.loads((SAMPLE_DATA / "golden_set.json").read_text(encoding="utf-8"))
    assert report["judge"] == "lexical-offline"
    assert report["generator"] == "extractive-offline"
    assert report["total_questions"] == len(golden_set)
    assert report["failed_judgments"] == 0
    assert report["retrieval_mode"] == "vector"
    assert report["graph_errors"] == []
    for value in report["aggregate"].values():
        assert 0.0 <= value <= 1.0
    assert "[Source 1]" in capsys.readouterr().out


def test_committed_offline_baseline_matches_a_fresh_run(tmp_path):
    output = tmp_path / "baseline.json"
    assert main(["ingest", "--manuals", str(SAMPLE_DATA / "manuals"), "--reset"]) == 0
    golden_set = SAMPLE_DATA / "golden_set.json"
    assert main(["evaluate", "--golden-set", str(golden_set), "--output", str(output)]) == 0

    committed = json.loads(
        (Path(__file__).parents[1] / "results" / "offline-baseline.json").read_text(
            encoding="utf-8"
        )
    )
    fresh = json.loads(output.read_text(encoding="utf-8"))
    assert fresh["aggregate"] == committed["aggregate"]
    assert fresh["per_question"] == committed["per_question"]


def test_ask_without_openai_key_in_openai_mode_fails_cleanly(monkeypatch, capsys):
    monkeypatch.setenv("LLM_MODE", "openai")
    from service_manual_rag.config import get_settings

    get_settings.cache_clear()
    assert main(["ask", "pump torque"]) == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_missing_manual_directory_is_reported(tmp_path, capsys):
    assert main(["ingest", "--manuals", str(tmp_path / "empty-dir-that-is-missing")]) == 1
    assert "Error" in capsys.readouterr().err


class RecordingPopulator:
    def __init__(self):
        self.batches = []

    def populate(self, chunks):
        self.batches.append(chunks)
        return {"sections_created": len(chunks)}

    def close(self):
        self.closed = True


def test_ingest_with_graph_populates_each_manual(monkeypatch, capsys):
    populator = RecordingPopulator()
    resets = []

    def open_populator(reset=False):
        resets.append(reset)
        return populator

    monkeypatch.setattr("service_manual_rag.cli.open_graph_populator", open_populator)
    manuals = SAMPLE_DATA / "manuals"
    assert main(["ingest", "--manuals", str(manuals), "--reset", "--graph"]) == 0

    assert resets == [True]
    assert len(populator.batches) == len(list(manuals.glob("*.md")))
    assert "graph" in capsys.readouterr().out
