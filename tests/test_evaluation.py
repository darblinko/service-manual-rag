import json

import pytest

from service_manual_rag.evaluation import LexicalJudge, RAGASEvaluator, load_golden_set
from service_manual_rag.evaluation.ragas_evaluator import retrieval_scores


def test_lexical_judge_scores_are_bounded_and_ordered():
    judge = LexicalJudge()
    contexts = ["Tighten the impeller nut to 45 N m."]
    good = judge.score("faithfulness", "torque?", "Tighten the impeller nut to 45 N m.", contexts)
    bad = judge.score("faithfulness", "torque?", "Replace the coolant filter.", contexts)
    assert good == 1.0
    assert 0.0 <= bad < good


def test_lexical_correctness_is_token_f1():
    judge = LexicalJudge()
    score = judge.score("correctness", "q", "impeller nut 45", [], ground_truth="impeller nut 50")
    assert score == pytest.approx(2 / 3)


def test_lexical_judge_rejects_unknown_metric():
    with pytest.raises(ValueError):
        LexicalJudge().score("fluency", "q", "a", [])


def test_retrieval_scores_use_rank_of_first_matching_section():
    sources = [{"section_name": "Ratings"}, {"section_name": "Bearing Lubrication"}]
    assert retrieval_scores(sources, "bearing lubrication") == {
        "section_hit": 1.0,
        "reciprocal_rank": 0.5,
    }
    assert retrieval_scores(sources, "Wiring")["section_hit"] == 0.0


def test_section_hit_requires_the_expected_model():
    sources = [
        {"model": "EM-30", "section_name": "Bearing Lubrication"},
        {"model": "T-1", "section_name": "Ratings"},
        {"model": "T-1", "section_name": "Bearing Lubrication"},
    ]
    scores = retrieval_scores(sources, "Bearing Lubrication", expected_model="T-1")
    assert scores["reciprocal_rank"] == pytest.approx(1 / 3)
    assert scores["model_precision"] == pytest.approx(2 / 3)


class FailingJudge:
    name = "failing"

    def score(self, metric, question, answer, contexts, ground_truth=None):
        if metric == "faithfulness":
            raise RuntimeError("judge unavailable")
        return 1.0


def test_failed_judgments_are_recorded_and_not_scored():
    scores, errors = RAGASEvaluator(judge=FailingJudge()).evaluate_answer(
        "q", "a", ["c"], ground_truth="g"
    )
    assert "faithfulness" not in scores
    assert errors == ["faithfulness: judge unavailable"]
    assert scores["correctness"] == 1.0


def test_load_golden_set_validates_fields(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text(json.dumps([{"id": "q1", "question": "q"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="ground_truth"):
        load_golden_set(path)
