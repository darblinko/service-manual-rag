"""Scoring backends for the RAGAS-style metrics.

OpenAIJudge asks an LLM for a 0 to 1 score per metric. LexicalJudge computes
deterministic token-overlap proxies with no network access; its numbers measure
lexical overlap and are not comparable with LLM-judged scores.
"""

import json
import re
from typing import Protocol

from openai import OpenAI

from service_manual_rag.config import Settings, get_settings
from service_manual_rag.utils.text import content_tokens, split_sentences

CITATION = re.compile(r"\[Source \d+\]")
CONTEXT_PREVIEW_CHARS = 500

PROMPTS = {
    "context_relevance": """Rate the relevance of the provided contexts to the question.
Score from 0 (not relevant) to 1 (highly relevant).

Question: {question}

Contexts:
{contexts}

Return ONLY a JSON object: {{"score": 0.0-1.0, "reasoning": "..."}}""",
    "answer_relevance": """Rate how well the answer addresses the question.
Score from 0 (does not answer) to 1 (complete answer).

Question: {question}

Answer: {answer}

Return ONLY a JSON object: {{"score": 0.0-1.0, "reasoning": "..."}}""",
    "faithfulness": """Rate whether the answer is faithful to the provided contexts.
Score 1 if the answer only uses information from the contexts, 0 if it is unsupported.

Answer: {answer}

Contexts:
{contexts}

Return ONLY a JSON object: {{"score": 0.0-1.0, "reasoning": "..."}}""",
    "correctness": """Compare the generated answer to the ground truth.
Score from 0 (wrong) to 1 (equivalent or better).

Generated Answer: {answer}

Ground Truth: {ground_truth}

Return ONLY a JSON object: {{"score": 0.0-1.0, "reasoning": "..."}}""",
}


class Judge(Protocol):
    name: str

    def score(
        self,
        metric: str,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> float: ...


def strip_citations(text: str) -> str:
    return CITATION.sub("", text)


def coverage(part: set[str], whole: set[str]) -> float:
    """Share of ``part`` tokens that appear in ``whole``; 0 when ``part`` is empty."""
    return len(part & whole) / len(part) if part else 0.0


def token_f1(predicted: set[str], expected: set[str]) -> float:
    common = len(predicted & expected)
    if common == 0:
        return 0.0
    precision = common / len(predicted)
    recall = common / len(expected)
    return 2 * precision * recall / (precision + recall)


class LexicalJudge:
    """Deterministic offline proxies.

    - context_relevance: mean share of question terms found in each context
    - answer_relevance: share of question terms found in the answer
    - faithfulness: mean share of each answer sentence's terms found in the contexts
    - correctness: token F1 between answer and ground truth
    Terms are lower-cased tokens with stopwords removed.
    """

    name = "lexical-offline"

    def score(
        self,
        metric: str,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> float:
        question_terms = content_tokens(question)
        answer_text = strip_citations(answer)
        if metric == "context_relevance":
            if not contexts:
                return 0.0
            return sum(coverage(question_terms, content_tokens(c)) for c in contexts) / len(
                contexts
            )
        if metric == "answer_relevance":
            return coverage(question_terms, content_tokens(answer_text))
        if metric == "faithfulness":
            context_terms = content_tokens(" ".join(contexts))
            sentences = [content_tokens(s) for s in split_sentences(answer_text)]
            sentences = [s for s in sentences if s]
            if not sentences:
                return 0.0
            return sum(coverage(s, context_terms) for s in sentences) / len(sentences)
        if metric == "correctness":
            if not ground_truth:
                raise ValueError("correctness needs a ground truth")
            return token_f1(content_tokens(answer_text), content_tokens(ground_truth))
        raise ValueError(f"Unknown metric: {metric}")


class OpenAIJudge:
    """LLM-as-judge scoring with JSON-mode responses."""

    def __init__(self, settings: Settings | None = None, client: OpenAI | None = None):
        settings = settings or get_settings()
        self.name = f"openai:{settings.openai_model}"
        self.model = settings.openai_model
        self.client = client or OpenAI(api_key=settings.require_openai_key())

    def score(
        self,
        metric: str,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> float:
        context_text = "\n".join(
            f"Context {i}: {c[:CONTEXT_PREVIEW_CHARS]}" for i, c in enumerate(contexts, start=1)
        )
        prompt = PROMPTS[metric].format(
            question=question, answer=answer, contexts=context_text, ground_truth=ground_truth
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        value = float(json.loads(response.choices[0].message.content)["score"])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Judge returned {value} for {metric}, outside 0 to 1")
        return value


def create_judge(settings: Settings | None = None) -> Judge:
    settings = settings or get_settings()
    if settings.use_openai:
        return OpenAIJudge(settings)
    return LexicalJudge()
