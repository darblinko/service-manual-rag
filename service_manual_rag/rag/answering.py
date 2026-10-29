"""Answer generators: OpenAI chat completion, or an offline extractive answerer."""

from typing import Protocol

from openai import OpenAI

from service_manual_rag.config import Settings, get_settings
from service_manual_rag.utils.text import content_tokens, split_sentences

SYSTEM_PROMPT = """You are a technical expert helping with industrial equipment service and maintenance.
Use the provided context from service manuals to answer questions accurately.
Include specific part numbers, procedures, and safety warnings when available.
If the context does not contain enough information, say so clearly."""

USER_PROMPT = """Context from service manuals:

{context}

Question: {question}

Provide a technical answer based on the context above. Include:
1. Direct answer to the question
2. Relevant part numbers if applicable
3. Step-by-step procedures if applicable
4. Safety warnings if applicable
5. Source citations (reference the context items as [Source N])"""

NO_CONTEXT_ANSWER = "The indexed manuals contain no passage relevant to this question."


class AnswerGenerator(Protocol):
    name: str

    def generate(self, question: str, context_results: list[dict]) -> str: ...


def build_context(results: list[dict]) -> str:
    """Number each retrieved passage so answers can cite [Source N]."""
    parts = []
    for number, result in enumerate(results, start=1):
        section = result.get("section_name") or "unknown section"
        model = result.get("model") or ""
        parts.append(f"[Source {number}] ({model}, {section})\n{result.get('content', '')}\n")
    return "\n---\n".join(parts)


class OpenAIAnswerGenerator:
    """Chat-completion answer grounded on the retrieved passages."""

    def __init__(self, settings: Settings | None = None, client: OpenAI | None = None):
        settings = settings or get_settings()
        self.name = settings.openai_model
        self.client = client or OpenAI(api_key=settings.require_openai_key())

    def generate(self, question: str, context_results: list[dict]) -> str:
        if not context_results:
            return NO_CONTEXT_ANSWER
        prompt = USER_PROMPT.format(context=build_context(context_results), question=question)
        response = self.client.chat.completions.create(
            model=self.name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content


class ExtractiveAnswerGenerator:
    """Deterministic offline answerer.

    Ranks the sentences and table rows of the retrieved passages by how many
    question terms they share, and returns the best ones in passage order with
    [Source N] citations. It never adds text that is not in the passages.
    """

    name = "extractive-offline"

    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences

    def generate(self, question: str, context_results: list[dict]) -> str:
        question_terms = content_tokens(question)
        candidates = []
        for source_number, result in enumerate(context_results, start=1):
            for position, sentence in enumerate(split_sentences(result.get("content", ""))):
                overlap = len(question_terms & content_tokens(sentence))
                if overlap:
                    candidates.append((overlap, -source_number, -position, sentence, source_number))

        if not candidates:
            return NO_CONTEXT_ANSWER

        best = sorted(candidates, reverse=True)[: self.max_sentences]
        best.sort(key=lambda c: (-c[1], -c[2]))
        return " ".join(f"{sentence} [Source {source}]" for *_, sentence, source in best)


def create_answer_generator(settings: Settings | None = None) -> AnswerGenerator:
    settings = settings or get_settings()
    if settings.use_openai:
        return OpenAIAnswerGenerator(settings)
    return ExtractiveAnswerGenerator()
