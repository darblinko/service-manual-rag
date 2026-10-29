"""Tokenizing helpers shared by the offline embedder, answerer and judge."""

import re

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

STOPWORDS = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "all",
        "also",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "out",
        "over",
        "own",
        "same",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    ]
)


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens; keeps part numbers such as cp-200 and decimals such as 0.25."""
    return TOKEN_PATTERN.findall(text.lower())


def content_tokens(text: str) -> set[str]:
    """Distinct tokens with stopwords removed."""
    return {token for token in tokenize(text) if token not in STOPWORDS}


def split_sentences(text: str) -> list[str]:
    """Split into lines, then split prose lines into sentences. Table rows stay whole."""
    sentences = []
    for line in text.splitlines():
        line = line.strip()
        if not line or set(line) <= set("|-: "):
            continue
        if line.startswith("|"):
            sentences.append(line)
            continue
        sentences.extend(part.strip() for part in SENTENCE_BOUNDARY.split(line) if part.strip())
    return sentences
