"""Turn ranking and extraction helpers for the two-stage pipeline."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from data import Document, Record
from graph import pagerank, select_top_k, similarity_matrix

GREETINGS: frozenset[str] = frozenset(
    {
        "hi",
        "hey",
        "hello",
        "good morning",
        "good night",
        "bye",
        "goodbye",
        "see you",
        "see ya",
        "ttyl",
        "gn",
        "gm",
    }
)

_FILE_PLACEHOLDER = re.compile(r"<file_[a-z_]+>")


def clean_turns(*, turns: list[str]) -> list[str]:
    """Drop greetings and file placeholders while keeping speaker prefixes."""
    kept: list[str] = []
    for turn in turns:
        content = turn.split(":", 1)[1].strip() if ":" in turn else turn
        lowered = content.strip().lower().strip(" .!?")
        if not lowered:
            continue
        if _FILE_PLACEHOLDER.fullmatch(lowered):
            continue
        if lowered in GREETINGS:
            continue
        kept.append(turn)
    return kept


def prepare_document(
    *,
    doc: Document,
    filter_noise: bool,
) -> Document:
    """Optionally clean a document, then rebuild its source text."""
    turns = clean_turns(turns=doc["turns"]) if filter_noise else doc["turns"]
    return {
        "id": doc["id"],
        "turns": turns,
        "source": "\n".join(turns),
        "summary": doc["summary"],
    }


def rank_documents(
    *,
    documents: Sequence[Document],
    vectorizer: TfidfVectorizer,
    threshold: float,
) -> list[np.ndarray]:
    """Return PageRank scores for every document."""
    return [
        pagerank(
            similarity=similarity_matrix(
                vectorizer=vectorizer,
                turns=doc["turns"],
            ),
            threshold=threshold,
        )
        for doc in documents
    ]


def extract_turns(
    *,
    doc: Document,
    scores: np.ndarray,
    m: int,
) -> tuple[str, list[int]]:
    """Select the top-m turns and join them in document order."""
    indices = select_top_k(scores=scores, k=m)
    text = "\n".join(doc["turns"][index] for index in indices)
    return text, indices


def build_extracted_docs(
    *,
    documents: Sequence[Document],
    vectorizer: TfidfVectorizer,
    m: int,
    filter_noise: bool,
    threshold: float = 0.5,
) -> list[Record]:
    """Create one source-and-summary record per extracted dialogue."""
    prepared = [
        prepare_document(doc=doc, filter_noise=filter_noise) for doc in documents
    ]
    scores = rank_documents(
        documents=prepared,
        vectorizer=vectorizer,
        threshold=threshold,
    )

    extracted: list[Record] = []
    for doc, score in zip(prepared, scores, strict=False):
        text, _ = extract_turns(doc=doc, scores=score, m=m)
        extracted.append(
            {
                "id": doc["id"],
                "source": text,
                "summary": doc["summary"],
            }
        )
    return extracted


def input_reduction(
    *,
    documents: Sequence[Document],
    vectorizer: TfidfVectorizer,
    m: int,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Measure how much text the extract keeps relative to the dialogues."""
    scores = rank_documents(
        documents=documents,
        vectorizer=vectorizer,
        threshold=threshold,
    )
    token_pattern = re.compile(r"[A-Za-z0-9]+")
    full_tokens = sum(len(token_pattern.findall(doc["source"])) for doc in documents)
    extract_tokens = 0
    turns_kept = 0
    turns_total = 0
    for doc, score in zip(documents, scores, strict=False):
        text, indices = extract_turns(doc=doc, scores=score, m=m)
        extract_tokens += len(token_pattern.findall(text))
        turns_kept += len(indices)
        turns_total += len(doc["turns"])
    return {
        "token_ratio": extract_tokens / full_tokens,
        "turn_ratio": turns_kept / turns_total,
    }
