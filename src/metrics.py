"""ROUGE-1/2/L, compression ratio, and extractive density.

ROUGE is computed per document on lower-cased alphanumeric tokens without
stemming and macro-averaged over documents. Extractive density is the fraction
of content words in the generated summary that also occur in the source. For an
extractive system it is 1.0 by construction.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TypedDict

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN_RE: re.Pattern[str] = re.compile(r"[a-z0-9]+")


class Metrics(TypedDict):
    """Per-document or averaged metric values."""

    compression_ratio: float
    extractive_density: float
    rouge1: float
    rouge2: float
    rougeL: float


def tokenize(*, text: str) -> list[str]:
    """Lower-case text and return alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


def _ngrams(*, tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _f1(*, overlap: int, n_hyp: int, n_ref: int) -> float:
    if n_hyp == 0 or n_ref == 0 or overlap == 0:
        return 0.0
    precision = overlap / n_hyp
    recall = overlap / n_ref
    return 2.0 * precision * recall / (precision + recall)


def rouge_n(*, hypothesis: str, reference: str, n: int) -> float:
    """Return ROUGE-n F1 for one document."""
    hyp_ngrams = _ngrams(tokens=tokenize(text=hypothesis), n=n)
    ref_ngrams = _ngrams(tokens=tokenize(text=reference), n=n)
    overlap = sum((hyp_ngrams & ref_ngrams).values())
    return _f1(
        overlap=overlap,
        n_hyp=sum(hyp_ngrams.values()),
        n_ref=sum(ref_ngrams.values()),
    )


def _lcs_length(*, a: list[str], b: list[str]) -> int:
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def rouge_l(*, hypothesis: str, reference: str) -> float:
    """Return ROUGE-L F1 for one document."""
    hyp_tokens = tokenize(text=hypothesis)
    ref_tokens = tokenize(text=reference)
    lcs = _lcs_length(a=hyp_tokens, b=ref_tokens)
    return _f1(overlap=lcs, n_hyp=len(hyp_tokens), n_ref=len(ref_tokens))


def _content_tokens(*, text: str) -> list[str]:
    return [
        token for token in tokenize(text=text) if token not in ENGLISH_STOP_WORDS
    ]


def extractive_density(*, hypothesis: str, source: str) -> float:
    """Return the fraction of summary content words present in the source."""
    hyp_content = _content_tokens(text=hypothesis)
    if not hyp_content:
        return 0.0
    source_content = Counter(_content_tokens(text=source))
    return (
        sum((Counter(hyp_content) & source_content).values()) / len(hyp_content)
    )


def evaluate(*, hypothesis: str, reference: str, source: str) -> Metrics:
    """Compute all metrics for one document."""
    return {
        "compression_ratio": len(tokenize(text=hypothesis))
        / max(1, len(tokenize(text=source))),
        "extractive_density": extractive_density(
            hypothesis=hypothesis,
            source=source,
        ),
        "rouge1": rouge_n(hypothesis=hypothesis, reference=reference, n=1),
        "rouge2": rouge_n(hypothesis=hypothesis, reference=reference, n=2),
        "rougeL": rouge_l(hypothesis=hypothesis, reference=reference),
    }


def mean_metrics(*, rows: list[Metrics]) -> Metrics:
    """Macro-average each metric over documents."""
    count = len(rows)
    if count == 0:
        raise ValueError("Cannot average metrics over zero documents.")
    return {
        "compression_ratio": sum(
            row["compression_ratio"] for row in rows
        )
        / count,
        "extractive_density": sum(
            row["extractive_density"] for row in rows
        )
        / count,
        "rouge1": sum(row["rouge1"] for row in rows) / count,
        "rouge2": sum(row["rouge2"] for row in rows) / count,
        "rougeL": sum(row["rougeL"] for row in rows) / count,
    }


def selection_score(*, metrics: Metrics) -> float:
    """Return the mean of the three ROUGE F1 scores."""
    return (metrics["rouge1"] + metrics["rouge2"] + metrics["rougeL"]) / 3.0
