"""TF-IDF turn graph, TextRank, and LexRank.

Each document is a complete weighted graph over its turns. Edge weights are
cosine similarities between L2-normalized TF-IDF vectors.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer

DAMPING: float = 0.85


def fit_vectorizer(*, train_turns: list[str]) -> TfidfVectorizer:
    """Fit TF-IDF on training turns only, never on validation or test."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    vectorizer.fit(train_turns)
    return vectorizer


def similarity_matrix(
    *,
    vectorizer: TfidfVectorizer,
    turns: list[str],
) -> NDArray[np.float64]:
    """Return pairwise cosine similarity for turns with zero self-similarity."""
    if len(turns) < 2:
        return np.zeros((len(turns), len(turns)))

    vectors = vectorizer.transform(turns)
    similarity = (vectors @ vectors.T).toarray()
    np.fill_diagonal(similarity, 0.0)
    return np.clip(similarity, 0.0, 1.0)


def pagerank(
    *,
    similarity: NDArray[np.float64],
    damping: float = DAMPING,
    threshold: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> NDArray[np.float64]:
    """Run PageRank on the optionally thresholded similarity graph.

    A threshold of zero gives the full graph used by TextRank. A positive
    threshold drops weak edges and gives LexRank. Dangling nodes spread their
    mass uniformly.
    """
    weights = similarity.copy()
    if threshold > 0.0:
        weights[weights < threshold] = 0.0
    np.fill_diagonal(weights, 0.0)

    n = len(weights)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    row_sums = weights.sum(axis=1)
    transition = np.zeros_like(weights)
    has_edge = row_sums > 0
    transition[has_edge] = weights[has_edge] / row_sums[has_edge, None]

    scores = np.full(n, 1.0 / n)
    dangling = ~has_edge
    for _ in range(max_iter):
        dangling_mass = scores[dangling].sum() / n
        updated = (1.0 - damping) / n + damping * (
            transition.T @ scores + dangling_mass
        )
        if np.abs(updated - scores).sum() < tol:
            scores = updated
            break
        scores = updated
    return scores


def select_top_k(*, scores: NDArray[np.float64], k: int) -> list[int]:
    """Return the top-k turn indices in original document order."""
    k = min(k, len(scores))
    # Stable sorting makes score ties deterministic, with earlier turns winning.
    ranked = np.argsort(-scores, kind="stable")[:k]
    return sorted(int(index) for index in ranked)
