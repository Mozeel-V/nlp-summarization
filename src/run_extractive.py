"""Tune and evaluate TextRank and LexRank extractive baselines on SAMSum.

The pipeline fits TF-IDF on training turns only, builds a complete
cosine-similarity graph per dialogue, and ranks turns with PageRank. TextRank
uses the full graph. LexRank removes weak edges before ranking. The script
tunes k and the LexRank threshold on validation, then evaluates both variants
on test.

Run from the repository root:

    python src/run_extractive.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer

from config import DATASET_NAME, OUTPUT_DIR
from data import Document, load_docs
from graph import (
    DAMPING,
    fit_vectorizer,
    pagerank,
    select_top_k,
    similarity_matrix,
)
from metrics import Metrics, evaluate, mean_metrics, selection_score

SEED: int = 42
THRESHOLDS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)

Variant = Literal["lexrank", "textrank"]


class Prediction(Metrics):
    """One test extract and its metrics."""

    extract: str
    id: str
    k: int
    reference: str
    selected_turn_indices: list[int]
    source: str
    threshold: float
    variant: Variant


def _summarize(
    *,
    turns: Sequence[str],
    scores: NDArray[np.float64],
    k: int,
) -> tuple[str, list[int]]:
    indices = select_top_k(scores=scores, k=k)
    return " ".join(turns[index] for index in indices), indices


def _tune_textrank(
    *,
    docs: Sequence[Document],
    vectorizer: TfidfVectorizer,
    k_grid: tuple[int, ...],
) -> dict[int, Metrics]:
    rows: dict[int, list[Metrics]] = {k: [] for k in k_grid}
    for doc in docs:
        similarity = similarity_matrix(
            vectorizer=vectorizer,
            turns=doc["turns"],
        )
        scores = pagerank(similarity=similarity, damping=DAMPING)
        for k in k_grid:
            hypothesis, _ = _summarize(turns=doc["turns"], scores=scores, k=k)
            rows[k].append(
                evaluate(
                    hypothesis=hypothesis,
                    reference=doc["summary"],
                    source=doc["source"],
                )
            )
    return {k: mean_metrics(rows=values) for k, values in rows.items()}


def _tune_lexrank(
    *,
    docs: Sequence[Document],
    vectorizer: TfidfVectorizer,
    k_grid: tuple[int, ...],
    thresholds: tuple[float, ...],
) -> dict[float, dict[int, Metrics]]:
    table: dict[float, dict[int, Metrics]] = {}
    for threshold in thresholds:
        rows: dict[int, list[Metrics]] = {k: [] for k in k_grid}
        for doc in docs:
            similarity = similarity_matrix(
                vectorizer=vectorizer,
                turns=doc["turns"],
            )
            scores = pagerank(
                similarity=similarity,
                damping=DAMPING,
                threshold=threshold,
            )
            for k in k_grid:
                hypothesis, _ = _summarize(
                    turns=doc["turns"],
                    scores=scores,
                    k=k,
                )
                rows[k].append(
                    evaluate(
                        hypothesis=hypothesis,
                        reference=doc["summary"],
                        source=doc["source"],
                    )
                )
        table[threshold] = {
            k: mean_metrics(rows=values) for k, values in rows.items()
        }
    return table


def _evaluate_variant(
    *,
    docs: Sequence[Document],
    vectorizer: TfidfVectorizer,
    k: int,
    threshold: float,
    variant: Variant,
) -> tuple[Metrics, list[Prediction]]:
    rows: list[Metrics] = []
    predictions: list[Prediction] = []
    for doc in docs:
        similarity = similarity_matrix(
            vectorizer=vectorizer,
            turns=doc["turns"],
        )
        scores = pagerank(
            similarity=similarity,
            damping=DAMPING,
            threshold=threshold,
        )
        hypothesis, indices = _summarize(
            turns=doc["turns"],
            scores=scores,
            k=k,
        )
        metrics = evaluate(
            hypothesis=hypothesis,
            reference=doc["summary"],
            source=doc["source"],
        )
        rows.append(metrics)
        prediction: Prediction = {
            "extract": hypothesis,
            "id": doc["id"],
            "k": k,
            "reference": doc["summary"],
            "selected_turn_indices": indices,
            "source": doc["source"],
            "threshold": threshold,
            "variant": variant,
            **metrics,
        }
        predictions.append(prediction)
    return mean_metrics(rows=rows), predictions


def _best_threshold_key(
    *,
    table: Mapping[float, Mapping[int, Metrics]],
    k_grid: tuple[int, ...],
) -> tuple[float, int]:
    """Return the (threshold, k) pair with the best mean validation ROUGE."""
    return max(
        ((threshold, k) for threshold in table for k in k_grid),
        key=lambda pair: selection_score(metrics=table[pair[0]][pair[1]]),
    )


def _write_jsonl(*, path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "extractive",
    )
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    k_grid: tuple[int, ...] = tuple(range(1, args.max_k + 1))
    docs = load_docs(dataset_name=args.dataset)
    train_turns = [turn for doc in docs["train"] for turn in doc["turns"]]
    vectorizer = fit_vectorizer(train_turns=train_turns)
    print(f"TF-IDF vocabulary: {len(vectorizer.vocabulary_)} terms")

    print("Tuning TextRank on validation ...")
    textrank_validation = _tune_textrank(
        docs=docs["validation"],
        vectorizer=vectorizer,
        k_grid=k_grid,
    )
    best_textrank_k = max(
        k_grid,
        key=lambda k: selection_score(metrics=textrank_validation[k]),
    )

    print("Tuning LexRank on validation ...")
    lexrank_validation = _tune_lexrank(
        docs=docs["validation"],
        vectorizer=vectorizer,
        k_grid=k_grid,
        thresholds=THRESHOLDS,
    )
    best_lexrank_threshold, best_lexrank_k = _best_threshold_key(
        table=lexrank_validation,
        k_grid=k_grid,
    )

    print(
        f"TextRank: k={best_textrank_k}, "
        f"val ROUGE-1="
        f"{textrank_validation[best_textrank_k]['rouge1']:.4f}"
    )
    print(
        f"LexRank: threshold={best_lexrank_threshold:.2f}, "
        f"k={best_lexrank_k}, "
        f"val ROUGE-1="
        f"{lexrank_validation[best_lexrank_threshold][best_lexrank_k]['rouge1']:.4f}"
    )

    print("Evaluating on test ...")
    textrank_test, textrank_predictions = _evaluate_variant(
        docs=docs["test"],
        vectorizer=vectorizer,
        k=best_textrank_k,
        threshold=0.0,
        variant="textrank",
    )
    lexrank_test, lexrank_predictions = _evaluate_variant(
        docs=docs["test"],
        vectorizer=vectorizer,
        k=best_lexrank_k,
        threshold=best_lexrank_threshold,
        variant="lexrank",
    )

    textrank_val = textrank_validation[best_textrank_k]
    lexrank_val = lexrank_validation[best_lexrank_threshold][best_lexrank_k]
    better_variant: Variant = (
        "textrank"
        if selection_score(metrics=textrank_val)
        >= selection_score(metrics=lexrank_val)
        else "lexrank"
    )

    results: dict[str, object] = {
        "method": "extractive",
        "dataset": args.dataset,
        "seed": SEED,
        "selection_unit": "utterance (speaker prefix kept)",
        "damping": DAMPING,
        "idf_fit_split": "train",
        "rouge_note": (
            "macro-average per-document F1 on lower-cased alphanumeric "
            "tokens, no stemming"
        ),
        "k_grid": list(k_grid),
        "threshold_grid": list(THRESHOLDS),
        "validation": {
            "textrank": {
                "by_k": textrank_validation,
                "best_k": best_textrank_k,
                "best_metrics": textrank_val,
            },
            "lexrank": {
                "by_threshold_and_k": lexrank_validation,
                "best_threshold": best_lexrank_threshold,
                "best_k": best_lexrank_k,
                "best_metrics": lexrank_val,
            },
        },
        "test": {"textrank": textrank_test, "lexrank": lexrank_test},
        "better_variant": better_variant,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_jsonl(
        path=args.output_dir / "textrank_test_predictions.jsonl",
        rows=textrank_predictions,
    )
    _write_jsonl(
        path=args.output_dir / "lexrank_test_predictions.jsonl",
        rows=lexrank_predictions,
    )

    print("\nTest results (macro-average):")
    print(f"{'variant':<10} {'R1':>7} {'R2':>7} {'RL':>7} {'CR':>7} {'ED':>7}")
    for name, metrics in (
        ("TextRank", textrank_test),
        ("LexRank", lexrank_test),
    ):
        print(
            f"{name:<10} {metrics['rouge1']:>7.4f} "
            f"{metrics['rouge2']:>7.4f} {metrics['rougeL']:>7.4f} "
            f"{metrics['compression_ratio']:>7.4f} "
            f"{metrics['extractive_density']:>7.4f}"
        )
    print(f"\nBetter variant: {better_variant}")
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
