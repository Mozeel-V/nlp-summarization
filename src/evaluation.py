"""Evaluation, prediction dumps, and training curves."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from data import Document, Record
from decode import summarize_ids, summarize_text
from metrics import (
    Metrics,
    evaluate,
    extractive_density,
    mean_metrics,
    tokenize as metric_tokenize,
)
from vocab import Example


class SystemMetrics(Metrics):
    """Averaged metrics with faithfulness against the model input."""

    consistency_vs_source: float


def eval_split(
    *,
    model,
    vocab,
    use_cov: bool,
    tag: str,
    docs: Mapping[str, Sequence[Document]],
    examples: Mapping[str, Sequence[Example]],
    device,
    limit: int | None = None,
    beam: int = 4,
    min_steps: int = 10,
    no_repeat_ngram: int = 0,
    split: str = "test",
) -> tuple[Metrics, list[dict]]:
    """Decode one split and return averaged metrics and per-document rows."""
    split_docs = docs[split][:limit]
    split_examples = examples[split][:limit]
    rows: list[Metrics] = []
    predictions: list[dict] = []
    start_time = time.time()

    for index, (doc, example) in enumerate(
        zip(split_docs, split_examples, strict=False)
    ):
        summary = summarize_ids(
            model=model,
            example=example,
            vocab=vocab,
            device=device,
            beam=beam,
            min_steps=min_steps,
            no_repeat_ngram=no_repeat_ngram,
            use_cov=use_cov,
        )
        metrics = evaluate(
            hypothesis=summary,
            reference=doc["summary"],
            source=doc["source"],
        )
        rows.append(metrics)
        predictions.append(
            {
                "id": doc["id"],
                "variant": tag,
                "source": doc["source"],
                "reference": doc["summary"],
                "summary": summary,
                **metrics,
            }
        )
        if (index + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(
                f"[{tag}] {index + 1}/{len(split_docs)}  {elapsed:.0f}s",
                flush=True,
            )

    return mean_metrics(rows=rows), predictions


def evaluate_sources(
    *,
    model,
    vocab,
    documents: Sequence[Record],
    sources: list[str],
    device,
    beam: int = 4,
    min_steps: int = 10,
    no_repeat_ngram: int = 0,
    use_cov: bool = False,
    tag: str = "system",
    selected_indices: list[list[int]] | None = None,
) -> tuple[SystemMetrics, list[dict]]:
    """Summarize arbitrary sources and score them against the references."""
    if len(documents) != len(sources):
        raise ValueError("documents and sources must have the same length.")

    rows: list[Metrics] = []
    consistencies: list[float] = []
    predictions: list[dict] = []
    pairs = list(zip(documents, sources, strict=False))
    for position, (doc, source) in enumerate(tqdm(pairs, desc=tag, leave=False)):
        summary = summarize_text(
            model=model,
            vocab=vocab,
            text=source,
            device=device,
            beam=beam,
            min_steps=min_steps,
            no_repeat_ngram=no_repeat_ngram,
            use_cov=use_cov,
        )
        metrics = evaluate(
            hypothesis=summary,
            reference=doc["summary"],
            source=doc["source"],
        )
        consistency = extractive_density(hypothesis=summary, source=source)
        rows.append(metrics)
        consistencies.append(consistency)
        prediction = {
            "id": doc["id"],
            "source": source,
            "reference": doc["summary"],
            "summary": summary,
            "consistency_vs_source": consistency,
            **metrics,
        }
        if selected_indices is not None:
            prediction["selected_turn_indices"] = selected_indices[position]
        predictions.append(prediction)

    averaged_base = mean_metrics(rows=rows)
    averaged: SystemMetrics = {
        "compression_ratio": averaged_base["compression_ratio"],
        "extractive_density": averaged_base["extractive_density"],
        "rouge1": averaged_base["rouge1"],
        "rouge2": averaged_base["rouge2"],
        "rougeL": averaged_base["rougeL"],
        "consistency_vs_source": float(np.mean(consistencies)),
    }
    return averaged, predictions


def dump_predictions(*, named_predictions: dict, out_dir: Path) -> None:
    """Write one JSONL file per system."""
    for name, predictions in named_predictions.items():
        path = Path(out_dir) / f"{name}_test_predictions.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def plot_curves(*, history: list[dict], out_dir: Path) -> None:
    """Plot training and validation loss for both phases."""
    phase_one = [row for row in history if row["phase"] == "pg"]
    phase_two = [row for row in history if row["phase"] == "cov"]

    figure, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    axes[0].plot(
        [row["step"] for row in phase_one],
        [row["train_nll"] for row in phase_one],
        label="train (running average)",
    )
    axes[0].plot(
        [row["step"] for row in phase_one],
        [row["val_nll"] for row in phase_one],
        label="validation",
    )
    axes[0].plot(
        [row["step"] for row in phase_two],
        [row["val_nll"] for row in phase_two],
        "o-",
        label="validation (coverage phase)",
    )
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("NLL")
    axes[0].legend()
    axes[0].set_title("Primary loss")

    axes[1].plot(
        [row["step"] for row in phase_two],
        [row["val_cov"] for row in phase_two],
        "o-",
    )
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("coverage loss")
    axes[1].set_title("Validation coverage loss (phase 2)")

    plt.tight_layout()
    plt.savefig(Path(out_dir) / "training_curves.png", dpi=130)
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(figure)


def make_table(
    *,
    extractive_results: Path | None,
    test_nocov: Metrics,
    test_cov: Metrics,
) -> dict[str, Metrics]:
    """Combine extractive and abstractive results into one table."""
    table: dict[str, Metrics] = {}
    if extractive_results is not None and Path(extractive_results).exists():
        print("using extractive results from", extractive_results)
        data = json.loads(Path(extractive_results).read_text())
        table["TextRank"] = data["test"]["textrank"]
        table["LexRank"] = data["test"]["lexrank"]
    else:
        print(
            "Extractive results not found. Run src/run_extractive.py first "
            "to include those rows."
        )
    table["Pointer-Generator"] = test_nocov
    table["Pointer-Generator + Coverage"] = test_cov
    return table


def show_table(*, table: Mapping[str, Metrics]) -> None:
    print(f"\n{'system':<30} {'R1':>7} {'R2':>7} {'RL':>7} " f"{'CR':>7} {'ED':>7}")
    for name, metrics in table.items():
        print(
            f"{name:<30} {metrics['rouge1']:>7.4f} "
            f"{metrics['rouge2']:>7.4f} "
            f"{metrics['rougeL']:>7.4f} "
            f"{metrics['compression_ratio']:>7.4f} "
            f"{metrics['extractive_density']:>7.4f}"
        )


def make_results(
    *,
    cfg,
    schedule,
    vocab,
    test_nocov: Metrics,
    test_cov: Metrics,
) -> dict:
    return {
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "train_mode": cfg.train_mode,
        "total_iters": schedule.total_iters,
        "phase1_iters": schedule.phase1_iters,
        "coverage_iters": schedule.cov_iters,
        "iters_per_epoch": schedule.iters_per_epoch,
        "batch_size": cfg.batch_size,
        "vocab_size": len(vocab),
        "hyperparameters": cfg.hparams(),
        "test": {
            "pointer_generator": test_nocov,
            "pointer_generator_coverage": test_cov,
        },
        "metrics_note": (
            "macro-average per-document F1 on lower-cased alphanumeric "
            "tokens without stemming"
        ),
    }


def dump_results(*, results: dict, out_dir: Path) -> None:
    path = Path(out_dir) / "results.json"
    path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("saved", path)


def _ngrams(*, tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def duplicate_percent(*, text: str, n: int) -> float:
    grams = _ngrams(tokens=metric_tokenize(text=text), n=n)
    if not grams:
        return 0.0
    return 100.0 * (1 - len(set(grams)) / len(grams))


def novel_percent(*, text: str, source: str, n: int) -> float:
    grams = _ngrams(tokens=metric_tokenize(text=text), n=n)
    if not grams:
        return 0.0
    source_grams = set(_ngrams(tokens=metric_tokenize(text=source), n=n))
    return 100.0 * sum(gram not in source_grams for gram in grams) / len(grams)


def report_stats(
    *,
    preds_nocov: list[dict],
    preds_cov: list[dict],
    n_examples: int = 5,
) -> None:
    """Print repetition, novelty, and a few side-by-side examples."""
    systems = {
        "reference": [(row["reference"], row["source"]) for row in preds_nocov],
        "PG": [(row["summary"], row["source"]) for row in preds_nocov],
        "PG + coverage": [(row["summary"], row["source"]) for row in preds_cov],
    }

    print("% duplicate n-grams within a summary (lower is less repetition)")
    print(f"{'':<16}" + "".join(f"{n}-gram".rjust(9) for n in (1, 2, 3, 4)))
    for name, items in systems.items():
        duplicate_means = [
            np.mean([duplicate_percent(text=text, n=n) for text, _ in items])
            for n in (1, 2, 3, 4)
        ]
        values = "".join(f"{value:9.2f}" for value in duplicate_means)
        print(f"{name:<16}{values}")

    print("\n% novel n-grams (not in source, higher is more abstractive)")
    for name, items in systems.items():
        novel_means = [
            np.mean(
                [novel_percent(text=text, source=source, n=n) for text, source in items]
            )
            for n in (1, 2, 3, 4)
        ]
        values = "".join(f"{value:9.2f}" for value in novel_means)
        print(f"{name:<16}{values}")

    for first, second in list(zip(preds_nocov, preds_cov, strict=False))[:n_examples]:
        print(
            "\n"
            + "=" * 90
            + f"\n{first['source']}\n"
            + f"--- reference : {first['reference']}\n"
            + f"--- PG        : {first['summary']}\n"
            + f"--- PG + cov  : {second['summary']}"
        )
