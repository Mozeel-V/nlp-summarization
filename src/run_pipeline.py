"""Extract-then-abstract summarization on SAMSum.

The pipeline ranks dialogue turns with LexRank, keeps the top ``m`` turns, and
feeds them to a trained pointer-generator. The default decoder uses beam 4, a
minimum of 15 steps, and blocks repeated 3-grams.

Run from the repository root:

    python src/run_pipeline.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from checkpoints import load_model, load_vocab
from config import (
    CHECKPOINT_DIR,
    Config,
    DATASET_NAME,
    OUTPUT_DIR,
    prepare_output_dir,
    resolve_device,
    set_seed,
)
from data import load_docs
from evaluation import evaluate_sources
from extraction import extract_turns, input_reduction, rank_documents
from graph import fit_vectorizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_DIR / "pg_nocov.pt",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=CHECKPOINT_DIR / "vocab.json",
    )
    parser.add_argument("--weights", choices=["auto", "best", "last"], default="auto")
    parser.add_argument("--use-cov", action="store_true")
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--min-steps", type=int, default=15)
    parser.add_argument("--no-repeat-ngram", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="0 uses the full split.")
    parser.add_argument(
        "--compare-full",
        action="store_true",
        help="Also decode full dialogues with the same decoder.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "pipeline",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(device=args.device)
    output_dir = prepare_output_dir(path=args.output_dir)
    set_seed(seed=args.seed)

    cfg = Config(seed=args.seed)
    docs = load_docs(dataset_name=args.dataset)
    vectorizer = fit_vectorizer(
        train_turns=[turn for doc in docs["train"] for turn in doc["turns"]]
    )
    vocab = load_vocab(path=args.vocab)
    model = load_model(
        checkpoint_path=args.checkpoint,
        vocab=vocab,
        cfg=cfg,
        device=device,
        weights=args.weights,
    )

    test_docs = docs["test"][: args.limit] if args.limit else docs["test"]
    scores = rank_documents(
        documents=test_docs,
        vectorizer=vectorizer,
        threshold=args.threshold,
    )
    extracts = [
        extract_turns(doc=doc, scores=score, m=args.m)
        for doc, score in zip(test_docs, scores)
    ]
    sources = [text for text, _ in extracts]
    indices = [selected for _, selected in extracts]

    metrics, predictions = evaluate_sources(
        model=model,
        vocab=vocab,
        documents=test_docs,
        sources=sources,
        device=device,
        beam=args.beam,
        min_steps=args.min_steps,
        no_repeat_ngram=args.no_repeat_ngram,
        use_cov=args.use_cov,
        tag="pipeline",
        selected_indices=indices,
    )

    table = {}
    if args.compare_full:
        print("Decoding full dialogues for comparison ...")
        full_metrics, _ = evaluate_sources(
            model=model,
            vocab=vocab,
            documents=test_docs,
            sources=[doc["source"] for doc in test_docs],
            device=device,
            beam=args.beam,
            min_steps=args.min_steps,
            no_repeat_ngram=args.no_repeat_ngram,
            use_cov=args.use_cov,
            tag="full-dialogue",
        )
        table["Pointer-Generator, full dialogue"] = full_metrics
    table[f"LexRank m={args.m}, extracted turns"] = metrics

    print(
        f"\n{'system':<40} {'R1':>7} {'R2':>7} {'RL':>7} "
        f"{'CR':>7} {'ED':>7} {'Cons':>7}"
    )
    for name, row in table.items():
        print(
            f"{name:<40} {row['rouge1']:>7.4f} "
            f"{row['rouge2']:>7.4f} {row['rougeL']:>7.4f} "
            f"{row['compression_ratio']:>7.4f} "
            f"{row['extractive_density']:>7.4f} "
            f"{row['consistency_vs_source']:>7.4f}"
        )
    print(
        "ED is extractive density against the full dialogue. "
        "Cons is the same function against the model input."
    )

    reduction = input_reduction(
        documents=test_docs,
        vectorizer=vectorizer,
        m=args.m,
        threshold=args.threshold,
    )
    print(
        f"\nExtract keeps {reduction['token_ratio']:.3f} of the tokens "
        f"and {reduction['turn_ratio']:.3f} of the turns."
    )

    results = {
        "dataset": args.dataset,
        "checkpoint": str(args.checkpoint),
        "weights": args.weights,
        "use_cov": args.use_cov,
        "m": args.m,
        "threshold": args.threshold,
        "beam": args.beam,
        "min_steps": args.min_steps,
        "no_repeat_ngram": args.no_repeat_ngram,
        "dialogues": len(test_docs),
        "input_reduction": reduction,
        "test": table,
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "pipeline_test_predictions.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("saved", output_dir / "results.json")


if __name__ == "__main__":
    main()
