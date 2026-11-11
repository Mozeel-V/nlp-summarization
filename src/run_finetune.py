"""Fine-tune the pointer-generator on extracted or full dialogues.

The extracts come from LexRank over turn-level TF-IDF graphs. Fine-tuning on
the same input distribution used at test time is what makes the two-stage
pipeline competitive with a full-dialogue model.

Run from the repository root:

    python src/run_finetune.py --train-input extract --m 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from checkpoints import load_model, load_vocab
from config import (
    CHECKPOINT_DIR,
    Config,
    DATASET_NAME,
    OUTPUT_DIR,
    Schedule,
    prepare_output_dir,
    resolve_device,
    set_seed,
)
from data import load_docs
from evaluation import evaluate_sources
from extraction import build_extracted_docs
from graph import fit_vectorizer
from trainer import Trainer, safe_save
from vocab import make_example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--filter-noise", action="store_true")
    parser.add_argument(
        "--train-input",
        choices=["extract", "full", "mixed"],
        default="extract",
        help="Use extracted turns, full dialogues, or both.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load ft_best.pt from --out-dir and only run test evaluation.",
    )
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
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--min-steps", type=int, default=15)
    parser.add_argument("--no-repeat-ngram", type=int, default=3)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT_DIR / "finetune",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(device=args.device)
    output_dir = prepare_output_dir(path=args.out_dir)
    set_seed(seed=args.seed)

    docs = load_docs(dataset_name=args.dataset)
    train_docs = (
        docs["train"][: args.limit_train]
        if args.limit_train
        else docs["train"]
    )
    val_docs = (
        docs["validation"][: args.limit_val]
        if args.limit_val
        else docs["validation"]
    )
    test_docs = docs["test"][: args.limit_test] if args.limit_test else docs["test"]

    vectorizer = fit_vectorizer(
        train_turns=[turn for doc in docs["train"] for turn in doc["turns"]]
    )
    vocab = load_vocab(path=args.vocab)
    cfg = Config(
        batch_size=args.batch_size,
        lr=args.lr,
        resume=False,
        seed=args.seed,
    )
    model = load_model(
        checkpoint_path=args.checkpoint,
        vocab=vocab,
        cfg=cfg,
        device=device,
    )

    best_path = output_dir / "ft_best.pt"
    total_iters = 0

    if args.eval_only:
        checkpoint = torch.load(
            best_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["state_dict"])
    else:
        if args.train_input == "full":
            train_data = [
                {"id": doc["id"], "source": doc["source"], "summary": doc["summary"]}
                for doc in train_docs
            ]
            val_data = [
                {"id": doc["id"], "source": doc["source"], "summary": doc["summary"]}
                for doc in val_docs
            ]
        else:
            extracted_train = build_extracted_docs(
                documents=train_docs,
                vectorizer=vectorizer,
                m=args.m,
                filter_noise=args.filter_noise,
                threshold=args.threshold,
            )
            if args.train_input == "mixed":
                train_data = extracted_train + [
                    {
                        "id": doc["id"],
                        "source": doc["source"],
                        "summary": doc["summary"],
                    }
                    for doc in train_docs
                ]
            else:
                train_data = extracted_train
            val_data = build_extracted_docs(
                documents=val_docs,
                vectorizer=vectorizer,
                m=args.m,
                filter_noise=args.filter_noise,
                threshold=args.threshold,
            )

        examples = {
            "train": [
                make_example(
                    doc=doc,
                    vocab=vocab,
                    max_enc_steps=cfg.max_enc_steps,
                    max_dec_steps=cfg.max_dec_steps,
                )
                for doc in train_data
            ],
            "validation": [
                make_example(
                    doc=doc,
                    vocab=vocab,
                    max_enc_steps=cfg.max_enc_steps,
                    max_dec_steps=cfg.max_dec_steps,
                )
                for doc in val_data
            ],
        }
        iters_per_epoch = max(1, len(examples["train"]) // cfg.batch_size)
        total_iters = (
            args.iters
            if args.iters
            else max(1, int(round(args.epochs * iters_per_epoch)))
        )
        schedule = Schedule(
            iters_per_epoch=iters_per_epoch,
            total_iters=total_iters,
            cov_iters=0,
            phase1_iters=total_iters,
            eval_every=args.eval_every,
        )
        print(
            f"train examples: {len(examples['train'])}, "
            f"validation: {len(examples['validation'])}, "
            f"iters: {total_iters}"
        )

        trainer = Trainer(
            cfg=cfg,
            schedule=schedule,
            examples=examples,
            device=device,
            out_dir=output_dir,
        )
        model = trainer.run_phase(
            model=model,
            n_iters=total_iters,
            use_cov=False,
            tag="ft",
            keep_best=True,
            start_step=0,
        )
        safe_save(
            payload={"state_dict": model.state_dict()},
            path=best_path,
        )
        print("saved", best_path)

    if args.train_input == "full":
        test_data = [
            {"id": doc["id"], "source": doc["source"], "summary": doc["summary"]}
            for doc in test_docs
        ]
    else:
        test_data = build_extracted_docs(
            documents=test_docs,
            vectorizer=vectorizer,
            m=args.m,
            filter_noise=args.filter_noise,
            threshold=args.threshold,
        )

    metrics, predictions = evaluate_sources(
        model=model,
        vocab=vocab,
        documents=test_docs,
        sources=[doc["source"] for doc in test_data],
        device=device,
        beam=args.beam,
        min_steps=args.min_steps,
        no_repeat_ngram=args.no_repeat_ngram,
        tag="finetune",
    )
    print(
        "test",
        {key: round(float(value), 4) for key, value in metrics.items()},
    )

    results = {
        "config": {
            "m": args.m,
            "threshold": args.threshold,
            "epochs": args.epochs,
            "iters": total_iters,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "filter_noise": args.filter_noise,
            "train_input": args.train_input,
            "seed": args.seed,
            "beam": args.beam,
            "min_steps": args.min_steps,
            "no_repeat_ngram": args.no_repeat_ngram,
        },
        "test": {key: float(value) for key, value in metrics.items()},
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "test_predictions.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("saved", output_dir / "results.json")


if __name__ == "__main__":
    main()
