"""Train and evaluate the abstractive pointer-generator on SAMSum.

Run from the repository root:

    python src/run_abstractive.py --device cuda

Use ``--eval-only`` to evaluate saved checkpoints without training. The
checkpoints and vocabulary must already exist in ``--checkpoint-dir``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from checkpoints import load_model, load_vocab
from config import (
    CHECKPOINT_DIR,
    Config,
    OUTPUT_DIR,
    prepare_output_dir,
    resolve_device,
    set_seed,
)
from data import DATASET_NAME, load_docs
from evaluation import (
    dump_predictions,
    dump_results,
    eval_split,
    make_results,
    make_table,
    plot_curves,
    report_stats,
    show_table,
)
from model import make_model
from trainer import Trainer
from vocab import make_data, make_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--iterations", type=int, default=33_000)
    parser.add_argument("--iterations-cov", type=int, default=3_000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-mode",
        choices=["iterations", "epochs"],
        default="iterations",
    )
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--min-steps", type=int, default=10)
    parser.add_argument("--no-repeat-ngram", type=int, default=0)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=CHECKPOINT_DIR,
        help="Directory holding pg_nocov.pt, pg_cov.pt, and vocab.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "abstractive",
        help="Directory for results and predictions.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Ignore existing training checkpoints and start from scratch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(device=args.device)

    cfg = Config(
        dataset=args.dataset,
        epochs_total=args.epochs,
        iterations_cov=args.iterations_cov,
        iterations_total=args.iterations,
        seed=args.seed,
        train_mode=args.train_mode,
        resume=args.resume,
    )
    checkpoint_dir = prepare_output_dir(path=args.checkpoint_dir)
    output_dir = prepare_output_dir(path=args.output_dir)

    print("output directory:", output_dir)
    print("checkpoint directory:", checkpoint_dir)
    set_seed(seed=cfg.seed)

    docs = load_docs(dataset_name=cfg.dataset)
    n_train = len(docs["train"])
    if args.eval_only:
        vocab = load_vocab(path=checkpoint_dir / "vocab.json")
        examples = make_examples(
            docs=docs,
            vocab=vocab,
            cfg=cfg,
            splits=("test",),
        )
        model_nocov = load_model(
            checkpoint_path=checkpoint_dir / "pg_nocov.pt",
            vocab=vocab,
            cfg=cfg,
            device=device,
        )
        model_cov = load_model(
            checkpoint_path=checkpoint_dir / "pg_cov.pt",
            vocab=vocab,
            cfg=cfg,
            device=device,
        )
        trainer_history: list[dict] = []
    else:
        vocab, examples = make_data(docs=docs, cfg=cfg)
        schedule = cfg.schedule(n_train=n_train)
        print(schedule.describe(mode=cfg.train_mode))

        trainer = Trainer(
            cfg=cfg,
            schedule=schedule,
            examples=examples,
            device=device,
            out_dir=checkpoint_dir,
        )
        model = make_model(vocab_size=len(vocab), cfg=cfg, device=device)
        print(
            "parameters: "
            f"{sum(parameter.numel() for parameter in model.parameters()):,}"
        )
        trainer.fit_nocov(model=model)
        model_cov = trainer.fit_cov(vocab_size=len(vocab))
        model_nocov = trainer.load_nocov(vocab_size=len(vocab))
        trainer.dump_vocab(vocab=vocab)
        trainer_history = trainer.history

    test_nocov, preds_nocov = eval_split(
        model=model_nocov,
        vocab=vocab,
        use_cov=False,
        tag="pointer_generator",
        docs=docs,
        examples=examples,
        device=device,
        limit=args.limit,
        beam=args.beam,
        min_steps=args.min_steps,
        no_repeat_ngram=args.no_repeat_ngram,
    )
    test_cov, preds_cov = eval_split(
        model=model_cov,
        vocab=vocab,
        use_cov=True,
        tag="pointer_generator_coverage",
        docs=docs,
        examples=examples,
        device=device,
        limit=args.limit,
        beam=args.beam,
        min_steps=args.min_steps,
        no_repeat_ngram=args.no_repeat_ngram,
    )
    dump_predictions(
        named_predictions={
            "pointer_generator": preds_nocov,
            "pointer_generator_coverage": preds_cov,
        },
        out_dir=output_dir,
    )

    table = make_table(
        extractive_results=OUTPUT_DIR / "extractive" / "results.json",
        test_nocov=test_nocov,
        test_cov=test_cov,
    )
    show_table(table=table)
    dump_results(
        results=make_results(
            cfg=cfg,
            schedule=cfg.schedule(n_train=n_train),
            vocab=vocab,
            test_nocov=test_nocov,
            test_cov=test_cov,
        ),
        out_dir=output_dir,
    )

    if trainer_history:
        plot_curves(history=trainer_history, out_dir=output_dir)
    report_stats(preds_nocov=preds_nocov, preds_cov=preds_cov)


if __name__ == "__main__":
    main()
