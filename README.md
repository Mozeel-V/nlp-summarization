# Extractive and Abstractive Dialogue Summarization on SAMSum

Three summarization systems for the SAMSum dialogue corpus, implemented from
scratch in PyTorch.

1. An extractive baseline that ranks speaker turns with TextRank and LexRank
   over TF-IDF similarity graphs.
2. An abstractive pointer-generator with a BiLSTM encoder, Bahdanau attention,
   a copy mechanism, an optional coverage vector, and beam search.
3. A two-stage pipeline that selects turns with LexRank and summarizes them with
   the pointer-generator, optionally fine-tuned on extracted dialogues.

## Results

Test split, 819 dialogues, macro-averaged ROUGE F1 without stemming.

<!-- markdownlint-disable MD013 -->

| System | R1 | R2 | RL | ED | Consistency |
| --- | ---: | ---: | ---: | ---: | ---: |
| TextRank | 27.77 | 8.51 | 21.22 | 1.000 | - |
| LexRank | 28.43 | 8.49 | 21.72 | 1.000 | - |
| Pointer-generator | 36.22 | 14.54 | 30.56 | 0.839 | - |
| Pointer-generator + coverage | 36.61 | 13.23 | 29.97 | 0.879 | - |
| Pipeline, beam 4, minimum 15 steps, no repeated 3-grams | 38.45 | 15.37 | 31.02 | 0.855 | 0.854 |
| Pipeline, fine-tuned on extracted turns | 39.55 | 15.83 | 31.29 | 0.851 | 0.851 |
| Full-dialogue model, matched fine-tune | 39.44 | 15.93 | 31.47 | 0.851 | 0.851 |

<!-- markdownlint-enable MD013 -->

ED is extractive density against the full dialogue. Consistency is the same
metric against the actual model input.

The fine-tuned pipeline and the matched full-dialogue model are within noise of
each other on every ROUGE score, so extraction does not add overlap with the
reference once both models receive the same training budget. It does provide a
shorter input at mild compression and a tunable quality against compression
tradeoff at aggressive settings. At top-8 the extract keeps 54 percent of the
tokens and ROUGE-1 falls from 39.55 to 35.28.

ROUGE is computed per document on lower-cased alphanumeric tokens and
macro-averaged. TF-IDF is fit on training turns only, model selection uses
validation only, and every script seeds Python, NumPy, and PyTorch to 42.

## Setup

Python 3.10 or newer is required. The code is tested on 3.12.

```bash
pip install -e .
```

The dependencies are declared in `pyproject.toml`.

The reported results were produced on Python 3.12 with torch 2.9.1,
scikit-learn 1.8.0, numpy 1.26.4, and pandas 2.2.2.

The dataset is loaded from `knkarthick/samsum` through `datasets.load_dataset`.
Checkpoints are hosted on the Hugging Face Hub:

```bash
hf download moz-v/samsum-pg-checkpoints --local-dir checkpoints \
  --include "pg_nocov.pt" --include "pg_cov.pt" --include "vocab.json"
```

This writes three files:

```text
checkpoints/pg_nocov.pt
checkpoints/pg_cov.pt
checkpoints/vocab.json
```

Run the tests with:

```bash
python -m unittest discover tests
```

Run all commands from the repository root.

## Usage

Extractive baselines. Tunes k and the LexRank threshold on validation, then
evaluates on test.

```bash
python src/run_extractive.py
```

Outputs go to `outputs/extractive/`.

Abstractive model. Trains 33000 iterations without coverage, then 3000 with
coverage, and evaluates both.

```bash
python src/run_abstractive.py --device cuda
```

Evaluation with existing checkpoints:

```bash
python src/run_abstractive.py --device cuda --eval-only
```

Outputs go to `outputs/abstractive/`. Checkpoints are read from and written to
`checkpoints/`. Use `--no-resume` to ignore existing training state.

Two-stage pipeline. Extracts top-20 LexRank turns and decodes them with the new
decoder.

```bash
python src/run_pipeline.py --device cuda
```

Add `--compare-full` to also decode full dialogues in the same run. Outputs go
to `outputs/pipeline/`.

Fine-tuning. Trains the pointer-generator on extracted turns for two epochs and
evaluates on test.

```bash
python src/run_finetune.py --device cuda --train-input extract --m 20
```

Other input modes are `--train-input full` for a matched full-dialogue baseline
and `--train-input mixed` for extracted plus full dialogues. Add
`--filter-noise` to remove greetings and file placeholders before extraction.
Outputs go to `outputs/finetune/`. Use `--eval-only` to re-evaluate a saved
`ft_best.pt`.

## Method

Each dialogue is a complete weighted graph over its speaker turns. Edges are
cosine similarities between L2-normalized TF-IDF vectors. TextRank runs PageRank
on the full graph. LexRank drops edges below a threshold before ranking. The
selected turns keep their speaker prefixes and original order.

The abstractive model uses a single-layer bidirectional LSTM encoder and an
LSTMCell decoder with Bahdanau attention. The pointer mechanism mixes the
vocabulary distribution with the attention distribution, which lets the model
copy source tokens, including out-of-vocabulary tokens. Coverage tracks the
attention history during training and adds a coverage loss. Inference uses beam
search with length normalization and an optional repeated n-gram block.

The pipeline introduces a new input distribution at test time, so the model is
fine-tuned on the same kind of extracted input it will see. Early stopping uses
validation NLL on the matching input. The matched full-dialogue baseline uses
the same epochs, learning rate, batch size, early stopping, and decoder, so the
comparison isolates the effect of extraction.

Training the abstractive model for the full schedule takes several hours on one
GPU. Fine-tuning takes about eight minutes. Inference is the slowest part of the
pipeline because beam search runs one dialogue at a time.
