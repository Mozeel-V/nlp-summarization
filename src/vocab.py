"""Tokenizer, vocabulary, and training examples for the pointer-generator."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from config import PAD, PAD_ID, START, START_ID, STOP, STOP_ID, UNK, UNK_ID
from data import Record, SummaryInput

_TOKEN_RE: re.Pattern[str] = re.compile(r"<[a-z_]+>|\w+|[^\w\s]")


def tokenize(*, text: str) -> list[str]:
    """Word and punctuation tokenization. Newlines become the token <nl>."""
    return _TOKEN_RE.findall(text.lower().replace("\n", " <nl> "))


class Vocab:
    """Index-to-token and token-to-index mapping with four reserved tokens."""

    def __init__(self, *, counter: Counter[str], max_size: int) -> None:
        self.itos: list[str] = [UNK, PAD, START, STOP]
        for word, _ in counter.most_common(max_size - len(self.itos)):
            self.itos.append(word)
        self.stoi: dict[str, int] = {
            word: index for index, word in enumerate(self.itos)
        }

    def __len__(self) -> int:
        return len(self.itos)

    def w2i(self, *, word: str) -> int:
        return self.stoi.get(word, UNK_ID)

    def to_json(self) -> list[str]:
        return list(self.itos)

    @classmethod
    def from_json(cls, *, itos: list[str]) -> Vocab:
        if itos[:4] != [UNK, PAD, START, STOP]:
            raise ValueError("Vocabulary file does not start with reserved tokens.")
        vocab = cls.__new__(cls)
        vocab.itos = list(itos)
        vocab.stoi = {word: index for index, word in enumerate(itos)}
        return vocab


def article2ids(
    *,
    words: list[str],
    vocab: Vocab,
) -> tuple[list[int], list[str]]:
    """Map source tokens to ids, collecting out-of-vocabulary tokens."""
    ids: list[int] = []
    extra: list[str] = []
    for word in words:
        index = vocab.w2i(word=word)
        if index == UNK_ID:
            if word not in extra:
                extra.append(word)
            ids.append(len(vocab) + extra.index(word))
        else:
            ids.append(index)
    return ids, extra


def abstract2ids(
    *,
    words: list[str],
    vocab: Vocab,
    oovs: list[str],
) -> list[int]:
    """Map summary tokens to ids, using extended ids for source OOV tokens."""
    ids: list[int] = []
    for word in words:
        index = vocab.w2i(word=word)
        if index == UNK_ID and word in oovs:
            ids.append(len(vocab) + oovs.index(word))
        else:
            ids.append(index)
    return ids


def ids2words(*, ids: list[int], vocab: Vocab, oovs: list[str]) -> list[str]:
    words: list[str] = []
    for index in ids:
        if index < len(vocab):
            words.append(vocab.itos[index])
        else:
            offset = index - len(vocab)
            words.append(oovs[offset] if offset < len(oovs) else UNK)
    return words


@dataclass
class Example:
    """One encoded source and target pair."""

    enc: list[int]
    enc_ext: list[int]
    oovs: list[str]
    dec_in: list[int]
    target: list[int]
    src_len: int


def make_example(
    *,
    doc: SummaryInput,
    vocab: Vocab,
    max_enc_steps: int = 400,
    max_dec_steps: int = 100,
) -> Example:
    tokens = tokenize(text=doc["source"])[:max_enc_steps]
    enc = [vocab.w2i(word=word) for word in tokens]
    enc_ext, oovs = article2ids(words=tokens, vocab=vocab)

    summary_tokens = tokenize(text=doc["summary"])
    abstract_ext = abstract2ids(words=summary_tokens, vocab=vocab, oovs=oovs)
    abstract_ids = [index if index < len(vocab) else UNK_ID for index in abstract_ext]
    decoder_input = [START_ID] + abstract_ids
    target = list(abstract_ext)
    if len(decoder_input) > max_dec_steps:
        decoder_input = decoder_input[:max_dec_steps]
        target = target[:max_dec_steps]
    else:
        target.append(STOP_ID)

    return Example(
        enc=enc,
        enc_ext=enc_ext,
        oovs=oovs,
        dec_in=decoder_input,
        target=target,
        src_len=len(tokens),
    )


def collate(*, examples: list[Example], device: torch.device) -> dict:
    """Pad a batch and move tensors to the requested device."""
    batch_size = len(examples)
    max_src = max(len(example.enc) for example in examples)
    max_tgt = max(len(example.dec_in) for example in examples)

    enc = torch.full((batch_size, max_src), PAD_ID, dtype=torch.long)
    enc_ext = torch.full((batch_size, max_src), PAD_ID, dtype=torch.long)
    enc_mask = torch.zeros(batch_size, max_src)
    enc_len = torch.zeros(batch_size, dtype=torch.long)
    dec_in = torch.full((batch_size, max_tgt), PAD_ID, dtype=torch.long)
    target = torch.full((batch_size, max_tgt), PAD_ID, dtype=torch.long)
    dec_mask = torch.zeros(batch_size, max_tgt)

    for row, example in enumerate(examples):
        n_src = len(example.enc)
        n_tgt = len(example.dec_in)
        enc[row, :n_src] = torch.tensor(example.enc)
        enc_ext[row, :n_src] = torch.tensor(example.enc_ext)
        enc_mask[row, :n_src] = 1
        enc_len[row] = max(n_src, 1)
        dec_in[row, :n_tgt] = torch.tensor(example.dec_in)
        target[row, :n_tgt] = torch.tensor(example.target)
        dec_mask[row, :n_tgt] = 1

    batch = {
        "enc": enc,
        "ext": enc_ext,
        "emask": enc_mask,
        "elen": enc_len,
        "din": dec_in,
        "tgt": target,
        "dmask": dec_mask,
        "n_oov": max(len(example.oovs) for example in examples),
    }
    return {
        key: (value.to(device) if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }


def make_examples(
    *,
    docs: Mapping[str, Sequence[Record]],
    vocab: Vocab,
    cfg,
    splits: tuple[str, ...] | None = None,
) -> dict:
    """Encode selected splits with an existing vocabulary."""
    keys = splits or tuple(docs)
    return {
        split: [
            make_example(
                doc=doc,
                vocab=vocab,
                max_enc_steps=cfg.max_enc_steps,
                max_dec_steps=cfg.max_dec_steps,
            )
            for doc in docs[split]
        ]
        for split in keys
    }


def make_data(*, docs: Mapping[str, Sequence[Record]], cfg) -> tuple[Vocab, dict]:
    """Build the vocabulary and encoded examples for every split."""
    counter: Counter[str] = Counter()
    for doc in docs["train"]:
        counter.update(tokenize(text=doc["source"]))
        counter.update(tokenize(text=doc["summary"]))
    vocab = Vocab(counter=counter, max_size=cfg.vocab_size)
    print(f"vocab size: {len(vocab)} (corpus types: {len(counter)})")

    examples = make_examples(docs=docs, vocab=vocab, cfg=cfg)

    src_lengths = [example.src_len for example in examples["train"]]
    tgt_lengths = [len(example.target) for example in examples["train"]]
    print(
        f"train: mean src tokens {np.mean(src_lengths):.1f} "
        f"(max {max(src_lengths)}), "
        f"mean summary tokens {np.mean(tgt_lengths):.1f} "
        f"(max {max(tgt_lengths)})"
    )
    return vocab, examples
