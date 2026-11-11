"""Beam search and text summarization with a trained pointer-generator."""

from __future__ import annotations

import torch

from config import START_ID, STOP_ID
from model import PointerGenerator
from vocab import Example, Vocab, collate, ids2words, make_example


def _ngram_blocked(
    *,
    tokens: list[int],
    token: int,
    no_repeat_ngram: int,
) -> bool:
    """Return True if appending ``token`` would repeat an n-gram."""
    if no_repeat_ngram <= 0 or token == STOP_ID:
        return False
    sequence = tokens + [token]
    if len(sequence) < no_repeat_ngram:
        return False
    last = tuple(sequence[-no_repeat_ngram:])
    previous = {
        tuple(sequence[index : index + no_repeat_ngram])
        for index in range(len(sequence) - no_repeat_ngram)
    }
    return last in previous


@torch.inference_mode()
def beam_search(
    *,
    model: PointerGenerator,
    example: Example,
    device: torch.device,
    beam: int = 4,
    min_steps: int = 10,
    max_steps: int = 120,
    length_norm: str = "mean",
    length_alpha: float = 1.0,
    no_repeat_ngram: int = 0,
    use_cov: bool = False,
) -> list[int]:
    """Return the best hypothesis as extended-vocabulary token ids."""
    model.eval()
    batch = collate(examples=[example], device=device)
    encoder_out, encoder_feat, h0, c0 = model.encode(
        batch["enc"],
        batch["elen"],
    )
    source_len = encoder_out.size(1)
    hidden_size = encoder_out.size(2)
    n_oov = len(example.oovs)

    beams = [
        {
            "tokens": [START_ID],
            "logps": [0.0],
            "h": h0[0],
            "c": c0[0],
            "context": encoder_out.new_zeros(hidden_size),
            "coverage": encoder_out.new_zeros(source_len),
        }
    ]
    done: list[dict] = []
    step = 0

    def score(hypothesis: dict) -> float:
        length = len(hypothesis["tokens"])
        total = sum(hypothesis["logps"])
        if length_norm == "gnmt":
            return total / (((5.0 + length) / 6.0) ** length_alpha)
        return total / length

    while step < max_steps and len(done) < beam:
        pending = len(beams)
        previous = torch.tensor(
            [item["tokens"][-1] for item in beams],
            device=device,
        )
        h = torch.stack([item["h"] for item in beams])
        c = torch.stack([item["c"] for item in beams])
        context = torch.stack([item["context"] for item in beams])
        coverage = torch.stack([item["coverage"] for item in beams])

        final, h2, c2, context2, _, _, coverage2 = model.step(
            previous,
            h,
            c,
            context,
            coverage,
            encoder_out.expand(pending, -1, -1),
            encoder_feat.expand(pending, -1, -1),
            batch["emask"].expand(pending, -1),
            batch["ext"].expand(pending, -1),
            n_oov,
            use_cov,
        )

        values, indices = torch.log(final.clamp_min(1e-12)).topk(
            min(2 * beam, final.size(1)),
            dim=-1,
        )
        values = values.tolist()
        indices = indices.tolist()

        candidates = []
        for parent, item in enumerate(beams):
            for logp, token in zip(values[parent], indices[parent], strict=False):
                if _ngram_blocked(
                    tokens=item["tokens"],
                    token=token,
                    no_repeat_ngram=no_repeat_ngram,
                ):
                    continue
                length = len(item["tokens"]) + 1
                total = sum(item["logps"]) + logp
                if length_norm == "gnmt":
                    rank_score = total / (((5.0 + length) / 6.0) ** length_alpha)
                else:
                    rank_score = total / length
                candidates.append((parent, token, logp, rank_score))
        candidates.sort(key=lambda value: value[3], reverse=True)

        next_beams = []
        for parent, token, logp, _ in candidates:
            new_beam = {
                "tokens": beams[parent]["tokens"] + [token],
                "logps": beams[parent]["logps"] + [logp],
                "h": h2[parent],
                "c": c2[parent],
                "context": context2[parent],
                "coverage": coverage2[parent],
            }
            if token == STOP_ID:
                if step >= min_steps:
                    done.append(new_beam)
            else:
                next_beams.append(new_beam)
            if len(next_beams) == beam or len(done) == beam:
                break

        beams = next_beams
        step += 1
        if not beams:
            break

    if not done:
        done = beams
    if not done:
        return []
    best = max(done, key=score)
    ids = best["tokens"][1:]
    return [index for index in ids if index != STOP_ID]


def summarize_ids(
    *,
    model: PointerGenerator,
    example: Example,
    vocab: Vocab,
    device: torch.device,
    beam: int = 4,
    min_steps: int = 10,
    max_steps: int = 120,
    length_norm: str = "mean",
    length_alpha: float = 1.0,
    no_repeat_ngram: int = 0,
    use_cov: bool = False,
) -> str:
    """Decode one prepared example and return the summary text."""
    ids = beam_search(
        model=model,
        example=example,
        device=device,
        beam=beam,
        min_steps=min_steps,
        max_steps=max_steps,
        length_norm=length_norm,
        length_alpha=length_alpha,
        no_repeat_ngram=no_repeat_ngram,
        use_cov=use_cov,
    )
    words = ids2words(ids=ids, vocab=vocab, oovs=example.oovs)
    return " ".join(word for word in words if word != "<nl>")


def summarize_text(
    *,
    model: PointerGenerator,
    vocab: Vocab,
    text: str,
    device: torch.device,
    beam: int = 4,
    min_steps: int = 10,
    max_steps: int = 120,
    length_norm: str = "mean",
    length_alpha: float = 1.0,
    no_repeat_ngram: int = 0,
    use_cov: bool = False,
) -> str:
    """Summarize an arbitrary string, such as a set of extracted turns."""
    example = make_example(doc={"source": text, "summary": ""}, vocab=vocab)
    return summarize_ids(
        model=model,
        example=example,
        vocab=vocab,
        device=device,
        beam=beam,
        min_steps=min_steps,
        max_steps=max_steps,
        length_norm=length_norm,
        length_alpha=length_alpha,
        no_repeat_ngram=no_repeat_ngram,
        use_cov=use_cov,
    )
