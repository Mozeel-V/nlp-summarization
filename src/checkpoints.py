"""Loading vocabularies and model checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from config import Config
from model import PointerGenerator, make_model
from vocab import Vocab


def load_vocab(*, path: Path) -> Vocab:
    """Load a vocabulary from a JSON list of tokens."""
    itos = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(itos, list) or not all(isinstance(item, str) for item in itos):
        raise ValueError(f"{path} is not a JSON list of strings.")
    return Vocab.from_json(itos=itos)


def load_model(
    *,
    checkpoint_path: Path,
    vocab: Vocab,
    cfg: Config,
    device: torch.device,
    weights: str = "auto",
) -> PointerGenerator:
    """Load a pointer-generator from a checkpoint file.

    ``weights`` selects ``state_dict``, ``best_state``, or ``model``. The default
    ``auto`` prefers a plain state dict, then the best validation weights, then
    the last iteration.
    """
    if weights not in {"auto", "best", "last"}:
        raise ValueError("weights must be 'auto', 'best', or 'last'.")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if "state_dict" in checkpoint:
        state, which = checkpoint["state_dict"], "state_dict"
    elif weights in {"auto", "best"} and checkpoint.get("best_state") is not None:
        state, which = checkpoint["best_state"], "best_state"
    else:
        state, which = checkpoint["model"], "model (last iteration)"

    if state["embedding.weight"].shape[0] != len(vocab):
        raise ValueError(
            "Vocabulary size does not match the checkpoint embedding. "
            f"vocab {len(vocab)}, checkpoint "
            f"{state['embedding.weight'].shape[0]}."
        )

    model = make_model(vocab_size=len(vocab), cfg=cfg, device=device)
    model.load_state_dict(state)
    model.eval()

    iteration = checkpoint.get("it")
    suffix = f" | iteration {iteration}" if iteration is not None else ""
    print(f"loaded {which} from {checkpoint_path}{suffix}")
    return model
