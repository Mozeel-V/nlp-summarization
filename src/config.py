"""Shared paths, model settings, and the training schedule."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT: Path = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR: Path = ROOT / "checkpoints"
OUTPUT_DIR: Path = ROOT / "outputs"

PAD, UNK, START, STOP = "[PAD]", "[UNK]", "[START]", "[STOP]"
UNK_ID, PAD_ID, START_ID, STOP_ID = 0, 1, 2, 3

DATASET_NAME: str = "knkarthick/samsum"
SPLIT_NAMES: tuple[str, ...] = ("test", "train", "validation")


@dataclass
class Schedule:
    """Iteration counts for one training run."""

    iters_per_epoch: int
    total_iters: int
    cov_iters: int
    phase1_iters: int
    eval_every: int

    def describe(self, *, mode: str) -> str:
        epochs = self.total_iters / self.iters_per_epoch
        return (
            f"mode={mode}: {self.iters_per_epoch} it/epoch | "
            f"phase 1 (no coverage) {self.phase1_iters} it + "
            f"phase 2 (coverage) {self.cov_iters} it "
            f"= {self.total_iters} it (~{epochs:.1f} epochs)"
        )


@dataclass
class Config:
    """Pointer-generator hyperparameters.

    The defaults reproduce the reference configuration, with 33000 iterations
    without coverage followed by 3000 iterations with coverage.
    """

    train_mode: str = "iterations"
    iterations_total: int = 33_000
    iterations_cov: int = 3_000
    epochs_total: int = 3
    epochs_cov_frac: float = 0.10

    dataset: str = DATASET_NAME

    vocab_size: int = 50_000
    emb_dim: int = 128
    hid_dim: int = 256
    batch_size: int = 16
    lr: float = 0.15
    adagrad_init_acc: float = 0.1
    max_grad_norm: float = 2.0
    rand_unif_init_mag: float = 0.02
    trunc_norm_init_std: float = 1e-4
    cov_loss_wt: float = 1.0

    max_enc_steps: int = 400
    max_dec_steps: int = 100
    max_dec_steps_test: int = 120
    min_dec_steps: int = 10
    beam_size: int = 4

    seed: int = 42
    resume: bool = True

    def schedule(self, *, n_train: int) -> Schedule:
        iters_per_epoch = n_train // self.batch_size
        if self.train_mode == "iterations":
            total = self.iterations_total
            coverage = self.iterations_cov
        elif self.train_mode == "epochs":
            total = self.epochs_total * iters_per_epoch
            coverage = max(1, round(self.epochs_cov_frac * total))
        else:
            raise ValueError("train_mode must be 'iterations' or 'epochs'.")
        eval_every = (
            500 if self.train_mode == "iterations" else max(1, iters_per_epoch // 2)
        )
        return Schedule(
            iters_per_epoch=iters_per_epoch,
            total_iters=total,
            cov_iters=coverage,
            phase1_iters=total - coverage,
            eval_every=eval_every,
        )

    def hparams(self) -> dict[str, float]:
        return {
            "emb_dim": self.emb_dim,
            "hid_dim": self.hid_dim,
            "lr": self.lr,
            "adagrad_init_acc": self.adagrad_init_acc,
            "max_grad_norm": self.max_grad_norm,
            "cov_loss_wt": self.cov_loss_wt,
            "beam_size": self.beam_size,
            "max_enc_steps": self.max_enc_steps,
            "max_dec_steps": self.max_dec_steps,
            "max_dec_steps_test": self.max_dec_steps_test,
            "min_dec_steps": self.min_dec_steps,
        }


def set_seed(*, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(*, device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. Pass --device cpu."
        )
    return torch.device(device)


def prepare_output_dir(*, path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
