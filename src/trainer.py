"""Adagrad training loop with validation-based checkpoint selection."""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from model import make_model
from vocab import collate


def batch_stream(*, examples: list, rng, batch_size: int, device: torch.device):
    """Yield shuffled batches forever."""
    while True:
        permutation = rng.permutation(len(examples))
        for start in range(0, len(permutation) - batch_size + 1, batch_size):
            batch = [
                examples[index] for index in permutation[start : start + batch_size]
            ]
            yield collate(examples=batch, device=device)


def safe_save(*, payload: dict, path: Path) -> None:
    """Write a checkpoint through a temporary file."""
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


class Trainer:
    """Train and evaluate one phase of the pointer-generator."""

    def __init__(
        self,
        *,
        cfg,
        schedule,
        examples: dict,
        device: torch.device,
        out_dir: Path,
    ) -> None:
        self.cfg = cfg
        self.schedule = schedule
        self.examples = examples
        self.device = device
        self.out_dir = Path(out_dir)
        self.checkpoint_nocov = self.out_dir / "pg_nocov.pt"
        self.checkpoint_cov = self.out_dir / "pg_cov.pt"
        self.history_path = self.out_dir / "training_history.json"
        self.history = (
            json.loads(self.history_path.read_text())
            if (cfg.resume and self.history_path.exists())
            else []
        )

    @torch.inference_mode()
    def val_loss(self, *, model, examples: list, use_cov: bool) -> tuple[float, float]:
        model.eval()
        total_nll, total_cov, count = 0.0, 0.0, 0
        for start in range(0, len(examples), self.cfg.batch_size):
            part = examples[start : start + self.cfg.batch_size]
            nll, coverage_loss = model(
                collate(examples=part, device=self.device),
                use_cov,
            )
            total_nll += nll.item() * len(part)
            total_cov += coverage_loss.item() * len(part)
            count += len(part)
        model.train()
        return total_nll / count, total_cov / count

    def get_opt(self, *, model):
        return torch.optim.Adagrad(
            model.parameters(),
            lr=self.cfg.lr,
            initial_accumulator_value=self.cfg.adagrad_init_acc,
        )

    def run_phase(
        self,
        *,
        model,
        n_iters: int,
        use_cov: bool,
        tag: str,
        keep_best: bool,
        start_step: int,
    ):
        cfg = self.cfg
        history = self.history
        optimizer = self.get_opt(model=model)
        checkpoint_path = self.out_dir / f"{tag}_last.pt"

        iteration_start = 0
        best = float("inf")
        best_state = None
        running_average = None
        start_time = time.time()

        if cfg.resume and checkpoint_path.exists():
            blob = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            model.load_state_dict(blob["model"])
            optimizer.load_state_dict(blob["opt"])
            iteration_start = blob["it"]
            best = blob["best_val"]
            best_state = blob["best_state"]
            running_average = blob["run_avg"]
            history[:] = blob["history"]
            print(
                f"[{tag}] resuming from iteration " f"{iteration_start}/{n_iters}",
                flush=True,
            )

        batches = batch_stream(
            examples=self.examples["train"],
            rng=np.random.RandomState(cfg.seed + start_step + iteration_start),
            batch_size=cfg.batch_size,
            device=self.device,
        )
        model.train()

        for iteration in range(iteration_start + 1, n_iters + 1):
            batch = next(batches)
            nll, coverage_loss = model(batch, use_cov)
            total_loss = nll + cfg.cov_loss_wt * coverage_loss if use_cov else nll
            if not torch.isfinite(total_loss):
                raise RuntimeError("Loss is not finite.")

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.max_grad_norm,
            )
            optimizer.step()

            running_average = (
                nll.item()
                if running_average is None
                else 0.99 * running_average + 0.01 * nll.item()
            )
            global_step = start_step + iteration

            if iteration % 100 == 0:
                elapsed = time.time() - start_time
                print(
                    f"[{tag}] it {iteration}/{n_iters} "
                    f"(global {global_step}) nll {running_average:.3f} "
                    f"cov {coverage_loss.item():.3f} "
                    f"gnorm {gradient_norm:.2f} | {elapsed:.0f}s",
                    flush=True,
                )

            if iteration % self.schedule.eval_every == 0 or iteration == n_iters:
                validation_nll, validation_cov = self.val_loss(
                    model=model,
                    examples=self.examples["validation"],
                    use_cov=use_cov,
                )
                history.append(
                    {
                        "phase": tag,
                        "step": global_step,
                        "train_nll": running_average,
                        "val_nll": validation_nll,
                        "val_cov": validation_cov,
                    }
                )
                print(
                    f"[{tag}] >>> step {global_step}: "
                    f"val NLL {validation_nll:.4f}  "
                    f"val coverage-loss {validation_cov:.4f}",
                    flush=True,
                )
                if keep_best and validation_nll < best:
                    best = validation_nll
                    best_state = copy.deepcopy(model.state_dict())

                model.train()
                safe_save(
                    payload={
                        "model": model.state_dict(),
                        "opt": optimizer.state_dict(),
                        "it": iteration,
                        "best_val": best,
                        "best_state": best_state,
                        "run_avg": running_average,
                        "history": history,
                    },
                    path=checkpoint_path,
                )

        if keep_best and best_state is not None:
            model.load_state_dict(best_state)
            print(
                f"[{tag}] restored best checkpoint (val NLL {best:.4f})",
                flush=True,
            )
        return model

    def fit_nocov(self, *, model):
        if self.cfg.resume and self.checkpoint_nocov.exists():
            checkpoint = torch.load(
                self.checkpoint_nocov,
                map_location=self.device,
            )
            model.load_state_dict(checkpoint["state_dict"])
            print("loaded", self.checkpoint_nocov)
            return model

        self.run_phase(
            model=model,
            n_iters=self.schedule.phase1_iters,
            use_cov=False,
            tag="pg",
            keep_best=True,
            start_step=0,
        )
        safe_save(
            payload={"state_dict": model.state_dict()},
            path=self.checkpoint_nocov,
        )
        self.history_path.write_text(json.dumps(self.history, indent=1))
        return model

    def fit_cov(self, *, vocab_size: int):
        model = make_model(vocab_size=vocab_size, cfg=self.cfg, device=self.device)
        model.load_state_dict(
            torch.load(
                self.checkpoint_nocov,
                map_location=self.device,
            )["state_dict"]
        )
        if self.cfg.resume and self.checkpoint_cov.exists():
            model.load_state_dict(
                torch.load(
                    self.checkpoint_cov,
                    map_location=self.device,
                )["state_dict"]
            )
            print("loaded", self.checkpoint_cov)
            return model

        self.run_phase(
            model=model,
            n_iters=self.schedule.cov_iters,
            use_cov=True,
            tag="cov",
            keep_best=False,
            start_step=self.schedule.phase1_iters,
        )
        safe_save(
            payload={"state_dict": model.state_dict()},
            path=self.checkpoint_cov,
        )
        self.history_path.write_text(json.dumps(self.history, indent=1))
        return model

    def load_nocov(self, *, vocab_size: int):
        model = make_model(vocab_size=vocab_size, cfg=self.cfg, device=self.device)
        model.load_state_dict(
            torch.load(
                self.checkpoint_nocov,
                map_location=self.device,
            )["state_dict"]
        )
        return model

    def dump_vocab(self, *, vocab) -> None:
        (self.out_dir / "vocab.json").write_text(json.dumps(vocab.to_json()))
