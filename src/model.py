"""Pointer-generator with a BiLSTM encoder and Bahdanau attention.

The model supports the copy mechanism and an optional coverage vector. The
architecture follows See et al., with the same parameter shapes as the released
checkpoints.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import UNK_ID


class PointerGenerator(nn.Module):
    """Sequence-to-sequence pointer-generator."""

    def __init__(
        self,
        *,
        vocab_size: int,
        emb_dim: int = 128,
        hidden_dim: int = 256,
        rand_unif_init_mag: float = 0.02,
        trunc_norm_init_std: float = 1e-4,
    ) -> None:
        super().__init__()
        self.V = vocab_size
        self.H = hidden_dim
        attention_dim = 2 * hidden_dim

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.encoder = nn.LSTM(
            emb_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.reduce_c = nn.Linear(2 * hidden_dim, hidden_dim)
        self.reduce_h = nn.Linear(2 * hidden_dim, hidden_dim)
        self.W_h = nn.Linear(attention_dim, attention_dim, bias=False)
        self.W_s = nn.Linear(attention_dim, attention_dim, bias=True)
        self.w_c = nn.Parameter(torch.empty(attention_dim))
        self.v = nn.Parameter(torch.empty(attention_dim))
        self.input_proj = nn.Linear(emb_dim + attention_dim, emb_dim)
        self.cell = nn.LSTMCell(emb_dim, hidden_dim)
        self.pgen_lin = nn.Linear(
            attention_dim + hidden_dim + hidden_dim + emb_dim,
            1,
        )
        self.out_proj = nn.Linear(hidden_dim + attention_dim, hidden_dim)
        self.vocab_proj = nn.Linear(hidden_dim, vocab_size)
        self._init_weights(
            rand_unif_init_mag=rand_unif_init_mag,
            trunc_norm_init_std=trunc_norm_init_std,
        )

    def _init_weights(
        self,
        *,
        rand_unif_init_mag: float,
        trunc_norm_init_std: float,
    ) -> None:
        def truncated_normal(tensor: torch.Tensor) -> None:
            nn.init.trunc_normal_(
                tensor,
                mean=0.0,
                std=trunc_norm_init_std,
                a=-2 * trunc_norm_init_std,
                b=2 * trunc_norm_init_std,
            )

        def glorot(layer: nn.Linear) -> None:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        truncated_normal(self.embedding.weight)
        for layer in (self.reduce_c, self.reduce_h, self.vocab_proj):
            truncated_normal(layer.weight)
            truncated_normal(layer.bias)

        for layer in (
            self.W_h,
            self.W_s,
            self.input_proj,
            self.pgen_lin,
            self.out_proj,
        ):
            glorot(layer)

        bound = math.sqrt(6.0 / (2 * self.W_h.in_features))
        nn.init.uniform_(self.v, -bound, bound)
        nn.init.uniform_(self.w_c, -bound, bound)

        for block in (self.encoder, self.cell):
            for parameter in block.parameters():
                nn.init.uniform_(
                    parameter,
                    -rand_unif_init_mag,
                    rand_unif_init_mag,
                )

        # Match the TensorFlow forget-bias initialization.
        with torch.no_grad():
            self.cell.bias_ih[self.H : 2 * self.H] += 1.0
            for name, parameter in self.encoder.named_parameters():
                if name.startswith("bias_ih"):
                    parameter[self.H : 2 * self.H] += 1.0

    def encode(
        self,
        enc_ids: torch.Tensor,
        enc_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.embedding(enc_ids)
        packed = pack_padded_sequence(
            embedded,
            enc_len.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoder_out, (hidden, cell) = self.encoder(packed)
        encoder_out, _ = pad_packed_sequence(
            encoder_out,
            batch_first=True,
            total_length=enc_ids.size(1),
        )
        decoder_hidden = F.relu(
            self.reduce_h(torch.cat([hidden[0], hidden[1]], -1))
        )
        decoder_cell = F.relu(
            self.reduce_c(torch.cat([cell[0], cell[1]], -1))
        )
        return (
            encoder_out,
            self.W_h(encoder_out),
            decoder_hidden,
            decoder_cell,
        )

    def step(
        self,
        y_prev: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        context: torch.Tensor,
        coverage: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_feat: torch.Tensor,
        encoder_mask: torch.Tensor,
        extended_ids: torch.Tensor,
        n_oov: int,
        use_cov: bool,
    ):
        """Run one decoder step for training or beam search."""
        y_prev = y_prev.masked_fill(y_prev >= self.V, UNK_ID)
        projected = self.input_proj(
            torch.cat([self.embedding(y_prev), context], -1)
        )
        h, c = self.cell(projected, (h, c))
        state = torch.cat([c, h], -1)

        features = encoder_feat + self.W_s(state).unsqueeze(1)
        if use_cov:
            features = features + coverage.unsqueeze(2) * self.w_c
        scores = (self.v * torch.tanh(features)).sum(-1)
        scores = scores.masked_fill(encoder_mask == 0, -1e30)
        attention = F.softmax(scores, dim=-1)
        context = torch.bmm(attention.unsqueeze(1), encoder_out).squeeze(1)

        p_gen = torch.sigmoid(
            self.pgen_lin(torch.cat([context, c, h, projected], -1))
        )
        output = self.out_proj(torch.cat([h, context], -1))
        vocab_dist = F.softmax(self.vocab_proj(output), dim=-1)
        mixed = torch.cat(
            [p_gen * vocab_dist, vocab_dist.new_zeros(vocab_dist.size(0), n_oov)],
            1,
        )
        final = mixed.scatter_add(
            1,
            extended_ids,
            (1.0 - p_gen) * attention,
        )
        return final, h, c, context, attention, p_gen, coverage + attention

    def forward(self, batch: dict, use_cov: bool) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_out, encoder_feat, hidden, cell = self.encode(
            batch["enc"],
            batch["elen"],
        )
        batch_size, source_len = batch["enc"].shape
        context = encoder_out.new_zeros(batch_size, encoder_out.size(2))
        coverage = encoder_out.new_zeros(batch_size, source_len)
        nll_steps: list[torch.Tensor] = []
        coverage_steps: list[torch.Tensor] = []

        for step in range(batch["din"].size(1)):
            final, hidden, cell, context, attention, _, next_cov = self.step(
                batch["din"][:, step],
                hidden,
                cell,
                context,
                coverage,
                encoder_out,
                encoder_feat,
                batch["emask"],
                batch["ext"],
                batch["n_oov"],
                use_cov,
            )
            gold = final.gather(1, batch["tgt"][:, step : step + 1]).squeeze(1)
            nll_steps.append(-torch.log(gold.clamp_min(1e-12)))
            coverage_steps.append(torch.minimum(attention, coverage).sum(1))
            coverage = next_cov

        mask = batch["dmask"]
        lengths = mask.sum(1)
        nll = ((torch.stack(nll_steps, 1) * mask).sum(1) / lengths).mean()
        coverage_loss = (
            (torch.stack(coverage_steps, 1) * mask).sum(1) / lengths
        ).mean()
        return nll, coverage_loss


def make_model(*, vocab_size: int, cfg, device: torch.device) -> PointerGenerator:
    """Construct the model from a :class:`Config`."""
    model = PointerGenerator(
        vocab_size=vocab_size,
        emb_dim=cfg.emb_dim,
        hidden_dim=cfg.hid_dim,
        rand_unif_init_mag=cfg.rand_unif_init_mag,
        trunc_norm_init_std=cfg.trunc_norm_init_std,
    )
    return model.to(device)
