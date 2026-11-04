"""Causal MIDI Transformer with optional relative attention and piece progress.

The default architecture retains the original state dictionary. New experiments
can opt into relative distance bias and whole-piece duration conditioning.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .tokenizer import PAD, TOKENS


def _relative_gelu(value):
    # PyTorch 2.6's fused encoder evaluation mishandles learned float masks.
    # A public callable activation selects its normal, differentiable path;
    # the mathematical GELU and serialized parameters remain unchanged.
    return F.gelu(value)


class MusicTransformer(nn.Module):
    def __init__(
        self, context=1024, d_model=256, layers=4, heads=8, dropout=.15,
        progress_bins=0, relative_attention=False,
    ):
        super().__init__()
        if progress_bins != 0 and not 2 <= progress_bins <= 64:
            raise ValueError('progress_bins must be zero or between 2 and 64')
        self.config = dict(context=context, d_model=d_model, layers=layers, heads=heads, dropout=dropout)
        self.context, self.heads = context, heads
        self.progress_bins, self.relative_attention = progress_bins, relative_attention
        if progress_bins:
            self.config['progress_bins'] = progress_bins
            self.progress = nn.Embedding(progress_bins, d_model)
        if relative_attention:
            self.config['relative_attention'] = True
            # A learned bias for each causal token distance and attention head.
            # Shared across layers; this is not the Music Transformer skew kernel.
            self.relative_bias = nn.Embedding(context, heads)
        self.token = nn.Embedding(len(TOKENS), d_model, padding_idx=PAD)
        self.genre = nn.Embedding(3, d_model)
        positions = torch.arange(context, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000) / d_model))
        pe = torch.zeros(context, d_model)
        pe[:, 0::2] = torch.sin(positions * scale)
        pe[:, 1::2] = torch.cos(positions * scale)
        self.register_buffer('positions', pe, persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, 4 * d_model, dropout,
            activation=_relative_gelu if relative_attention else 'gelu',
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d_model, len(TOKENS), bias=False)
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        with torch.no_grad():
            self.token.weight[PAD].zero_()
            if relative_attention:
                self.relative_bias.weight.zero_()

    def forward(self, x, genre, progress_ids=None):
        if x.shape[1] > self.context:
            raise ValueError('Input exceeds trained context')
        embedded = self.token(x) * math.sqrt(self.token.embedding_dim)
        embedded = embedded + self.genre(genre).unsqueeze(1)
        if self.progress_bins:
            if progress_ids is None or progress_ids.shape != x.shape:
                raise ValueError('Progress-conditioned model needs one progress ID per input token')
            embedded = embedded + self.progress(progress_ids)
        causal = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device).triu(1)
        padding = x.eq(PAD)
        if self.relative_attention:
            positions = torch.arange(x.shape[1], device=x.device)
            distances = (positions[:, None] - positions[None, :]).clamp(min=0)
            bias = self.relative_bias(distances).permute(2, 0, 1)
            mask = bias.masked_fill(causal.unsqueeze(0), float('-inf'))
            mask = mask.unsqueeze(0).expand(x.shape[0], -1, -1, -1).reshape(
                x.shape[0] * self.heads, x.shape[1], x.shape[1]
            )
            padding = torch.zeros_like(x, dtype=embedded.dtype).masked_fill(padding, float('-inf'))
        else:
            embedded = embedded + self.positions[:x.shape[1]]
            mask = causal
        hidden = self.encoder(self.dropout(embedded), mask=mask, src_key_padding_mask=padding)
        return self.output(hidden)
