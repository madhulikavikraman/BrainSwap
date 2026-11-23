"""
Generation models for Track B.

Extends TransformerDecoder with Hopfield bank options via CLI flags:
  --bank_size P    (default 0 → standard transformer)
  --learn_beta     (default off)
  --steps N        (default 1)

The baseline (bank_size=0, learn_beta=False, steps=1) is identical to the
standard TransformerDecoder in model.py.

The class is still named RetrievalDecoder for naming parity with the
existing codebase convention (which used RetrievalClassifier).
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from model import PreNormTransformerBlock, count_params
from hopfield_layers import MultiHeadHopfieldRetrieval


class RetrievalDecoder(nn.Module):
    """
    Causal language model with optional Hopfield bank.

    With bank_size=0, learn_beta=False, steps=1 this is the same model as
    TransformerDecoder in model.py (standard transformer).
    """

    def __init__(
        self,
        vocab_size: int = 20_000,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        ffn_dim: int = 1024,
        max_len: int = 128,
        dropout: float = 0.1,
        bank_size: int = 0,
        learn_beta: bool = False,
        steps: int = 1,
    ):
        super().__init__()
        self.max_len = max_len

        if bank_size == 0 and not learn_beta and steps == 1:
            # Standard attention path — use nn.MultiheadAttention for parity
            attn_cls = nn.MultiheadAttention
            attn_kwargs = {"batch_first": True}
        else:
            # Hopfield path
            attn_cls = MultiHeadHopfieldRetrieval
            attn_kwargs = {
                "bank_size": bank_size,
                "learn_beta": learn_beta,
                "steps": steps,
                "batch_first": True,
            }

        self.tok_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            PreNormTransformerBlock(embed_dim, num_heads, ffn_dim, dropout,
                                    attn_cls, attn_kwargs)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self._init_weights()
        self._causal_cache: dict = {}

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, 0.0, 0.02)
        nn.init.normal_(self.pos_emb.weight, 0.0, 0.02)
        nn.init.zeros_(self.lm_head.bias)
        nn.init.normal_(self.lm_head.weight, 0.0, 0.02)

    def _causal_mask(self, T, device):
        if (T, device) not in self._causal_cache:
            m = torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)
            self._causal_cache[(T, device)] = m
        return self._causal_cache[(T, device)]

    def forward(self, tokens, pad_mask=None):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.dropout(self.tok_emb(tokens) + self.pos_emb(pos))
        causal = self._causal_mask(T, tokens.device)
        for layer in self.layers:
            x = layer(x, attn_mask=causal)
        x = self.norm(x)
        return self.lm_head(x)


def build_decoder(args) -> RetrievalDecoder:
    return RetrievalDecoder(
        vocab_size=getattr(args, "vocab_size", 20_000),
        embed_dim=getattr(args, "embed_dim", 256),
        num_heads=getattr(args, "num_heads", 4),
        num_layers=getattr(args, "num_layers", 4),
        ffn_dim=getattr(args, "ffn_dim", 1024),
        max_len=getattr(args, "max_len", 128),
        dropout=getattr(args, "dropout", 0.1),
        bank_size=getattr(args, "bank_size", 0),
        learn_beta=getattr(args, "learn_beta", False),
        steps=getattr(args, "steps", 1),
    )
