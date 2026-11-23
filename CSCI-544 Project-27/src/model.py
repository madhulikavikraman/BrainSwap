"""
Transformer baseline models for 20 Newsgroups.

Architecture (matches the brief's spec exactly):
  - 4 pre-norm layers, width 256, 4 heads (d_head=64)
  - GELU feedforward 1024, learned positions, dropout 0.1
  - Encoder: masked mean pooling → N-class linear head
  - Decoder: causal self-attention → 20k next-token head (untied)

Every layer's attention module is swappable via the attn_cls argument
so Hopfield variants can reuse this code unchanged.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreNormTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout, attn_cls, attn_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = attn_cls(embed_dim, num_heads, dropout=dropout, **attn_kwargs)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Classification model: encoder + masked mean pooling + N-class head."""

    def __init__(
        self,
        vocab_size: int = 20_000,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        ffn_dim: int = 1024,
        max_len: int = 128,
        dropout: float = 0.1,
        num_classes: int = 20,
        attn_cls=None,
        attn_kwargs=None,
    ):
        super().__init__()
        if attn_cls is None:
            attn_cls = nn.MultiheadAttention
            attn_kwargs = {"batch_first": True}
        if attn_kwargs is None:
            attn_kwargs = {}

        self.embed_dim = embed_dim
        self.tok_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            PreNormTransformerBlock(embed_dim, num_heads, ffn_dim, dropout,
                                    attn_cls, attn_kwargs)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, 0.0, 0.02)
        nn.init.normal_(self.pos_emb.weight, 0.0, 0.02)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, 0.0, 0.02)

    def encode(self, tokens, pad_mask):
        """Return (B, embed_dim) pooled representation."""
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.dropout(self.tok_emb(tokens) + self.pos_emb(pos))
        for layer in self.layers:
            x = layer(x, key_padding_mask=pad_mask)
        x = self.norm(x)
        # Masked mean pooling: ignore padding positions
        mask = (~pad_mask).float().unsqueeze(-1)   # (B, T, 1) 1=real, 0=pad
        pooled = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        return pooled

    def forward(self, tokens, pad_mask):
        return self.head(self.encode(tokens, pad_mask))


class TransformerDecoder(nn.Module):
    """Language model: causal self-attention → 20k next-token head (untied)."""

    def __init__(
        self,
        vocab_size: int = 20_000,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        ffn_dim: int = 1024,
        max_len: int = 128,
        dropout: float = 0.1,
        attn_cls=None,
        attn_kwargs=None,
    ):
        super().__init__()
        if attn_cls is None:
            attn_cls = nn.MultiheadAttention
            attn_kwargs = {"batch_first": True}
        if attn_kwargs is None:
            attn_kwargs = {}

        self.max_len = max_len
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
        # Cache causal mask on first forward
        self._causal_cache: dict = {}

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, 0.0, 0.02)
        nn.init.normal_(self.pos_emb.weight, 0.0, 0.02)
        nn.init.zeros_(self.lm_head.bias)
        nn.init.normal_(self.lm_head.weight, 0.0, 0.02)

    def _causal_mask(self, T, device):
        key = (T, device)
        if key not in self._causal_cache:
            m = torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)
            self._causal_cache[key] = m
        return self._causal_cache[key]

    def forward(self, tokens, pad_mask=None):
        """
        tokens:   (B, T) input token IDs (positions 1..T-1 for next-token targets)
        pad_mask: (B, T) True = padding (ignored in loss, still attends causally)
        returns:  (B, T, vocab_size) logits
        """
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.dropout(self.tok_emb(tokens) + self.pos_emb(pos))
        causal = self._causal_mask(T, tokens.device)
        for layer in self.layers:
            x = layer(x, attn_mask=causal)  # no padding mask in generation
        x = self.norm(x)
        return self.lm_head(x)


def build_encoder(cfg: dict, num_classes: int, attn_cls=None, attn_kwargs=None):
    """Convenience builder from a config dict."""
    return TransformerEncoder(
        vocab_size=cfg.get("vocab_size", 20_000),
        embed_dim=cfg.get("embed_dim", 256),
        num_heads=cfg.get("num_heads", 4),
        num_layers=cfg.get("num_layers", 4),
        ffn_dim=cfg.get("ffn_dim", 1024),
        max_len=cfg.get("max_len", 128),
        dropout=cfg.get("dropout", 0.1),
        num_classes=num_classes,
        attn_cls=attn_cls,
        attn_kwargs=attn_kwargs,
    )


def build_decoder(cfg: dict, attn_cls=None, attn_kwargs=None):
    return TransformerDecoder(
        vocab_size=cfg.get("vocab_size", 20_000),
        embed_dim=cfg.get("embed_dim", 256),
        num_heads=cfg.get("num_heads", 4),
        num_layers=cfg.get("num_layers", 4),
        ffn_dim=cfg.get("ffn_dim", 1024),
        max_len=cfg.get("max_len", 128),
        dropout=cfg.get("dropout", 0.1),
        attn_cls=attn_cls,
        attn_kwargs=attn_kwargs,
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
