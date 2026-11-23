"""
Modern Hopfield retrieval module (Phase 0).

MultiHeadHopfieldRetrieval: drop-in for nn.MultiheadAttention.
  - bank_size=0, learn_beta=False, steps=1  →  identical to standard MHA
  - bank_size=P > 0                         →  P learned K/V patterns appended
  - learn_beta=True                         →  per-head learnable log-beta
  - steps=N > 1                             →  N-1 extra Hopfield updates in key space

HopfieldPooling: S learned query patterns aggregate token states.
  Replaces masked mean pooling; with S=1 returns (B, embed_dim).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadHopfieldRetrieval(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bank_size: int = 0,
        learn_beta: bool = False,
        steps: int = 1,
        batch_first: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_head = embed_dim // num_heads
        self.bank_size = bank_size
        self.learn_beta = learn_beta
        self.steps = steps
        self.batch_first = batch_first
        self.dropout_p = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        log_beta_init = math.log(1.0 / math.sqrt(self.d_head))
        if learn_beta:
            self.log_beta = nn.Parameter(torch.full((num_heads,), log_beta_init))
        else:
            self.register_buffer("log_beta", torch.full((num_heads,), log_beta_init))

        if bank_size > 0:
            self.k_bank = nn.Parameter(torch.empty(num_heads, bank_size, self.d_head))
            self.v_bank = nn.Parameter(torch.empty(num_heads, bank_size, self.d_head))
            nn.init.normal_(self.k_bank, 0.0, 0.02)
            nn.init.normal_(self.v_bank, 0.0, 0.02)
        else:
            self.k_bank = None
            self.v_bank = None

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
                nn.init.zeros_(m.bias)

    def _make_mask(self, B, Tq, Tk, device, dtype, attn_mask, key_padding_mask):
        """Additive attention mask, shape broadcastable to (B, H, Tq, Tk+P)."""
        P = self.bank_size

        if attn_mask is not None and attn_mask.dtype == torch.bool:
            attn_mask = torch.zeros_like(attn_mask, dtype=dtype).masked_fill(
                attn_mask, float("-inf")
            )

        per_head = attn_mask is not None and attn_mask.dim() == 3

        if per_head:
            ctx = attn_mask.view(B, self.num_heads, Tq, Tk).to(dtype)
        else:
            ctx = torch.zeros(B, 1, Tq, Tk, device=device, dtype=dtype)
            if attn_mask is not None:
                ctx = ctx + attn_mask.unsqueeze(0).unsqueeze(0).to(dtype)

        if key_padding_mask is not None:
            pad = key_padding_mask.to(torch.bool).unsqueeze(1).unsqueeze(2)
            ctx = ctx.masked_fill(pad, float("-inf"))

        if P > 0:
            bank = torch.zeros(ctx.shape[:-1] + (P,), device=device, dtype=dtype)
            return torch.cat([ctx, bank], dim=-1)
        return ctx

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        attn_mask=None,
        need_weights: bool = False,
    ):
        if not self.batch_first:
            query, key, value = (x.transpose(0, 1) for x in (query, key, value))

        B, Tq, _ = query.shape
        Tk = key.shape[1]
        H, d = self.num_heads, self.d_head

        q = self.q_proj(query).view(B, Tq, H, d).transpose(1, 2)  # (B,H,Tq,d)
        k = self.k_proj(key).view(B, Tk, H, d).transpose(1, 2)    # (B,H,Tk,d)
        v = self.v_proj(value).view(B, Tk, H, d).transpose(1, 2)

        if self.k_bank is not None:
            kb = self.k_bank.unsqueeze(0).expand(B, -1, -1, -1)
            vb = self.v_bank.unsqueeze(0).expand(B, -1, -1, -1)
            k = torch.cat([k, kb], dim=2)   # (B,H,Tk+P,d)
            v = torch.cat([v, vb], dim=2)

        beta = self.log_beta.exp().view(1, H, 1, 1)
        mask = self._make_mask(B, Tq, Tk, query.device, query.dtype,
                               attn_mask, key_padding_mask)

        # Hopfield updates (steps-1 iterations in key space)
        xi = q
        for _ in range(self.steps - 1):
            scores = beta * torch.matmul(xi, k.transpose(-2, -1)) + mask
            xi = F.softmax(scores, dim=-1) @ k

        # Final readout
        scores = beta * torch.matmul(xi, k.transpose(-2, -1)) + mask
        w = F.softmax(scores, dim=-1)
        if self.training and self.dropout_p > 0.0:
            w = F.dropout(w, p=self.dropout_p)

        out = (w @ v).transpose(1, 2).contiguous().view(B, Tq, self.embed_dim)
        out = self.out_proj(out)

        if not self.batch_first:
            out = out.transpose(0, 1)
        return out, (w.mean(1) if need_weights else None)

    def extra_repr(self):
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"bank_size={self.bank_size}, learn_beta={self.learn_beta}, "
            f"steps={self.steps}"
        )


class HopfieldPooling(nn.Module):
    """S learned query patterns pool over token states, replacing masked mean pooling.

    Parameter count (embed_dim=256, S=1, num_heads=4):
      stored patterns: S * 256
      retrieval attn:  4 * (256*256 + 256)  [q,k,v,out projections]
      layer norm:      2 * 256
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_patterns: int = 1,
        dropout: float = 0.0,
        learn_beta: bool = False,
    ):
        super().__init__()
        self.num_patterns = num_patterns
        self.stored = nn.Parameter(torch.empty(num_patterns, embed_dim))
        nn.init.normal_(self.stored, 0.0, 0.02)
        self.attn = MultiHeadHopfieldRetrieval(
            embed_dim, num_heads, dropout=dropout, learn_beta=learn_beta
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, padding_mask=None):
        """
        x:            (B, T, embed_dim)
        padding_mask: (B, T)  True = padding, should be ignored
        returns:      (B, embed_dim) when num_patterns=1, else (B, S*embed_dim)
        """
        B = x.shape[0]
        q = self.stored.unsqueeze(0).expand(B, -1, -1)  # (B, S, d)
        x_norm = self.norm(x)
        out, _ = self.attn(q, x_norm, x_norm, key_padding_mask=padding_mask)
        if self.num_patterns == 1:
            return out.squeeze(1)   # (B, d)
        return out.reshape(B, -1)   # (B, S*d)

    def extra_repr(self):
        return (
            f"num_patterns={self.num_patterns}, "
            f"embed_dim={self.stored.shape[1]}"
        )
