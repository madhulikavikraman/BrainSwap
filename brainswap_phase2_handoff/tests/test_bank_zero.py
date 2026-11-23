"""
Checks for the bank-zero control (midterm feedback: separate temperature from memory).

Run from the repo root:
    python tests/test_bank_zero.py        (plain asserts, prints PASS lines)
    python -m pytest tests/test_bank_zero.py -q   (if pytest is installed)

What they show:
  1. With bank_size=0, fixed beta and one step, the Hopfield layer computes exactly
     what nn.MultiheadAttention computes, once the weights are the same (float64).
  2. A whole decoder built with --force_hopfield gives the same logits as the
     attn decoder when its weights are copied over, and has the same parameter count.
  3. --force_hopfield is what routes bank0 through the Hopfield module; without it
     bank0 is nn.MultiheadAttention (the "attn" run).
  4. The parameter differences match the midterm report: bank0+beta adds 16
     (4 layers x 4 heads), bank64 adds 131,072.
  5. The default Hopfield init is wider than PyTorch attention's; init="mha" matches it.
     This is the other difference between "attn" and the Hopfield runs, besides beta.
  6. Variant names for the 2x2 come out as bank0, bank0+beta, bank64, bank64+beta.
"""
import math
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from hopfield_layers import MultiHeadHopfieldRetrieval          # noqa: E402
from generation_models import RetrievalDecoder                   # noqa: E402
from model import count_params                                   # noqa: E402
from train_generation import _variant_name                       # noqa: E402

E, H, L = 64, 4, 2           # small sizes keep the float64 checks fast
SMALL = dict(vocab_size=97, embed_dim=E, num_heads=H, num_layers=L,
             ffn_dim=128, max_len=24, dropout=0.1)


def _copy_mha_into_hopfield(mha: nn.MultiheadAttention, hop: MultiHeadHopfieldRetrieval):
    e = mha.embed_dim
    with torch.no_grad():
        w, b = mha.in_proj_weight, mha.in_proj_bias
        hop.q_proj.weight.copy_(w[:e]);        hop.q_proj.bias.copy_(b[:e])
        hop.k_proj.weight.copy_(w[e:2 * e]);   hop.k_proj.bias.copy_(b[e:2 * e])
        hop.v_proj.weight.copy_(w[2 * e:]);    hop.v_proj.bias.copy_(b[2 * e:])
        hop.out_proj.weight.copy_(mha.out_proj.weight)
        hop.out_proj.bias.copy_(mha.out_proj.bias)


def _copy_decoder(src: RetrievalDecoder, dst: RetrievalDecoder):
    """Copy an attn decoder into a force_hopfield decoder (same everything but attention)."""
    dst_sd = dst.state_dict()
    with torch.no_grad():
        for name, t in src.state_dict().items():
            if ".attn." not in name:
                dst_sd[name].copy_(t)
    for ls, ld in zip(src.layers, dst.layers):
        _copy_mha_into_hopfield(ls.attn, ld.attn)


def test_layer_bank0_fixed_beta_equals_mha():
    torch.manual_seed(0)
    mha = nn.MultiheadAttention(E, H, dropout=0.0, batch_first=True).double().eval()
    hop = MultiHeadHopfieldRetrieval(E, H, dropout=0.0, bank_size=0,
                                     learn_beta=False, steps=1).double().eval()
    _copy_mha_into_hopfield(mha, hop)
    # log_beta was created in float32; refill it in float64 so beta is exactly 1/sqrt(d)
    hop.log_beta.fill_(math.log(1.0 / math.sqrt(E // H)))
    B, T = 3, 11
    x = torch.randn(B, T, E, dtype=torch.float64)
    causal = torch.triu(torch.full((T, T), float("-inf"), dtype=torch.float64), diagonal=1)
    pad = torch.zeros(B, T, dtype=torch.bool)
    pad[0, -3:] = True                       # trailing padding only, so no row is all -inf
    pad[2, -5:] = True
    for kw in ({"attn_mask": causal}, {"key_padding_mask": pad},
               {"attn_mask": causal, "key_padding_mask": pad}):
        ref, _ = mha(x, x, x, need_weights=False, **kw)
        got, _ = hop(x, x, x, **kw)
        assert torch.allclose(ref, got, atol=1e-10, rtol=0), \
            f"max diff {(ref - got).abs().max().item():.3e} for {list(kw)}"


def test_force_hopfield_routes_bank0_through_hopfield():
    plain = RetrievalDecoder(**SMALL)
    forced = RetrievalDecoder(**SMALL, force_hopfield=True)
    assert all(isinstance(l.attn, nn.MultiheadAttention) for l in plain.layers)
    assert all(isinstance(l.attn, MultiHeadHopfieldRetrieval) for l in forced.layers)
    a = forced.layers[0].attn
    assert a.bank_size == 0 and a.steps == 1 and not a.learn_beta
    assert "log_beta" not in dict(a.named_parameters())       # fixed beta = buffer
    assert count_params(plain) == count_params(forced)


def test_decoder_bank0_equals_attn_end_to_end():
    # float64 throughout (the decoder builds its causal mask in the default dtype)
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(1)
        attn = RetrievalDecoder(**SMALL).eval()
        bank0 = RetrievalDecoder(**SMALL, force_hopfield=True).eval()
        _copy_decoder(attn, bank0)
        tokens = torch.randint(2, SMALL["vocab_size"], (4, 20))
        ref, got = attn(tokens), bank0(tokens)
    finally:
        torch.set_default_dtype(old)
    assert torch.allclose(ref, got, atol=1e-9, rtol=0), \
        f"max logit diff {(ref - got).abs().max().item():.3e}"


def test_parameter_differences_match_report():
    full = dict(vocab_size=20_000)             # the real decoder size
    attn = count_params(RetrievalDecoder(**full))
    bank0 = count_params(RetrievalDecoder(**full, force_hopfield=True))
    bank0_beta = count_params(RetrievalDecoder(**full, learn_beta=True))
    bank64 = count_params(RetrievalDecoder(**full, bank_size=64))
    bank64_beta = count_params(RetrievalDecoder(**full, bank_size=64, learn_beta=True))
    assert bank0 == attn
    assert bank0_beta - attn == 4 * 4                       # 4 layers x 4 heads
    assert bank64 - attn == 4 * 2 * 4 * 64 * 64             # 131,072
    assert bank64_beta - bank64 == 16
    print(f"    params: attn {attn:,} | bank0 {bank0:,} | bank0+beta {bank0_beta:,} | "
          f"bank64 {bank64:,} | bank64+beta {bank64_beta:,}")


def test_init_scales():
    torch.manual_seed(2)
    e = 256
    mha = nn.MultiheadAttention(e, 4, batch_first=True)
    hop_default = MultiHeadHopfieldRetrieval(e, 4)
    hop_mha = MultiHeadHopfieldRetrieval(e, 4, init="mha")
    qkv_xavier = math.sqrt(6 / (2 * e))       # per-projection xavier bound
    qkv_mha = math.sqrt(6 / (4 * e))          # xavier over the stacked (3E, E) weight
    out_default_linear = 1 / math.sqrt(e)     # nn.Linear default bound
    assert mha.in_proj_weight.abs().max() <= qkv_mha + 1e-6
    assert mha.out_proj.weight.abs().max() <= out_default_linear + 1e-6
    # default Hopfield init reaches past PyTorch's bounds ...
    assert hop_default.q_proj.weight.abs().max() > qkv_mha
    assert hop_default.out_proj.weight.abs().max() > out_default_linear
    # ... init="mha" stays inside them
    for p in (hop_mha.q_proj, hop_mha.k_proj, hop_mha.v_proj):
        assert p.weight.abs().max() <= qkv_mha + 1e-6
    assert hop_mha.out_proj.weight.abs().max() <= out_default_linear + 1e-6
    sd = lambda t: t.detach().std().item()
    print(f"    q/k/v weight std: PyTorch attn {sd(mha.in_proj_weight):.4f} | "
          f"Hopfield default {sd(hop_default.q_proj.weight):.4f} | "
          f"Hopfield init=mha {sd(hop_mha.q_proj.weight):.4f}")
    print(f"    out_proj std:     PyTorch attn {sd(mha.out_proj.weight):.4f} | "
          f"Hopfield default {sd(hop_default.out_proj.weight):.4f} | "
          f"Hopfield init=mha {sd(hop_mha.out_proj.weight):.4f}")
    print(f"    bounds: xavier per-proj {qkv_xavier:.4f}, stacked {qkv_mha:.4f}, "
          f"linear {out_default_linear:.4f}")


def test_variant_names():
    assert _variant_name(0, False, 1) == "attn"
    assert _variant_name(0, False, 1, force_hopfield=True) == "bank0"
    assert _variant_name(0, True, 1) == "bank0+beta"
    assert _variant_name(64, False, 1) == "bank64"
    assert _variant_name(64, True, 1) == "bank64+beta"
    assert _variant_name(0, False, 1, True, "mha") == "bank0+mhainit"
    assert _variant_name(64, True, 2) == "bank64+beta+steps2"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} checks passed")
