"""
Test-set evaluation for Track B generation models.

Computes:
  - test perplexity (all tokens)
  - test perplexity excluding unk targets
  - unk rate in greedy decodes from the first 50 val articles

Usage:
    python src/generation_test.py \\
        --results_base results/ --data_dir data/ \\
        --seeds 27 28 29

Output: results/generation-test-results.json
"""
import argparse
import glob
import json
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import NewsgroupDataset, load_split, load_vocab, load_config
from generation_models import RetrievalDecoder
from model import count_params

PAD_ID = 0
UNK_ID = 1
DECODE_SAMPLES = 50
DECODE_LEN = 64


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_run_dirs(results_base, variant_prefix, seed):
    pattern = os.path.join(results_base, f"gen-{variant_prefix}-seed{seed}-*")
    return sorted(glob.glob(pattern))


def compute_perplexity(model, loader, device):
    model.eval()
    total_nll = total_toks = 0
    nonunk_nll = nonunk_toks = 0
    crit = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="sum")
    with torch.no_grad():
        for tokens, _, _ in loader:
            tokens = tokens.to(device)
            inp, tgt = tokens[:, :-1], tokens[:, 1:]
            logits = model(inp)
            B, Tm, V = logits.shape
            flat_l = logits.reshape(-1, V)
            flat_t = tgt.reshape(-1)
            nll = crit(flat_l, flat_t)
            total_nll += nll.item()
            total_toks += (flat_t != PAD_ID).sum().item()

            nu_mask = (flat_t != UNK_ID) & (flat_t != PAD_ID)
            if nu_mask.any():
                nonunk_nll += F.cross_entropy(
                    flat_l[nu_mask], flat_t[nu_mask], reduction="sum"
                ).item()
                nonunk_toks += nu_mask.sum().item()

    ppl = math.exp(total_nll / max(total_toks, 1))
    nu_ppl = math.exp(nonunk_nll / max(nonunk_toks, 1))
    return ppl, nu_ppl


def greedy_unk_rate(model, tokens_prefix, device, steps=DECODE_LEN):
    """Generate greedily from a prefix; return fraction of unk tokens generated."""
    model.eval()
    inp = tokens_prefix.to(device)  # (N, T)
    unk_count = total = 0
    with torch.no_grad():
        for _ in range(steps):
            logits = model(inp)         # (N, T+i, V)
            next_tok = logits[:, -1, :].argmax(-1, keepdim=True)  # (N, 1)
            unk_count += (next_tok == UNK_ID).sum().item()
            total += next_tok.numel()
            inp = torch.cat([inp, next_tok], dim=1)
    return unk_count / max(total, 1)


def load_model_from_dir(run_dir, vocab_size, device):
    cfg_path = os.path.join(run_dir, "run_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    model = RetrievalDecoder(
        vocab_size=vocab_size,
        embed_dim=cfg.get("embed_dim", 256),
        num_heads=cfg.get("num_heads", 4),
        num_layers=cfg.get("num_layers", 4),
        ffn_dim=cfg.get("ffn_dim", 1024),
        max_len=cfg.get("max_len", 128),
        dropout=cfg.get("dropout", 0.1),
        bank_size=cfg.get("bank_size", 0),
        learn_beta=cfg.get("learn_beta", False),
        steps=cfg.get("steps", 1),
    ).to(device)
    ckpt_path = os.path.join(run_dir, "best.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model, ckpt_path


def evaluate_seed(seed, results_base, data_dir, device):
    vocab = load_vocab(data_dir)
    dcfg = load_config(data_dir)
    vocab_size = len(vocab)

    test_texts, _, _ = load_split(data_dir, "test")
    val_texts, _, _ = load_split(data_dir, "val")
    dummy_lbl = [0] * len(test_texts)      # per-text, not reused across splits
    dummy_val_lbl = [0] * len(val_texts)

    test_ds = NewsgroupDataset(test_texts, dummy_lbl, vocab)
    val_ds = NewsgroupDataset(val_texts, dummy_val_lbl, vocab)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

    # Prefix for greedy decode: first DECODE_SAMPLES val articles, first 16 tokens
    val_prefixes = torch.stack([val_ds[i][0][:16] for i in range(
        min(DECODE_SAMPLES, len(val_ds))
    )])

    # Discover all variants
    VARIANT_PREFIXES = ["attn", "bank64", "bank64+beta", "bank64+beta+steps2"]
    results = {}
    for vpfx in VARIANT_PREFIXES:
        dirs = _find_run_dirs(results_base, vpfx, seed)
        if not dirs:
            continue
        run_dir = dirs[-1]
        model, ckpt = load_model_from_dir(run_dir, vocab_size, device)
        ppl, nu_ppl = compute_perplexity(model, test_loader, device)
        unk_rate = greedy_unk_rate(model, val_prefixes, device)
        results[vpfx] = {
            "test_ppl": ppl,
            "test_nonunk_ppl": nu_ppl,
            "greedy_unk_rate": unk_rate,
            "n_params": count_params(model),
            "ckpt_sha256": sha256_file(ckpt),
            "run_dir": run_dir,
        }
        print(f"  seed={seed} {vpfx:20s}: ppl={ppl:.2f} nu_ppl={nu_ppl:.2f} "
              f"unk_rate={unk_rate:.3f}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_base", default="results/")
    ap.add_argument("--data_dir", default="data/")
    ap.add_argument("--seeds", nargs="+", type=int, default=[27, 28, 29])
    ap.add_argument("--output", default="results/generation-test-results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import numpy as np

    all_results = {}
    for seed in args.seeds:
        all_results[seed] = evaluate_seed(seed, args.results_base, args.data_dir, device)

    # Aggregate across seeds
    variants = set(v for r in all_results.values() for v in r)
    summary = {}
    for v in sorted(variants):
        ppls = [all_results[s][v]["test_ppl"] for s in args.seeds if v in all_results[s]]
        nu_ppls = [all_results[s][v]["test_nonunk_ppl"] for s in args.seeds if v in all_results[s]]
        if ppls:
            summary[v] = {
                "per_seed_ppl": ppls,
                "mean_ppl": float(np.mean(ppls)),
                "std_ppl": float(np.std(ppls, ddof=1)) if len(ppls) > 1 else 0.0,
                "mean_nonunk_ppl": float(np.mean(nu_ppls)),
            }

    output = {"per_seed": all_results, "summary": summary}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")
    print(f"\nSummary:")
    for v, s in summary.items():
        print(f"  {v:25s}: ppl={s['mean_ppl']:.2f}±{s['std_ppl']:.2f}  "
              f"nu_ppl={s['mean_nonunk_ppl']:.2f}")


if __name__ == "__main__":
    main()
