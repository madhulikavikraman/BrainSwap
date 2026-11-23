"""
Generation (language model) training for Track B.

Trains to convergence (up to --max_epochs) with early stopping on val
perplexity (patience --patience epochs).

Usage:
    python src/train_generation.py --seed 27 --data_dir data/ \\
        --results_base results/ --bank_size 0     # attn (baseline)
    python src/train_generation.py --seed 27 --data_dir data/ \\
        --results_base results/ --bank_size 64    # bank64
    python src/train_generation.py --seed 27 --data_dir data/ \\
        --results_base results/ --bank_size 64 --learn_beta  # bank64+beta

The variant name is inferred from --bank_size / --learn_beta / --steps.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import NewsgroupDataset, load_split, load_vocab, load_config
from generation_models import RetrievalDecoder, build_decoder
from model import count_params


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_perplexity(model, loader, device, vocab_size, unk_id=1, pad_id=0):
    """Return (all_ppl, nonunk_ppl, unk_rate_in_greedy)."""
    model.eval()
    total_nll = 0.0
    total_toks = 0
    nonunk_nll = 0.0
    nonunk_toks = 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")

    with torch.no_grad():
        for tokens, pad_mask, _ in loader:
            tokens = tokens.to(device)   # (B, T)
            inp = tokens[:, :-1]         # positions 1..T-1 as context
            tgt = tokens[:, 1:]          # positions 2..T as targets
            tgt_pad = (tgt == pad_id)

            logits = model(inp)          # (B, T-1, vocab)
            B, Tm1, V = logits.shape

            # Flat for cross-entropy
            flat_logits = logits.reshape(-1, V)
            flat_tgt = tgt.reshape(-1)

            nll = criterion(flat_logits, flat_tgt)
            valid_toks = (~tgt_pad).sum().item()
            total_nll += nll.item()
            total_toks += valid_toks

            # Non-unk targets
            non_unk_mask = (flat_tgt != unk_id) & (flat_tgt != pad_id)
            if non_unk_mask.any():
                nll_nu = nn.functional.cross_entropy(
                    flat_logits[non_unk_mask],
                    flat_tgt[non_unk_mask],
                    reduction="sum"
                )
                nonunk_nll += nll_nu.item()
                nonunk_toks += non_unk_mask.sum().item()

    ppl = math.exp(total_nll / max(total_toks, 1))
    nu_ppl = math.exp(nonunk_nll / max(nonunk_toks, 1))
    return ppl, nu_ppl


def _variant_name(bank_size, learn_beta, steps):
    if bank_size == 0 and not learn_beta and steps == 1:
        return "attn"
    name = f"bank{bank_size}"
    if learn_beta:
        name += "+beta"
    if steps > 1:
        name += f"+steps{steps}"
    return name


def train(args):
    variant = _variant_name(args.bank_size, args.learn_beta, args.steps)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_dir = os.path.join(
        args.results_base,
        f"gen-{variant}-seed{args.seed}-{job_id}"
    )
    os.makedirs(run_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)

    vocab = load_vocab(args.data_dir)
    dcfg = load_config(args.data_dir)

    train_texts, _, _ = load_split(args.data_dir, "train")
    val_texts, _, _ = load_split(args.data_dir, "val")

    # For generation: labels don't matter — we use all tokens as language model data
    train_ds = NewsgroupDataset(train_texts, [0] * len(train_texts), vocab)
    val_ds = NewsgroupDataset(val_texts, [0] * len(val_texts), vocab)

    g = torch.Generator(); g.manual_seed(args.seed + 3000)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, generator=g
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)

    torch.manual_seed(args.seed)
    # Pass vocab_size as an attribute for build_decoder
    args.vocab_size = len(vocab)
    model = build_decoder(args).to(device)
    n_params = count_params(model)
    print(f"[gen-{variant}] seed={args.seed}, params={n_params:,}, device={device}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)   # ignore padding

    run_cfg = {**vars(args), "n_params": n_params, "variant": variant,
               "device": str(device), "job_id": job_id}
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)

    best_val_ppl = float("inf")
    no_improve = 0
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0
        for tokens, pad_mask, _ in train_loader:
            tokens = tokens.to(device)
            inp, tgt = tokens[:, :-1], tokens[:, 1:]
            optimizer.zero_grad()
            logits = model(inp)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        val_ppl, val_nu_ppl = compute_perplexity(model, val_loader, device, len(vocab))
        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d} | loss {avg_loss:.4f} | "
              f"val_ppl {val_ppl:.2f} | val_nu_ppl {val_nu_ppl:.2f} | {elapsed:.1f}s")

        row = {"epoch": epoch, "train_loss": avg_loss,
               "val_ppl": val_ppl, "val_nonunk_ppl": val_nu_ppl,
               "elapsed_s": elapsed}
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        torch.save(model.state_dict(), os.path.join(run_dir, "last.pt"))
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))
            no_improve = 0
            print(f"  → new best val ppl: {best_val_ppl:.2f}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  Early stop (patience {args.patience})")
                break

    best_sha = sha256_file(os.path.join(run_dir, "best.pt"))
    last_sha = sha256_file(os.path.join(run_dir, "last.pt"))
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump({"best_val_ppl": best_val_ppl, "variant": variant,
                   "best_pt_sha256": best_sha, "last_pt_sha256": last_sha}, f, indent=2)
    print(f"Done. best_val_ppl={best_val_ppl:.2f}, run_dir={run_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--data_dir", default="data/")
    ap.add_argument("--results_base", default="results/")
    ap.add_argument("--bank_size", type=int, default=0)
    ap.add_argument("--learn_beta", action="store_true")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--max_epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--ffn_dim", type=int, default=1024)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
